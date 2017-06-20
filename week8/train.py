import tensorflow as tf

from attention_model import AttentionChatBotModel
from basic_model import BasicChatBotModel
from util import get_buckets
import config
import data
import numpy as np


def need_print_log(step):
    if step < 100:
        return step % 10 == 0
    else:
        return step % 200 == 0


def train(use_attention, num_steps=1000, ckpt_dir="./ckp-dir/", write_summary=True, tag=None):
    bucket_id = 0
    test_buckets, data_buckets, train_buckets_scale = get_buckets()

    if not use_attention:
        model = BasicChatBotModel(batch_size=config.BATCH_SIZE)
    else:
        model = AttentionChatBotModel(batch_size=config.BATCH_SIZE)
    model.build()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        log_root = "./logs/"
        exp_name = ("attention" if use_attention else "basic") + \
                   "-step_" + str(num_steps) + \
                   "-batch_" + str(config.BATCH_SIZE) + \
                   "-lr_" + str(config.LR)
        if tag:
            exp_name += "-" + tag
        summary_writer = tf.summary.FileWriter(log_root + exp_name, graph=sess.graph)
        sess.run(tf.global_variables_initializer())
        for step in range(num_steps + 1):
            encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(
                data_buckets[bucket_id], bucket_id, batch_size=config.BATCH_SIZE)
            decoder_lens = np.sum(np.transpose(np.array(decoder_masks), (1, 0)), axis=1)
            feed_dict = {model.encoder_inputs_tensor: encoder_inputs, model.decoder_inputs_tensor: decoder_inputs,
                         model.decoder_length_tensor: decoder_lens}
            output_logits, res_loss, _ = sess.run([model.final_outputs, model.loss, model.train_op],
                                                  feed_dict=feed_dict)

            if need_print_log(step):
                print("Iteration {} - loss:{}".format(step, res_loss))
                if write_summary:
                    summaries = sess.run(model.summaries, feed_dict=feed_dict)
                    summary_writer.add_summary(summaries, step)
                saver.save(sess, ckpt_dir + exp_name + "/checkpoints", global_step=step)


if __name__ == '__main__':
    train(False, num_steps=100, write_summary=True, tag="3_layers_with_weights")
