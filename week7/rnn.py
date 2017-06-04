import time
import numpy as np
import reader
import tensorflow as tf

flags = tf.flags

flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")

FLAGS = flags.FLAGS


def main(_):
    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data

    init_scale = 0.05
    batch_size = 20
    num_steps = 35
    size = 650
    vocab_size = 10000
    num_layers = 4
    max_grad_norm = 5
    max_max_epoch = 39
    lr_decay = 0.8
    max_epoch = 6
    learning_rate = 1.0

    with tf.Graph().as_default():

        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        # train_input = PTBInput(config=config, data=train_data, name="TrainInput")

        epoch_size = (len(train_data) // batch_size - 1) // num_steps
        input_data, targets = reader.ptb_producer(
            train_data, batch_size, num_steps, name="TrainInput")

        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            # m = PTBModel(is_training=True, config=config, input_=train_input)
            def lstm_cell():
                return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
            with tf.variable_scope("RNN"):
                cell = tf.contrib.rnn.MultiRNNCell(
                    [lstm_cell() for _ in range(num_layers)], state_is_tuple=True)

                _initial_state = cell.zero_state(batch_size, tf.float32)

                embedding = tf.get_variable(
                    "embedding", [vocab_size, size], dtype=tf.float32)
                inputs = tf.nn.embedding_lookup(embedding, input_data)

                outputs = []
                state = _initial_state
                for time_step in range(num_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(inputs[:, time_step, :], state)
                    outputs.append(cell_output)

            output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
            softmax_w = tf.get_variable(
                "softmax_w", [size, vocab_size], dtype=tf.float32)
            softmax_b = tf.get_variable(
                "softmax_b", [vocab_size], dtype=tf.float32)
            logits = tf.matmul(output, softmax_w) + softmax_b
            loss = tf.contrib.seq2seq.sequence_loss(
                tf.reshape(logits, [-1, 1, vocab_size]),
                tf.reshape(targets, [-1, 1]),
                tf.ones([batch_size * num_steps, 1], dtype=tf.float32),
                average_across_batch=False)
            cost_op = tf.reduce_sum(loss) / batch_size
            final_state = state
            lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost_op, tvars),
                                            max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(lr)
            train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())

            new_lr = tf.placeholder(
                tf.float32, shape=[], name="new_learning_rate")
            lr_update = tf.assign(lr, new_lr)

            tf.summary.scalar("Training Loss", cost_op)
            tf.summary.scalar("Learning Rate", lr)

        sv = tf.train.Supervisor(logdir="./train_log")

        with sv.managed_session() as session:
            for i in range(max_max_epoch):
                lr_decay = lr_decay ** max(i + 1 - max_epoch, 0.0)
                session.run(lr_update, feed_dict={
                            new_lr: learning_rate * lr_decay})

                print("Epoch: %d Learning rate: %.3f" %
                        (i + 1, session.run(lr)))
                # train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                #                             verbose=True)
                start_time = time.time()
                costs = 0.0
                iters = 0
                state = session.run(_initial_state)
                fetches = {
                    "cost": cost_op,
                    "final_state": final_state
                }

                for step in range(epoch_size):
                    feed_dict = {}
                    for i, (c, h) in enumerate(_initial_state):
                        feed_dict[c] = state[i].c
                        feed_dict[h] = state[i].h
                    vals = session.run(fetches, feed_dict)
                    cost = vals["cost"]
                    state = vals["final_state"]

                    costs += cost
                    iters += num_steps

                    if step % (epoch_size // 10) == 10:
                        print("%.3f perplexity: %.3f speed: %.0f wps" %
                                (step * 1.0 / epoch_size, np.exp(costs / iters),
                                iters * batch_size / (time.time() - start_time)))

                print("Epoch: %d Train Perplexity: %.3f" %
                        (i + 1, np.exp(costs / iters)))


if __name__ == "__main__":
    main(None)
