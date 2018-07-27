import datetime
import os
import time
import numpy as np
import tensorflow as tf

import data_helper
from CNN_MODEL import CNN


def preprocess():
    # prepare data

    # =============================

    # load data
    print("load data ...")
    sentence_A, sentence_B, y = data_helper.load_data('data/SICK_data.txt')
    # load pre-trained word vector and build vocabulary.
    word_vector = data_helper.word_vector('data/glove.6B.100d.txt')
    max_document_length = max([len(x.split(' ')) for x in sentence_A + sentence_B])
    word_vector.vocab_processor.max_document_length = max_document_length

    sentence_A = np.array(list(word_vector.vocab_processor.transform(sentence_A)))
    sentence_B = np.array(list(word_vector.vocab_processor.transform(sentence_B)))

    # randomly shuffle the data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    A_shuffled = sentence_A[shuffle_indices]
    B_shuffled = sentence_B[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # split train/dev set
    dev_sample_index = -1*int(0.2 * float(len(y)))
    A_train, A_dev = A_shuffled[:dev_sample_index], A_shuffled[dev_sample_index:]
    B_train, B_dev = B_shuffled[:dev_sample_index], B_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    del A_shuffled, B_shuffled, y_shuffled

    print("Vocabulary size: {:d}".format(len(word_vector.vocab_processor.vocabulary_)))
    print("Train/dev split: {:d}".format(len(y_train), len(y_dev)))
    return A_train, B_train, A_dev, B_dev, y_train, y_dev, word_vector

def train(A_train, B_train, A_dev, B_dev, y_train, y_dev, word_vector):
    # Training
    # =====================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = CNN(embedding_size=word_vector.embedding_size,
                      sequence_len=word_vector.vocab_processor.max_document_length,
                      filter_sizes=[3, 4, 5],
                      num_filters=128,
                      word_vector=word_vector.data)

            # Define training procedures
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summary for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev Summaries
            dev_summary_op = tf.summary.merge([loss_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            # Write vocabulary
            word_vector.vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(A_batch, B_batch, y_batch):
                """
                single train step
                """
                feed_dict = {
                    cnn.input_A: A_batch,
                    cnn.input_B: B_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 0.6
                }

                _, step, summaries, loss = sess.run([train_op, global_step, train_summary_op, cnn.loss],
                                                    feed_dict=feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}".format(time_str, step, loss))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(A_batch, B_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_A: A_batch,
                    cnn.input_B: B_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }

                step, summaries, loss, pearson = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.pearson],
                    feed_dict=feed_dict
                )

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, pearson {:g}".format(time_str, step, loss, pearson))
                if writer:
                    writer.add_summary(summaries, step)

            # generate batches
            data_train = zip(A_train, B_train, y_train)
            batches_train = data_helper.batches_generate(list(data_train), epoch_size=200, batch_size=64)

            for batch in batches_train:
                A_batch, B_batch, y_batch = zip(*batch)
                train_step(A_batch, B_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % 100 == 0:
                    print("\n Evaluation:")
                    dev_step(A_dev, B_dev, y_dev,  writer=dev_summary_writer)
                    print("")
                if current_step % 100 == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

def main(argv = None):
    A_train, B_train, A_dev, B_dev, y_train, y_dev, word_vector = preprocess()
    train(A_train, B_train, A_dev, B_dev, y_train, y_dev, word_vector)

if __name__ == '__main__':
    tf.app.run()