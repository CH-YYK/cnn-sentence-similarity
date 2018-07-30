import tensorflow as tf

class CNN(object):
    """
    CNN for sentence similarity.

    setup an embedding layer but use a pre-trained embedding dictionary

    input_A: placeholder, array for integers that represent sentence A
    input_B: placeholder, array for integers that represent sentence B
    input_y: placeholder, the real relatedness-score
    dropout_keep_prob: placeholder, the probability that the values in dropout layer is kept.
    """
    def __init__(self, sequence_len, embedding_size, filter_sizes, num_filters, word_vector, l2_reg_lambda=0.0):

        # define placeholders
        self.input_A = tf.placeholder(tf.int32, [None, sequence_len], name='input_A')
        self.input_B = tf.placeholder(tf.int32, [None, sequence_len], name='input_B')
        self.input_y = tf.placeholder(tf.float32, [None, 1], 'input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # keeping track of l2_regularization loss(optional)
        l2_loss = tf.constant(0.0)

        # embedding layers
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            # initialize weights
            W = tf.Variable(dtype=tf.float32, initial_value=word_vector, name="embedding_weights", trainable=False)

            # embedding layer for both two types of sentences
            self.embedded_chars_A = tf.nn.embedding_lookup(W, self.input_A)
            self.embedded_chars_B = tf.nn.embedding_lookup(W, self.input_B)

            # concate two types of sentences and then expand dim.
            self.embedded_chars = tf.concat([self.embedded_chars_A, self.embedded_chars_B], axis=1)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, axis=-1)


        # convolutional_layers and max_pool
        pooled_output = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv_maxpool-%s' % filter_size):
                # filter shape
                filter_shape = [filter_size, embedding_size, 1, num_filters]    # height, width, thickness, amount

                # Initial value of filters: W
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                # Initial value of bias: b
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv')

                # apply non-linearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # pooling output
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_len*2-filter_size+1, 1, 1],   # pool filter output will be batch_size*1*1*1
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name='pool')
                pooled_output.append(pooled)

        # combine all pooled features, concatenate pooled features and spread it into a list
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_output, axis=3)      # 1*1*1*num_filters_total
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # scores and prediction
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, 1], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[1]), name='b')

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')

        # add loss accuracy
        with tf.name_scope("loss"):
            losses = tf.square(self.scores - self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # pearson correlation
        with tf.name_scope("pearson"):
            numerator = tf.reduce_mean(self.scores * self.input_y) - \
                        tf.reduce_mean(self.scores)*tf.reduce_mean(self.input_y)

            denominator = tf.sqrt(tf.reduce_mean(tf.square(self.scores)) - tf.square(tf.reduce_mean(self.scores))) * \
                          tf.sqrt(tf.reduce_mean(tf.square(self.input_y)) - tf.square(tf.reduce_mean(self.input_y)))

            self.pearson = numerator / denominator
