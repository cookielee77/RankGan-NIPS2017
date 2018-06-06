import tensorflow as tf
import numpy as np


# An alternative to tf.nn.rnn_cell._linear function, which has been removed in Tensorfow 1.0.1
# The highway layer is borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn
def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term

def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


class Ranker(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, opt, FLAGS):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, opt.seq_len], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, opt.num_class], name="input_y")
        self.input_ref = tf.placeholder(tf.int32, [FLAGS.ref_size, opt.seq_len], name="input_ref")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.seq_len = opt.seq_len
        self.vocab_size = opt.vocab_size
        self.embedding_size = opt.rank_emb_dim
        self.filter_sizes = opt.rank_filter_sizes
        self.num_filters = opt.rank_num_filters
        self.ref_size = FLAGS.ref_size
        self.rank_lr = FLAGS.rank_lr
        self.gamma = opt.gamma


        self.feature = self.build_ranker(self.input_x, reuse = False)
        self.ref_feature = self.build_ranker(self.input_ref, reuse = True)
        self.rank_loss()


    def build_ranker(self, input_x, reuse = False):
        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope('ranker'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            l2_loss = tf.constant(0.0)
            # Embedding layer
            with tf.device('/cpu:0'), tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("W", [self.vocab_size, self.embedding_size], "float32", random_uniform_init)
                embedded_chars = tf.nn.embedding_lookup(word_emb_W, input_x)
                embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.embedding_size, 1, num_filter]
                    W = tf.get_variable("W", filter_shape, "float32", random_uniform_init)
                    b = tf.get_variable("b", [num_filter], "float32", random_uniform_init)
                    conv = tf.nn.conv2d(
                        embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.seq_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
            
            # Combine all the pooled features
            num_filters_total = sum(self.num_filters)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            # Add highway
            with tf.variable_scope("highway"):
                h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            with tf.variable_scope("dropout"):
                h_drop = tf.nn.dropout(h_highway, self.dropout_keep_prob)

        return h_drop


    def rank_loss(self):
        with tf.name_scope("output"):
            # ranking
            scores = tf.matmul(tf.nn.l2_normalize(self.feature, 1), tf.transpose(tf.nn.l2_normalize(self.ref_feature, 1)))
            self.scores = self.gamma * tf.reshape(tf.reduce_sum(scores, 1), [-1])
            self.rank_score = tf.reshape(tf.nn.softmax(self.scores), [-1])
            self.log_rank = tf.log(self.rank_score)

        # ranking loss
        with tf.name_scope("loss"):
            trans_y = tf.transpose(self.input_y)
            pos_ind = trans_y[1]
            neg_ind = trans_y[0]
            pos_loss = tf.reduce_sum(pos_ind*self.log_rank) / tf.reduce_sum(pos_ind)
            neg_loss = tf.reduce_sum(neg_ind*self.log_rank) / tf.reduce_sum(neg_ind)
            self.loss = -(pos_loss - neg_loss)


        self.params = [param for param in tf.trainable_variables() if 'ranker' in param.name]
        d_optimizer = tf.train.AdamOptimizer(self.rank_lr)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)
