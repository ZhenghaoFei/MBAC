import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers

# Model Based Actor-Critic Network
class GameACMBNetwork(object):
  def __init__(self,
               action_size):
    print "MBAC"

# function parameters
    with tf.name_scope('feature_extraction') as feature_scope:

        # feature extraction function
        self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16], name='Conv1')  # stride=4
        self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32], name='Conv2') # stride=2
        self.W_fc1, self.b_fc1 = self._fc_variable([2592, 256], name='fc1')

    with tf.name_scope('imagination') as imagine_scope:

        with tf.name_scope('model_function') as model_scope:
            # model function
            factor_num = 2048
            # W dec
            self.W_dec, self.b_dec = self._fc_variable([factor_num, 256], name='dec')
            # W enc
            self.W_enc, _ = self._fc_variable([256, factor_num], name='enc')
            # W action
            self.W_a, _ = self._fc_variable([action_size, factor_num], name='a')
            # W reward
            self.W_r, self.b_r = self._fc_variable([factor_num, 1], name='r')

        with tf.name_scope('value_function') as value_scope:
            # value function
            self.W_v, self.b_v = self._fc_variable([256, 1], name='v')

    with tf.name_scope('policy_function') as policy_scope:
        # policy function
        self.W_pi, self.b_pi = self._fc_variable([256, action_size], name='pi')


    # constants
    # self.g = tf.constant(0., dtype=tf.float32, name='return') # return
    self.gamma = tf.constant(0.99, dtype=tf.float32, name='gamma') # gamma

# network
    n = 3 # imagine depth
    k = 10 # imagine samples

    with tf.name_scope('state_input') as scope:

        # state (input)
        self.s = tf.placeholder("float", [None, 84, 84, 4])
        s_ = tf.tile(self.s, [k, 1, 1, 1]) # copy state for k times and make them a fake batch

    # feature extraction
    with tf.name_scope(feature_scope):
        # Batch size
        h_conv1 = tf.nn.relu(self._conv2d(self.s ,  self.W_conv1, 4) + self.b_conv1)
        h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)
        h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
        h_t = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)     

        # Batch size * samples
        h_conv1 = tf.nn.relu(self._conv2d(s_,  self.W_conv1, 4) + self.b_conv1)
        h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)
        h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
        f_t = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

    with tf.name_scope(imagine_scope):

        # imagination 
        for i in range(n):
            with tf.name_scope(policy_scope):
                # imagination policy
                pi_t =  tf.matmul(f_t, self.W_pi) + self.b_pi
                a_t = tf.multinomial(tf.log(pi_t), 1) # rollout action at acording to policy
                a_t_onehot = tf.one_hot(a_t, action_size, on_value=1.0, off_value=0.0, axis=-1)
                a_t_onehot = tf.reshape(a_t_onehot, [-1, action_size])
           
            with tf.name_scope(model_scope):
                # imagination model
                hidden_f_t = tf.multiply(tf.matmul(f_t, self.W_enc), tf.matmul(a_t_onehot, self.W_a))
                r_t = tf.matmul(hidden_f_t, self.W_r) + self.b_r
                f_t_next = tf.nn.relu(tf.matmul(hidden_f_t, self.W_dec) + self.b_dec)
                f_t = f_t_next
         
            with tf.name_scope('discounted_return') as return_scope:
                # discounted return
                if i == 0:
                    g_ = r_t
                else:
                    g_ += tf.pow(self.gamma, i) * r_t

        with tf.name_scope(return_scope):

            # last state value
            v_ = tf.matmul(f_t, self.W_v) + self.b_v
            g_ += tf.pow(self.gamma, n-1) * v_


    # expect return of fake batch
    with tf.name_scope('value_out') as scope:
        g_ = tf.reshape(g_, [-1, k, 1])
        self.sample_returns = g_
        self.v = tf.reduce_mean(g_, axis=1)
        self.v = tf.reshape(self.v, [-1] )

    # policy (output)
    with tf.name_scope(policy_scope):
        self.pi = tf.matmul(h_t, self.W_pi) + self.b_pi
        
    with tf.name_scope('policy_out') as scope:
        self.pi = tf.nn.softmax(self.pi)

    self.opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    print "pi", self.pi
    print "v", self.v
    self.train = self.opt.minimize(self.v+self.pi)


  def _fc_variable(self, weight_shape, name="Variables"):
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d), name='w_'+name)
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d), name='b_'+name)
    return weight, bias

  def _conv_variable(self, weight_shape, name="Variables"):
    w = weight_shape[0]
    h = weight_shape[1]
    input_channels  = weight_shape[2]
    output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d), name='w_'+name)
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d), name='b_'+name)
    return weight, bias

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

def count_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("total_parameters:", total_parameters)

def test():
    action_size = 16
    net = GameACMBNetwork(action_size)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        count_parameters()

        # tensorboard
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./', sess.graph)

        state = np.random.rand(10, 84, 84, 4)

        sess.run(init_op)
        v, sample_returns, _ = sess.run([net.v, net.sample_returns, net.train], feed_dict={net.s:state})
        print "sample_returns ", sample_returns
        print "value ", v

if __name__ == "__main__":
    test()



















