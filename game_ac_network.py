# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from network_parameter import FACTOR_NUM
from network_parameter import FEATURE_SIZE
from network_parameter import IMAG_DEPTH, IMAG_SAMPLES
from network_parameter import ALPHA_MODEL_STATE_LOSS, ALPHA_MODEL_REWARD_LOSS
from constants import MODEL_LOSS

# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(object):
  def __init__(self,
               action_size,
               thread_index, # -1 for global               
               device="/cpu:0"):
    self._action_size = action_size
    self._thread_index = thread_index
    self._device = device    


  def prepare_loss(self, entropy_beta):
    with tf.device(self._device):

      with tf.name_scope('loss') as loss_scope:

        # taken action (input for policy)
        self.a = tf.placeholder("float", [None, self._action_size])

        # temporary difference (R-V) (input for policy)
        self.td = tf.placeholder("float", [None])

        # avoid NaN with clipping when value in pi becomes zero
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))
        
        # policy entropy
        self.entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)
        self.entropy_mean = tf.reduce_mean(self.entropy)

        # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
        self.policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.multiply( log_pi, self.a ), reduction_indices=1 ) * self.td + self.entropy * entropy_beta )

        # R (input for value)
        self.r = tf.placeholder("float", [None])
        
        # value loss (output)
        # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
        self.value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

        # model loss (optional, including state model loss and reward model loss)
        if MODEL_LOSS:
          print("model_loss on")

          # imagination based on current action to train the model (only valid for model loss = true)
          hidden_h_t = tf.multiply(tf.matmul(self.h_t, self.W_enc), tf.matmul(self.a, self.W_a))
          self.imag_r = tf.matmul(hidden_h_t, self.W_r) + self.b_r
          self.imag_nf = tf.nn.relu(tf.matmul(hidden_h_t, self.W_dec) + self.b_dec)

          # next state feature (the flowing state of input state)
          self.nf = tf.placeholder("float", [None, FEATURE_SIZE])

          # next imagination state
          # self.imag_nf = tf.placeholder("float", [None, FEATURE_SIZE])

          self.model_state_loss = 0.5 * tf.nn.l2_loss(self.nf - self.imag_nf)

          self.model_reward_loss = 0.5 * tf.nn.l2_loss(self.r - self.imag_r)
          
          self.model_loss = ALPHA_MODEL_STATE_LOSS * self.model_state_loss + ALPHA_MODEL_REWARD_LOSS * self.model_reward_loss

          # gradienet of policy, value and model are summed up
          self.total_loss = self.policy_loss + self.value_loss + self.model_loss

        else:
          print("model_loss off")

          # gradienet of policy and value are summed up
          self.total_loss = self.policy_loss + self.value_loss

  def run_policy_and_value(self, sess, s_t):
    raise NotImplementedError()
    
  def run_policy(self, sess, s_t):
    raise NotImplementedError()

  def run_value(self, sess, s_t):
    raise NotImplementedError()    

  def get_vars(self):
    raise NotImplementedError()

  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "GameACNetwork", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  # weight initialization based on muupan's code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
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

  def count_parameters(self):
      total_parameters = 0
      for variable in tf.trainable_variables():
          # shape is an array of tf.Dimension
          shape = variable.get_shape()
          variable_parametes = 1
          for dim in shape:
              variable_parametes *= dim.value
          total_parameters += variable_parametes
      print("total_parameters:", total_parameters)

# Actor-Critic FF Network
class GameACFFNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0"):
    GameACNetwork.__init__(self, action_size, thread_index, device)

    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16])  # stride=4
      self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32]) # stride=2

      self.W_fc1, self.b_fc1 = self._fc_variable([2592, 256])

      # weight for policy output layer
      self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size])

      # weight for value output layer
      self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])
    
      # state (input)
      self.s = tf.placeholder("float", [None, 84, 84, 4])

      h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 4) + self.b_conv1)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

      h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
      h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

      # policy (output)
      self.pi = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
      # value (output)
      v_ = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
      self.v = tf.reshape( v_, [-1] )

  def run_policy_and_value(self, sess, s_t):
    pi_out, v_out = sess.run( [self.pi, self.v], feed_dict = {self.s : [s_t]} )
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, s_t):
    pi_out = sess.run( self.pi, feed_dict = {self.s : [s_t]} )
    return pi_out[0]

  def run_value(self, sess, s_t):
    v_out = sess.run( self.v, feed_dict = {self.s : [s_t]} )
    return v_out[0]

  def get_vars(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3]

# Actor-Critic LSTM Network
class GameACLSTMNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0" ):
    GameACNetwork.__init__(self, action_size, thread_index, device)

    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16])  # stride=4
      self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32]) # stride=2
      
      self.W_fc1, self.b_fc1 = self._fc_variable([2592, 256])

      # lstm
      self.lstm = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

      # weight for policy output layer
      self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size])

      # weight for value output layer
      self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])

      # state (input)
      self.s = tf.placeholder("float", [None, 84, 84, 4])
    
      h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 4) + self.b_conv1)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

      h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
      h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)
      # h_fc1 shape=(5,256)

      h_fc1_reshaped = tf.reshape(h_fc1, [1,-1,256])
      # h_fc_reshaped = (1,5,256)

      # place holder for LSTM unrolling time step size.
      self.step_size = tf.placeholder(tf.float32, [1])

      self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, 256])
      self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, 256])
      self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0,
                                                              self.initial_lstm_state1)
      
      # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
      # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
      # Unrolling step size is applied via self.step_size placeholder.
      # When forward propagating, step_size is 1.
      # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
      lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                        h_fc1_reshaped,
                                                        initial_state = self.initial_lstm_state,
                                                        sequence_length = self.step_size,
                                                        time_major = False,
                                                        scope = scope)

      # lstm_outputs: (1,5,256) for back prop, (1,1,256) for forward prop.
      
      lstm_outputs = tf.reshape(lstm_outputs, [-1,256])

      # policy (output)
      self.pi = tf.nn.softmax(tf.matmul(lstm_outputs, self.W_fc2) + self.b_fc2)
      
      # value (output)
      v_ = tf.matmul(lstm_outputs, self.W_fc3) + self.b_fc3
      self.v = tf.reshape( v_, [-1] )

      scope.reuse_variables()
      self.W_lstm = tf.get_variable("basic_lstm_cell/kernel")
      self.b_lstm = tf.get_variable("basic_lstm_cell/bias")

      self.reset_state()
      
  def reset_state(self):
    self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 256]),
                                                        np.zeros([1, 256]))

  def run_policy_and_value(self, sess, s_t):
    # This run_policy_and_value() is used when forward propagating.
    # so the step size is 1.
    pi_out, v_out, self.lstm_state_out = sess.run( [self.pi, self.v, self.lstm_state],
                                                   feed_dict = {self.s : [s_t],
                                                                self.initial_lstm_state0 : self.lstm_state_out[0],
                                                                self.initial_lstm_state1 : self.lstm_state_out[1],
                                                                self.step_size : [1]} )
    # pi_out: (1,3), v_out: (1)
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, s_t):
    # This run_policy() is used for displaying the result with display tool.    
    pi_out, self.lstm_state_out = sess.run( [self.pi, self.lstm_state],
                                            feed_dict = {self.s : [s_t],
                                                         self.initial_lstm_state0 : self.lstm_state_out[0],
                                                         self.initial_lstm_state1 : self.lstm_state_out[1],
                                                         self.step_size : [1]} )
                                            
    return pi_out[0]

  def run_value(self, sess, s_t):
    # This run_value() is used for calculating V for bootstrapping at the 
    # end of LOCAL_T_MAX time step sequence.
    # When next sequcen starts, V will be calculated again with the same state using updated network weights,
    # so we don't update LSTM state here.
    prev_lstm_state_out = self.lstm_state_out
    v_out, _ = sess.run( [self.v, self.lstm_state],
                         feed_dict = {self.s : [s_t],
                                      self.initial_lstm_state0 : self.lstm_state_out[0],
                                      self.initial_lstm_state1 : self.lstm_state_out[1],
                                      self.step_size : [1]} )
    
    # roll back lstm state
    self.lstm_state_out = prev_lstm_state_out
    return v_out[0]

  def get_vars(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_lstm, self.b_lstm,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3]

# Model Based Actor-Critic FF Network 
class GameACMBFFNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0" ):
    GameACNetwork.__init__(self, action_size, thread_index, device)
    print "MBAC FF"

    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:

      # Function parameters
      with tf.name_scope('feature_extraction') as feature_scope:

          # feature extraction function
          self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16], name='Conv1')  # stride=4
          self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32], name='Conv2') # stride=2
          self.W_fc1, self.b_fc1 = self._fc_variable([2592, FEATURE_SIZE], name='fc1')

      with tf.name_scope('imagination') as imagine_scope:

          with tf.name_scope('model_function') as model_scope:
              # model function
              # W dec
              self.W_dec, self.b_dec = self._fc_variable([FACTOR_NUM, FEATURE_SIZE], name='dec')
              # W enc
              self.W_enc, _ = self._fc_variable([FEATURE_SIZE, FACTOR_NUM], name='enc')
              # W action
              self.W_a, _ = self._fc_variable([action_size, FACTOR_NUM], name='a')
              # W reward
              self.W_r, self.b_r = self._fc_variable([FACTOR_NUM, 1], name='r')

          with tf.name_scope('value_function') as value_scope:
              # value function
              self.W_v, self.b_v = self._fc_variable([FEATURE_SIZE, 1], name='v')

      with tf.name_scope('policy_function') as policy_scope:
          # policy function
          self.W_pi, self.b_pi = self._fc_variable([FEATURE_SIZE, action_size], name='pi')

      # constants
      self.gamma = tf.constant(0.99, dtype=tf.float32, name='gamma') # gamma

      # Network
      with tf.name_scope('state_input') as scope:

          # state (input)
          self.s = tf.placeholder("float", [None, 84, 84, 4])

      # feature extraction
      with tf.name_scope(feature_scope):

          # Batch size
          h_conv1 = tf.nn.relu(self._conv2d(self.s ,  self.W_conv1, 4) + self.b_conv1)
          h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)
          h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
          self.h_t  = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)     
          f_t = tf.tile(self.h_t, [IMAG_SAMPLES, 1]) # copy state for k times and make them a fake batch

      with tf.name_scope(imagine_scope):

          # imagination based on policy to evaluate value
          for i in range(IMAG_DEPTH):
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
              g_ += tf.pow(self.gamma, IMAG_DEPTH-1) * v_

      # expect return of fake batch
      with tf.name_scope('value_out') as scope:
          g_ = tf.reshape(g_, [-1, IMAG_SAMPLES, 1])
          self.sample_returns = g_
          self.v = tf.reduce_mean(g_, axis=1)
          self.v = tf.reshape(self.v, [-1] )

      # policy (output)
      with tf.name_scope(policy_scope):
          self.pi = tf.matmul(self.h_t, self.W_pi) + self.b_pi
          
      with tf.name_scope('policy_out') as scope:
          self.pi = tf.nn.softmax(self.pi)

  def run_policy_and_value(self, sess, s_t):
    pi_out, v_out = sess.run( [self.pi, self.v], feed_dict = {self.s : [s_t]} )
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, s_t):
    pi_out = sess.run( self.pi, feed_dict = {self.s : [s_t]} )
    return pi_out[0]

  def run_value(self, sess, s_t):
    v_out = sess.run( self.v, feed_dict = {self.s : [s_t]} )
    return v_out[0]

  def run_feature(self, sess, s_t):
    f_out = sess.run( self.h_t, feed_dict = {self.s : s_t} )
    return f_out

  def get_vars(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_dec, self.b_dec,
            self.W_enc, 
            self.W_a,
            self.W_r, self.b_r, 
            self.W_v, self.b_v,
            self.W_pi, self.b_pi]

# Model Based Actor-Critic LSTM Network
class GameACMBLSTMNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0" ):
    GameACNetwork.__init__(self, action_size, thread_index, device)
    print "MBAC LSTM"

    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:

      # Function parameters
      with tf.name_scope('feature_extraction') as feature_scope:

        # feature extraction function
        self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16], name='Conv1')  # stride=4
        self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32], name='Conv2') # stride=2
        self.W_fc1, self.b_fc1 = self._fc_variable([2592, FEATURE_SIZE], name='fc1')

        # lstm
        self.lstm = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

      with tf.name_scope('imagination') as imagine_scope:

        with tf.name_scope('model_function') as model_scope:
          # model function
          # W dec
          self.W_dec, self.b_dec = self._fc_variable([FACTOR_NUM, FEATURE_SIZE], name='dec')
          # W enc
          self.W_enc, _ = self._fc_variable([FEATURE_SIZE, FACTOR_NUM], name='enc')
          # W action
          self.W_a, _ = self._fc_variable([action_size, FACTOR_NUM], name='a')
          # W reward
          self.W_r, self.b_r = self._fc_variable([FACTOR_NUM, 1], name='r')

        with tf.name_scope('value_function') as value_scope:
          # value function
          self.W_v, self.b_v = self._fc_variable([FEATURE_SIZE, 1], name='v')

      with tf.name_scope('policy_function') as policy_scope:
          # policy function
          self.W_pi, self.b_pi = self._fc_variable([FEATURE_SIZE, action_size], name='pi')

      # Constants
      self.gamma = tf.constant(0.99, dtype=tf.float32, name='gamma') # gamma

      # Network
      with tf.name_scope('state_input') as state_scope:

        # state (input)
        self.s = tf.placeholder("float", [None, 84, 84, 4])

      # feature extraction
      with tf.name_scope(feature_scope):

        h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 4) + self.b_conv1)
        h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

        h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)
        # h_fc1 shape=(5,256)

        h_fc1_reshaped = tf.reshape(h_fc1, [1,-1,256])
        # h_fc_reshaped = (1,5,256)

        # place holder for LSTM unrolling time step size.
        self.step_size = tf.placeholder(tf.float32, [1])

        self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, 256])
        self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, 256])
        self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0,
                                                                self.initial_lstm_state1)
        
        # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
        # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
        # Unrolling step size is applied via self.step_size placeholder.
        # When forward propagating, step_size is 1.
        # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
        lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                          h_fc1_reshaped,
                                                          initial_state = self.initial_lstm_state,
                                                          sequence_length = self.step_size,
                                                          time_major = False,
                                                          scope = scope)

        # lstm_outputs: (1,5,256) for back prop, (1,1,256) for forward prop.
        
        self.h_t = lstm_outputs = tf.reshape(lstm_outputs, [-1,256])
        f_t = tf.tile(self.h_t, [IMAG_SAMPLES, 1]) # copy state for k times and make them a fake batch

      with tf.name_scope(imagine_scope):

        # imagination based on policy to evaluate value
        for i in range(IMAG_DEPTH):
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
            g_ += tf.pow(self.gamma, IMAG_DEPTH-1) * v_

      # expect return of fake batch
      with tf.name_scope('value_out') as value_scope:
          g_ = tf.reshape(g_, [-1, IMAG_SAMPLES, 1])
          self.sample_returns = g_
          self.v = tf.reduce_mean(g_, axis=1)
          self.v = tf.reshape(self.v, [-1] )

      # policy (output)
      with tf.name_scope(policy_scope):
          self.pi = tf.matmul(self.h_t, self.W_pi) + self.b_pi
          
      with tf.name_scope('policy_out') as policy_scope:
          self.pi = tf.nn.softmax(self.pi)

      # with tf.name_scope(scope):

      scope.reuse_variables()
      self.W_lstm = tf.get_variable("basic_lstm_cell/kernel")
      self.b_lstm = tf.get_variable("basic_lstm_cell/bias")

      self.reset_state()
      
  def reset_state(self):
    self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 256]),
                                                        np.zeros([1, 256]))

  def run_policy_and_value(self, sess, s_t):
    # This run_policy_and_value() is used when forward propagating.
    # so the step size is 1.
    pi_out, v_out, self.lstm_state_out = sess.run( [self.pi, self.v, self.lstm_state],
                                                   feed_dict = {self.s : [s_t],
                                                                self.initial_lstm_state0 : self.lstm_state_out[0],
                                                                self.initial_lstm_state1 : self.lstm_state_out[1],
                                                                self.step_size : [1]} )
    # pi_out: (1,3), v_out: (1)
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, s_t):
    # This run_policy() is used for displaying the result with display tool.    
    pi_out, self.lstm_state_out = sess.run( [self.pi, self.lstm_state],
                                            feed_dict = {self.s : [s_t],
                                                         self.initial_lstm_state0 : self.lstm_state_out[0],
                                                         self.initial_lstm_state1 : self.lstm_state_out[1],
                                                         self.step_size : [1]} )
                                            
    return pi_out[0]

  def run_value(self, sess, s_t):
    # This run_value() is used for calculating V for bootstrapping at the 
    # end of LOCAL_T_MAX time step sequence.
    # When next sequcen starts, V will be calculated again with the same state using updated network weights,
    # so we don't update LSTM state here.
    prev_lstm_state_out = self.lstm_state_out
    v_out, _ = sess.run( [self.v, self.lstm_state],
                         feed_dict = {self.s : [s_t],
                                      self.initial_lstm_state0 : self.lstm_state_out[0],
                                      self.initial_lstm_state1 : self.lstm_state_out[1],
                                      self.step_size : [1]} )
    
    # roll back lstm state
    self.lstm_state_out = prev_lstm_state_out
    return v_out[0]

  def run_feature(self, sess, s_t):
    f_out = sess.run( self.h_t,  feed_dict = {self.s : s_t,
                                      self.initial_lstm_state0 : self.lstm_state_out[0],
                                      self.initial_lstm_state1 : self.lstm_state_out[1],
                                      self.step_size : [1]} )
    return f_out

  def get_vars(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_lstm, self.b_lstm,
            self.W_dec, self.b_dec,
            self.W_enc, 
            self.W_a,
            self.W_r, self.b_r, 
            self.W_v, self.b_v,
            self.W_pi, self.b_pi]

def test():
    action_size = 16
    net = GameACMBLSTMNetwork(action_size, -1)
    start_lstm_state = net.lstm_state_out    

    init_op = tf.global_variables_initializer()
    net.prepare_loss(1)

    with tf.Session() as sess:
        net.count_parameters()

        # tensorboard
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./', sess.graph)

        state = np.random.rand(10, 84, 84, 4)

        sess.run(init_op)

        # v, sample_returns = sess.run([net.v, net.sample_returns], feed_dict={net.s:state})
  
        v, sample_returns = sess.run([net.v, net.sample_returns], feed_dict={net.s:state, net.initial_lstm_state:start_lstm_state, net.step_size:[10.0]})

        print "sample_returns ", sample_returns
        print "value ", v


        print

        f = net.run_feature(sess, state)
        print "f: ", f.shape

if __name__ == "__main__":
    test()



