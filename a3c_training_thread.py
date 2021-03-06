# -*- coding: utf-8 -*-
import numpy as np
import random
import time
import sys
# import matplotlib.pyplot as plt
from game_state import GameState
from game_state import ACTION_SIZE
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork, GameACMBFFNetwork, GameACMBLSTMNetwork
import tensorflow as tf

from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import USE_LSTM
from constants import USE_MODEL
from constants import MODEL_LOSS

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000

class A3CTrainingThread(object):
  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device):

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step

    if USE_LSTM:
      if USE_MODEL:
        self.local_network = GameACMBLSTMNetwork(ACTION_SIZE, thread_index, device)
      else:
        self.local_network = GameACLSTMNetwork(ACTION_SIZE, thread_index, device)
    else:
      if USE_MODEL:
        self.local_network = GameACMBFFNetwork(ACTION_SIZE, thread_index, device)
      else:
        self.local_network = GameACFFNetwork(ACTION_SIZE, thread_index, device)

    self.local_network.prepare_loss(ENTROPY_BETA)

    with tf.device(device):
      var_refs = [v._ref() for v in self.local_network.get_vars()]
      self.gradients = tf.gradients(
        self.local_network.total_loss, var_refs,
        gate_gradients=False,
        aggregation_method=None,
        colocate_gradients_with_ops=False)

      # print "var_refs: ", var_refs
      # print "gradients: ", self.gradients

    self.apply_gradients = grad_applier.apply_gradients(
      global_network.get_vars(),
      self.gradients )
      
    self.sync = self.local_network.sync_from(global_network)
    
    self.game_state = GameState(113 * thread_index)
    
    self.local_t = 0

    self.initial_learning_rate = initial_learning_rate

    self.episode_reward = 0

    # variable controling log output
    self.prev_local_t = 0

  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  def _record_score_model_loss_on(self, sess, summary_writer, summary_op, score_input, score, global_t, losses_input, losses):
    model_state_loss_input, model_reward_loss_input, value_loss_input, policy_loss_input, total_loss_input, entropy_input = losses_input
    model_state_loss, model_reward_loss, value_loss, policy_loss, total_loss, entropy = losses
    summary_str = sess.run(summary_op, feed_dict={
      score_input: score,
      model_state_loss_input: model_state_loss,
      model_reward_loss_input: model_reward_loss,
      value_loss_input: value_loss,
      policy_loss_input: policy_loss,
      total_loss_input: total_loss,
      entropy_input: entropy,
      })
    summary_writer.add_summary(summary_str, global_t)

    summary_writer.flush()

  def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      score_input: score
      })
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()

  def set_start_time(self, start_time):
    self.start_time = start_time

  def process(self, sess, global_t, summary_writer, summary_op, score_input, losses_input, losses):
    states = []
    actions = []
    rewards = []
    values = []
    steps = []

    last_state = [] # only useful in learning model
    last_step = []
    terminal_end = False

    # copy weights from shared to local
    sess.run( self.sync )

    start_local_t = self.local_t

    if USE_LSTM:
      start_lstm_state = self.local_network.lstm_state_out
    
    # t_max times loop
    for i in range(LOCAL_T_MAX):
      pi_, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
      action = self.choose_action(pi_)
      states.append(self.game_state.s_t)
      actions.append(action)
      values.append(value_)
      steps.append(i)

      if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
        print("pi={}".format(pi_))
        print(" V={}".format(value_))

      # process game
      self.game_state.process(action)

      # receive game result
      reward = self.game_state.reward
      terminal = self.game_state.terminal

      self.episode_reward += reward

      # clip reward
      rewards.append( np.clip(reward, -1, 1) )

      self.local_t += 1

      # s_t1 -> s_t
      self.game_state.update()
      
      if terminal:
        last_state.append(0 * self.game_state.s_t) # terminal state

        terminal_end = True
        print("score={}".format(self.episode_reward))

        if MODEL_LOSS:
          print losses
          self._record_score_model_loss_on(sess, summary_writer, summary_op, score_input,
                             self.episode_reward, global_t, losses_input, losses)
        else:
          self._record_score(sess, summary_writer, summary_op, score_input,
                             self.episode_reward, global_t)          
          
        self.episode_reward = 0
        self.game_state.reset()
        if USE_LSTM:
          self.local_network.reset_state()
        break

    R = 0.0
    if not terminal_end:
      last_state.append(self.game_state.s_t) # non terminal last state
      R = self.local_network.run_value(sess, self.game_state.s_t)

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()
    steps.reverse()

    batch_si = []
    batch_a = []
    batch_td = []
    batch_R = []


    # compute and accmulate gradients
    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + GAMMA * R
      td = R - Vi
      a = np.zeros([ACTION_SIZE])
      a[ai] = 1

      batch_si.append(si)
      batch_a.append(a)
      batch_td.append(td)
      batch_R.append(R)

    # print "batch_si states equal?  ", (batch_si == states)

    cur_learning_rate = self._anneal_learning_rate(global_t)

    # print("length of batch_si: ", len(batch_si))

    last_step.append(20)

    batch_nsi = last_state + batch_si   # next state batch
    del batch_nsi[-1]

    steps = last_step +steps
    del steps[-1]

    if MODEL_LOSS:
      # print("model_loss on")

      batch_nfi = self.local_network.run_feature(sess, batch_nsi)

      if USE_LSTM:
        batch_si.reverse()
        batch_a.reverse()
        batch_td.reverse()
        batch_R.reverse()
        steps.reverse()
        # print steps
        model_state_loss, model_reward_loss, value_loss, policy_loss, total_loss, entropy, _ = sess.run(
        [self.local_network.model_state_loss,self.local_network.model_reward_loss,self.local_network.value_loss,self.local_network.policy_loss,self.local_network.total_loss, self.local_network.entropy_mean, self.apply_gradients],
                  feed_dict = {
                    self.local_network.s: batch_si,
                    self.local_network.a: batch_a,
                    self.local_network.td: batch_td,
                    self.local_network.r: batch_R,
                    self.local_network.nf: batch_nfi,
                    self.local_network.initial_lstm_state: start_lstm_state,
                    self.local_network.step_size : [len(batch_a)],
                    self.learning_rate_input: cur_learning_rate } )
        losses = model_state_loss, model_reward_loss, value_loss, policy_loss, total_loss, entropy

      else:
        # print steps
        model_state_loss, model_reward_loss, value_loss, policy_loss, total_loss, entropy, _ = sess.run(
        [self.local_network.model_state_loss,self.local_network.model_reward_loss,self.local_network.value_loss,self.local_network.policy_loss,self.local_network.total_loss, self.local_network.entropy_mean, self.apply_gradients],
                  feed_dict = {
                    self.local_network.s: batch_si,
                    self.local_network.a: batch_a,
                    self.local_network.td: batch_td,
                    self.local_network.r: batch_R,
                    self.local_network.nf: batch_nfi,
                    self.learning_rate_input: cur_learning_rate} )    
        losses = model_state_loss, model_reward_loss, value_loss, policy_loss, total_loss, entropy
 
        # print model_state_loss, model_reward_loss, value_loss, policy_loss, entropy
    else:
      if USE_LSTM:
        batch_si.reverse()
        batch_a.reverse()
        batch_td.reverse()
        batch_R.reverse()
        sess.run( self.apply_gradients,
                  feed_dict = {
                    self.local_network.s: batch_si,
                    self.local_network.a: batch_a,
                    self.local_network.td: batch_td,
                    self.local_network.r: batch_R,
                    self.local_network.initial_lstm_state: start_lstm_state,
                    self.local_network.step_size : [len(batch_a)],
                    self.learning_rate_input: cur_learning_rate } )
      else:
        sess.run( self.apply_gradients,
                  feed_dict = {
                    self.local_network.s: batch_si,
                    self.local_network.a: batch_a,
                    self.local_network.td: batch_td,
                    self.local_network.r: batch_R,
                    self.learning_rate_input: cur_learning_rate} )
      
    if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
      self.prev_local_t += PERFORMANCE_LOG_INTERVAL
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
        global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    # losses = model_state_loss, model_reward_loss, value_loss, policy_loss, total_loss, entropy 
    return diff_local_t, losses
    
