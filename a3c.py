# -*- coding: utf-8 -*-
import threading
import numpy as np

import signal
import random
import math
import os
import time

import tensorflow as tf

from game_ac_network import GameACFFNetwork, GameACLSTMNetwork, GameACMBFFNetwork, GameACMBLSTMNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE
from constants import PARALLEL_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import USE_LSTM
from constants import USE_MODEL
from constants import MODEL_LOSS

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

device = "/cpu:0"
if USE_GPU:
  device = "/gpu:0"

initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                    INITIAL_ALPHA_HIGH,
                                    INITIAL_ALPHA_LOG_RATE)

print("initial_learning_rate: ", initial_learning_rate)

global_t = 0

stop_requested = False

if USE_LSTM:
  if USE_MODEL:
    global_network = GameACMBLSTMNetwork(ACTION_SIZE, -1, device)
  else:
    global_network = GameACLSTMNetwork(ACTION_SIZE, -1, device)
else:
  if USE_MODEL:
    global_network = GameACMBFFNetwork(ACTION_SIZE, -1, device)
  else:
    global_network = GameACFFNetwork(ACTION_SIZE, -1, device)

training_threads = []

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = RMSP_ALPHA,
                              momentum = 0.0,
                              epsilon = RMSP_EPSILON,
                              clip_norm = GRAD_NORM_CLIP,
                              device = device)

for i in range(PARALLEL_SIZE):
  training_thread = A3CTrainingThread(i, global_network, initial_learning_rate,
                                      learning_rate_input,
                                      grad_applier, MAX_TIME_STEP,
                                      device = device)
  training_threads.append(training_thread)

# prepare session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))

init = tf.global_variables_initializer()
sess.run(init)

# summary for tensorboard
score_input = tf.placeholder(tf.int32)
model_state_loss_input = tf.placeholder(tf.float32)
model_reward_loss_input = tf.placeholder(tf.float32)
value_loss_input = tf.placeholder(tf.float32)
policy_loss_input = tf.placeholder(tf.float32)
total_loss_input = tf.placeholder(tf.float32)
entropy_input = tf.placeholder(tf.float32)
losses_input = model_state_loss_input, model_reward_loss_input, value_loss_input, policy_loss_input, total_loss_input, entropy_input

tf.summary.scalar("score", score_input)
if MODEL_LOSS:
  tf.summary.scalar("loss/model/state", model_state_loss_input)
  tf.summary.scalar("loss/model/reward", model_reward_loss_input)
  tf.summary.scalar("loss/value", value_loss_input)
  tf.summary.scalar("loss/policy", policy_loss_input)
  tf.summary.scalar("loss/total", total_loss_input)
  tf.summary.scalar("policy_entropy", entropy_input)

summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

# init or load checkpoint with saver
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
  tokens = checkpoint.model_checkpoint_path.split("-")
  # set global step
  global_t = int(tokens[1])
  print(">>> global step set: ", global_t)
  # set wall time
  wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
  with open(wall_t_fname, 'r') as f:
    wall_t = float(f.read())
else:
  print("Could not find old checkpoint")
  # set wall time
  wall_t = 0.0


def train_function(parallel_index):
  global global_t
  
  training_thread = training_threads[parallel_index]
  # set start_time
  start_time = time.time() - wall_t
  training_thread.set_start_time(start_time)

  model_state_loss, model_reward_loss, value_loss, policy_loss, total_loss, entropy = 0, 0, 0, 0, 0, 0 # initial losses
  losses = model_state_loss, model_reward_loss, value_loss, policy_loss, total_loss, entropy 
  while True:
    if stop_requested:
      break
    if global_t > MAX_TIME_STEP:
      break

    diff_global_t, losses = training_thread.process(sess, global_t, summary_writer,
                                            summary_op, score_input, losses_input, losses)
    global_t += diff_global_t
    
    
def signal_handler(signal, frame):
  global stop_requested
  print('You pressed Ctrl+C!')
  stop_requested = True
  
train_threads = []
for i in range(PARALLEL_SIZE):
  train_threads.append(threading.Thread(target=train_function, args=(i,)))
  
signal.signal(signal.SIGINT, signal_handler)

# set start time
start_time = time.time() - wall_t

for t in train_threads:
  t.start()

print('Press Ctrl+C to stop')
signal.pause()

print('Now saving data. Please wait')
  
for t in train_threads:
  t.join()

if not os.path.exists(CHECKPOINT_DIR):
  os.mkdir(CHECKPOINT_DIR)  

# write wall time
wall_t = time.time() - start_time
wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
with open(wall_t_fname, 'w') as f:
  f.write(str(wall_t))

saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)

