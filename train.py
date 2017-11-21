import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from grab_train_images import build_image_input, BATCH_SIZE, TRAIN_LEN
from model import CDNA
import tensorflow as tf

class Trainer:
    def __init__(self, rnn, state_predictor):
        self.rnn = rnn
        self.state_predictor = state_predictor
        print("Preparing to get data from tfrecord.")
        self.data_getter = build_image_input()
        sess = tf.InteractiveSession()
        tf.train.start_queue_runners(sess)
        sess.run(tf.global_variables_initializer())
        self.sess = sess

    def train(self):
        train_videos, train_states, train_actions = self.sess.run(self.data_getter)
        return

def train():
    a = build_image_input()
    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())
    train_videos, train_states, train_actions = sess.run(a)

    print(train_states[0])
    print(np.shape(train_states))
    print(np.shape(train_videos))
    print(train_videos[0][2][3][4][0])
    print(train_videos[1][2][3][4][0])

    train_videos, train_states, train_actions = sess.run(a)
    print(train_videos[0][2][3][4][0])
    print(train_videos[1][2][3][4][0])

    print(train_states[0])
    print(train_actions[0])

#train()

if __name__ == '__main__':
    rnn = CDNA()
    # On page 11: "The next state is predicted linearly from the current state and action."
    state_predictor = nn.Linear(10, 5)
    trainer = Trainer(rnn, state_predictor)
    while True:
        print("!")
        trainer.train()
