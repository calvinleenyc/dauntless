import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from grab_train_images import build_image_input, BATCH_SIZE, TRAIN_LEN

import tensorflow as tf

class Trainer:
    def __init__(self, rnn):
        self.rnn = rnn
        data_getter = build_image_input()
        sess = tf.InteractiveSession()
        tf.train.start_queue_runners(sess)
        sess.run(tf.global_variables_initializer())
        self.sess = sess

    def train():
        pass

def train():
    a = build_image_input()
    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())
    train_videos = sess.run(a)

    print(np.shape(train_videos))
    print(train_videos[0][2][3][4][0])
    print(train_videos[1][2][3][4][0])

    train_videos = sess.run(a)
    print(train_videos[0][2][3][4][0])
    print(train_videos[1][2][3][4][0])


train()
