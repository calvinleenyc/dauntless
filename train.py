import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from grab_train_images import build_image_input

import tensorflow as tf

TRAIN_LEN = 8

def train():
    a = build_image_input()
    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.initialize_all_variables())
    train_videos = sess.run(a)
    print(type(train_videos))
    print(len(train_videos))

train()
