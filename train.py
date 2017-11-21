import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from grab_train_images import build_image_input, BATCH_SIZE, TRAIN_LEN
from model import CDNA
import tensorflow as tf

lr_rate = 0.001

class Trainer:
    def __init__(self, rnn, state_predictor):
        self.rnn = rnn
        self.state_predictor = state_predictor
        print("Preparing to get data from tfrecord.")
        self.data_getter = build_image_input()
        self.test_data = build_image_input(train = False, novel = False) # Eventually, we'll want this to be True
        sess = tf.InteractiveSession()
        tf.train.start_queue_runners(sess)
        sess.run(tf.global_variables_initializer())
        self.sess = sess

        self.optimizer = torch.optim.Adam(rnn.parameters(), lr = lr_rate)
        self.writer = SummaryWriter()
        self.epoch = 0

    def wrap(array):
        return Variable(torch.FloatTensor(array))

    def train(self):
        epoch += 1
        videos, states, actions = self.sess.run(self.data_getter)
        stactions = np.concatenate([states, actions], axis = 2)
        
        # Spatial tiling of state and action, as described on p.5
        tiled = []
        for b in range(BATCH_SIZE):
            this_batch = []
            for t in range(TRAIN_LEN):
                spatial_tiling = np.tile(stactions[b, t, :], (8, 8, 1))
                assert(np.shape(spatial_tiling) == (8, 8, 10))
                this_batch.append(spatial_tiling)
            tiled.append(this_batch)
        tiled = np.array(tiled)
        
        # Need to rearrange [videos], so that channel comes before height, width
        videos = np.transpose(videos, axes = (0, 1, 4, 2, 3))
        
        hidden = rnn.initHidden(BATCH_SIZE)
        cell = rnn.initCell(BATCH_SIZE)

        for t in range(TRAIN_LEN):
            out, kernels, hidden, cell = rnn(wrap(videos[:, t, :, :, :]), wrap(tiled[:, t, :, :, :]), hidden, cell)
            predicted_state = state_predictor(wrap(stactions[:, t, :]))

            
            
        return

    def test(self):
        # p.5: We only get the agent's internal state at the beginning.
        videos, states, actions = self.sess.run(self.test_data)
        return
        
if __name__ == '__main__':
    rnn = CDNA()
    # On page 11: "The next state is predicted linearly from the current state and action."
    state_predictor = nn.Linear(10, 5)
    trainer = Trainer(rnn, state_predictor)

    
    while True:
        trainer.train()
