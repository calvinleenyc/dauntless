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
        # self.test_data = build_image_input(train = False, novel = False) # Eventually, we'll want this to be True
        sess = tf.InteractiveSession()
        tf.train.start_queue_runners(sess)
        sess.run(tf.global_variables_initializer())
        self.sess = sess

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(rnn.parameters(), lr = lr_rate)
        self.state_predict_optimizer = torch.optim.Adam(state_predictor.parameters(), lr = lr_rate)
        self.writer = SummaryWriter()
        self.epoch = 0

    def train(self):
        def wrap(array):
            return Variable(torch.FloatTensor(array))

        
        self.epoch += 1
        videos, states, actions = self.sess.run(self.data_getter)
        # CONSIDER wrapping things here instead of down there
        
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

        loss = 0
        state_prediction_loss = 0
        for t in range(TRAIN_LEN - 1):
            masks, kernels, hidden, cell = rnn(wrap(videos[:, t, :, :, :]), wrap(tiled[:, t, :, :, :]), hidden, cell)

            
            for b in range(BATCH_SIZE):
                ########### SUBTLE CODE, PLEASE REVIEW ###############
                # We pretend that the batch is the 3 input channels.
                transformed_images = F.conv2d(videos[b, t, :, :, :].view([3, 1, 64, 64]), kernels.view([10, 1, 5, 5]), padding = 2) # 3 x 10 x 64 x 64
                transformed_images = torch.transpose(transformed_images, 0, 1) # 10 x 3 x 64 x 64
                # append original # TODO: Replace this with STATIC BACKGROUND
                transformed_images = torch.cat((transformed_images, videos[b, t, :, :, :].view([1, 10, 64, 64])), 0) # 11 x 3 x 64 x 64
                
                # Now need to take an average, as dictated by the mask
                # Potentially there's a more subtle way here, using broadcasting
                for c in range(3):
                    prediction = torch.sum(transformed_images[:, c, :, :] * masks, dim = 0)
                    loss += loss_fn(prediction, wrap(videos[b, t, c, :, :]))
                
            
            predicted_state = state_predictor(wrap(stactions[:, t, :]))
            state_prediction_loss += loss_fn(predicted_state, wrap(states[:, t, :]))


        loss.backward()
        state_prediction_loss.backward()
        self.optimizer.step()
        self.state_predict_optimizer.step()
        return loss

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
