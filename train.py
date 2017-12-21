import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from grab_train_images import build_image_input, BATCH_SIZE, TRAIN_LEN
from model import CDNA
import argparse
import tensorflow as tf

lr_rate = 0.001

class Trainer:
    def __init__(self, rnn, state_predictor, use_cuda):
        self.rnn = rnn
        self.state_predictor = state_predictor
        self.use_cuda = use_cuda
        print("Preparing to get data from tfrecord.")
        self.data_getter = build_image_input()
        self.test_data = build_image_input(train = False, novel = False)
        sess = tf.InteractiveSession()
        tf.train.start_queue_runners(sess)
        sess.run(tf.global_variables_initializer())
        self.sess = sess

        self.optimizer = torch.optim.Adam(rnn.parameters(), lr = lr_rate)
        self.state_predict_optimizer = torch.optim.Adam(state_predictor.parameters(), lr = lr_rate)
        self.writer = SummaryWriter()
        self.epoch = 0

    def normalize_and_downsample(self, videos):
        # videos.size() = -1 x TRAIN_LEN x 512 x 640 x 3
        videos = np.array(videos, dtype = np.float32)
        # Need to rearrange [videos], so that channel comes before height, width
        videos = np.transpose(videos, axes = (0, 1, 4, 2, 3))

        if self.use_cuda:
            videos = Variable(torch.cuda.FloatTensor(videos))
        else:
            videos = Variable(torch.FloatTensor(videos))
        videos = F.avg_pool3d(videos, (1, 8, 10))
        return videos / 256 - 0.5
    
    def make_batch(self):
        bg, videos, states, actions = self.sess.run(self.data_getter)
        small_videos = self.normalize_and_downsample(videos)
        # bg has size (BATCH_SIZE, 1, 512, 640, 3)
        small_bg = torch.squeeze(self.normalize_and_downsample(bg))
        
        videos = torch.unbind(small_videos, dim = 1)
        return videos, small_bg, states, actions
    
    def spatial_tiling(stactions):
        # Compute spatial tiling of state and action, as described on p.5
        tiled = []
        for b in range(BATCH_SIZE):
            this_batch = []
            for t in range(np.shape(stactions)[1]):
                spatial_tiling = np.tile(stactions[b, t, :], (8, 8, 1))
                assert(np.shape(spatial_tiling) == (8, 8, 10))
                spatial_tiling = np.transpose(spatial_tiling)
                this_batch.append(spatial_tiling)
            tiled.append(this_batch)
        tiled = np.array(tiled) # maybe np.stack

        if self.use_cuda:
            tiled = Variable(torch.cuda.FloatTensor(tiled))
        else:
            tiled = Variable(torch.FloatTensor(tiled))
        # Each frame will now be processed separately
        tiled = torch.unbind(tiled, dim = 1)
        return tiled
    
    def train(self):
        self.epoch += 1
        videos, bg, states, actions = self.make_batch()
        stactions = np.concatenate([states, actions], axis = 2)
        loss = 0
        
        self.optimizer.zero_grad()
        self.state_predict_optimizer.zero_grad()

        state_prediction_loss = 0
        
        tiled = spatial_tiling(stactions)
        hidden = self.rnn.initHidden(BATCH_SIZE)
        cell = self.rnn.initCell(BATCH_SIZE)
        
        for t in range(TRAIN_LEN - 1):
            prediction, hidden, cell = self.rnn(bg, videos[t], tiled[t], hidden, cell)
            loss += F.mse_loss(prediction, Variable(videos[t + 1]))
            
            predicted_state = self.state_predictor(Variable(torch.FloatTensor(stactions[:, t, :])))

            state_prediction_loss += F.mse_loss(predicted_state, Variable(torch.FloatTensor(states[:, t + 1, :])))

        loss.backward()
        state_prediction_loss.backward()
        self.optimizer.step()
        self.state_predict_optimizer.step()
        self.writer.add_scalar('state_prediction_loss', state_prediction_loss.data.cpu().numpy(), self.epoch)
        self.writer.add_scalar('loss', loss.data.cpu().numpy(), self.epoch)
        self.writer.add_scalar('log_loss', np.log(loss.data.cpu().numpy()), self.epoch)
        return loss

    def test(self):
        videos, bg, states, actions = self.make_batch()
        assert(np.shape(states)[1] == 20) # TEST_LEN
        
        # p.5: We only get the agent's internal state at the beginning.
        # For the rest, we must predict it.
        predicted_states = [states[:, 0, :]]
        for i in range(1, np.shape(states)[1]):
            prev_staction = np.concatenate((predicted_states[i-1], actions[:, i-1, :]), axis = 1)
            next_state = self.state_predictor(Variable(torch.FloatTensor(prev_staction))).data.numpy()
            predicted_states.append(next_state)
        states = np.stack(predicted_states, axis = 1)
        
        stactions = np.concatenate([states, actions], axis = 2)
        
        tiled = spatial_tiling(stactions)
        hidden = self.rnn.initHidden(BATCH_SIZE)
        cell = self.rnn.initCell(BATCH_SIZE)
        
        predictions = []
        
        for t in range(18):
            if t <= 1:
                video_input = videos[t] # allow it to see first two frames
            else:
                video_input = ans[-1].data
            prediction, hidden, cell = self.rnn(bg, video_input, tiled[t], hidden, cell)
            predictions.append(prediction)
        
        # re-group by original batch
        predictions = torch.stack(predictions, dim = 1)
        predictions = torch.unbind(predictions, dim = 0)

        for b in range(BATCH_SIZE):
            # send it to range (0,1)
            seq = vutils.make_grid(predictions[b].data + 0.5, nrow = 1)
            self.writer.add_image('test ' + str(b), seq, self.epoch)
        return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDNA Model Training')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    args = parser.parse_args()
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    if args.cuda:
        rnn = CDNA(use_cuda = True).cuda()

        # On page 11: "The next state is predicted linearly from the current state and action."
        state_predictor = nn.Linear(10, 5).cuda()
    else:
        rnn = CDNA(use_cuda = False)
        state_predictor = nn.Linear(10, 5)
        
    trainer = Trainer(rnn, state_predictor, args.cuda)

    
    while True:
        trainer.train()
        if trainer.epoch % 50 == 1:
            trainer.test()
