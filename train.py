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
        #self.writer = SummaryWriter()
        self.epoch = 0

    @staticmethod
    def normalize_and_downsample(videos):
        # videos.size() = BATCH_SIZE x TRAIN_LEN x 512 x 640 x 3
        videos = np.array(videos, dtype = np.float32)
        # Need to rearrange [videos], so that channel comes before height, width
        videos = np.transpose(videos, axes = (0, 1, 4, 2, 3))

        videos = torch.FloatTensor(videos)
        videos = F.max_pool3d(videos, (1, 8, 10))
        return videos / 256 - 0.5

    @staticmethod
    def slow_normalize_and_downsample(videos):
        videos = np.array(videos, dtype = np.float32)
        # Need to rearrange [videos], so that channel comes before height, width
        videos = np.transpose(videos, axes = (0, 1, 4, 2, 3))

        ans = np.zeros([BATCH_SIZE, TRAIN_LEN, 3, 64, 64])
        for b in range(BATCH_SIZE):
            for t in range(TRAIN_LEN):
                for c in range(3):
                    for i in range(64):
                        for j in range(64):
                            ans[b,t,c,i,j] = np.max([[videos[b, t, c, i * 8 + di, j * 10 + dj] for di in range(8)] for dj in range(10)])
                            
        ans = np.array(ans, dtype = np.float32)
        return torch.FloatTensor(ans) / 256 - 0.5

    @staticmethod
    def old_apply_kernels(imgs, kernels):
        # imgs has size (BATCH_SIZE, 3, 64, 64)
        # kernels has size (BATCH_SIZE, 10, 5, 5)
        # output has size (BATCH_SIZE, 11, 3, 64, 64)
        ans = []
        for b in range(BATCH_SIZE):
            # We pretend that the batch is the 3 input channels.
            transformed_images = F.conv2d(imgs[b, :, :, :].view([3, 1, 64, 64]), kernels[b].view([10, 1, 5, 5]), padding = 2) # 3 x 10 x 64 x 64
            transformed_images = torch.transpose(transformed_images, 0, 1) # 10 x 3 x 64 x 64
            # append original # TODO: Replace this with STATIC BACKGROUND
            transformed_images = torch.cat((transformed_images, imgs[b, :, :, :].view([1, 3, 64, 64])), 0) # 11 x 3 x 64 x 64
            ans.append(transformed_images)
        return torch.stack(ans)

    @staticmethod
    def apply_kernels(imgs, kernels):
        # imgs has size (BATCH_SIZE, 3, 64, 64)
        # kernels has size (BATCH_SIZE, 10, 5, 5)
        # output has size (BATCH_SIZE, 11, 3, 64, 64)
        ans = []
        for b in range(BATCH_SIZE):
            # We pretend that the batch is the 3 input channels.
            transformed_images = F.conv2d(imgs[b, :, :, :].view([3, 1, 64, 64]), torch.unsqueeze(kernels[b], dim = 1), padding = 2) # 3 x 10 x 64 x 64
            transformed_images = torch.transpose(transformed_images, 0, 1) # 10 x 3 x 64 x 64
            # append original # TODO: Replace this with STATIC BACKGROUND
            transformed_images = torch.cat((transformed_images, torch.unsqueeze(imgs[b, :, :, :], dim = 0)), 0) # 11 x 3 x 64 x 64
            ans.append(transformed_images)
        return torch.stack(ans)

    @staticmethod
    def expected_pixel(options, masks):
        # options has size (BATCH_SIZE, 11, 3, 64, 64)
        # masks has size (BATCH_SIZE, 11, 64, 64)
        # output has size (BATCH_SIZE, 3, 64, 64)
        ans = []
        for b in range(BATCH_SIZE):
            here = []
            for c in range(3):
                prediction = torch.sum(options[b, :, c, :, :] * masks[b], dim = 0)
                here.append(prediction)
            ans.append(torch.stack(here))
        return torch.stack(ans)

    def train(self):
        def wrap(array):
            return Variable(torch.FloatTensor(array))

        
        self.epoch += 1
        #videos, states, actions = self.sess.run(self.data_getter)
        videos = np.random.randn(BATCH_SIZE,TRAIN_LEN,512,640,3)
        states = np.random.randn(BATCH_SIZE,TRAIN_LEN,5)
        actions = np.random.randn(BATCH_SIZE, TRAIN_LEN,5)

        
        videos = Trainer.normalize_and_downsample(videos)
        
        # Each frame will now be processed separately
        videos = torch.unbind(videos, dim = 1)
                     
        # CONSIDER wrapping more things here instead of down there
        
        stactions = np.concatenate([states, actions], axis = 2)
        # Spatial tiling of state and action, as described on p.5
        tiled = []
        for b in range(BATCH_SIZE):
            this_batch = []
            for t in range(TRAIN_LEN):
                spatial_tiling = np.tile(stactions[b, t, :], (8, 8, 1))
                assert(np.shape(spatial_tiling) == (8, 8, 10))
                spatial_tiling = np.transpose(spatial_tiling)
                this_batch.append(spatial_tiling)
            tiled.append(this_batch)
        tiled = np.array(tiled) # maybe np.stack

        tiled = torch.FloatTensor(tiled)
        # Each frame will now be processed separately
        tiled = torch.unbind(tiled, dim = 1)
        
        hidden = self.rnn.initHidden(BATCH_SIZE)
        cell = self.rnn.initCell(BATCH_SIZE)

        self.optimizer.zero_grad()
        loss = 0
        state_prediction_loss = 0
        
        for t in range(TRAIN_LEN - 1):
            print(type(tiled[t]))
            masks, kernels, hidden, cell = self.rnn(videos[t], Variable(tiled[t]), hidden, cell)

            transformed_images = Trainer.apply_kernels(videos[t], kernels)

            predictions = Trainer.expected_pixel(transformed_images, masks)

            loss += self.loss_fn(predictions, videos[t + 1])
            
            predicted_state = self.state_predictor(wrap(stactions[:, t, :]))
            state_prediction_loss += self.loss_fn(predicted_state, wrap(states[:, t + 1, :]))


        loss.backward()
        state_prediction_loss.backward()
        self.optimizer.step()
        self.state_predict_optimizer.step()
        return loss

    def test(self):
        # p.5: We only get the agent's internal state at the beginning.
        videos, states, actions = self.sess.run(self.test_data)
        return


run_tests = False
if run_tests:
    run_all = False
    run_n_and_d = False
    if run_n_and_d or run_all:
        videos = np.random.randn(BATCH_SIZE,TRAIN_LEN,512,640,3)
        ans = Trainer.normalize_and_downsample(videos)
        ans2 = Trainer.slow_normalize_and_downsample(videos)
        diff = F.mse_loss(ans, ans2)
        print(diff)

    run_apply_kernels = True
    if run_apply_kernels or run_all:
        imgs = Variable(torch.FloatTensor(np.random.randn(BATCH_SIZE, 3, 64, 64)))
        kernels = Variable(torch.FloatTensor(np.random.randn(BATCH_SIZE, 10, 5, 5)))
        ans = Trainer.apply_kernels(imgs, kernels)
        ans2 = Trainer.old_apply_kernels(imgs, kernels)
        print(ans.size())
        diff = F.mse_loss(ans, ans2)
        print(diff)

    run_expected_pixel = True
    if run_expected_pixel or run_all:
        options = Variable(torch.FloatTensor(np.random.randn(BATCH_SIZE, 11, 3, 64, 64)))
        masks = Variable(torch.FloatTensor(np.random.randn(BATCH_SIZE, 11, 64, 64)))
        ans = Trainer.expected_pixel(options, masks)
        print(ans.size())
    
    
if __name__ == '__main__' and not run_tests:
    rnn = CDNA()
    # On page 11: "The next state is predicted linearly from the current state and action."
    state_predictor = nn.Linear(10, 5)
    trainer = Trainer(rnn, state_predictor)

    
    for i in range(3):
        print("HELLO!")
        trainer.train()
