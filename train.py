import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

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
        self.test_data = build_image_input(train = False, novel = False)
        self.novel_test_data = build_image_input(train = False, novel = True)
        sess = tf.InteractiveSession()
        tf.train.start_queue_runners(sess)
        sess.run(tf.global_variables_initializer())
        self.sess = sess

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(rnn.parameters(), lr = lr_rate)
        self.state_predict_optimizer = torch.optim.Adam(state_predictor.parameters(), lr = lr_rate)
        self.writer = SummaryWriter()
        self.epoch = 0

    @staticmethod
    def normalize_and_downsample(videos):
        # videos.size() = -1 x TRAIN_LEN x 512 x 640 x 3
        videos = np.array(videos, dtype = np.float32)
        # Need to rearrange [videos], so that channel comes before height, width
        videos = np.transpose(videos, axes = (0, 1, 4, 2, 3))

        videos = torch.FloatTensor(videos)
        videos = F.avg_pool3d(videos, (1, 8, 10)).data
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
    def apply_kernels(bg, imgs, kernels):
        # imgs has size (BATCH_SIZE, 3, 64, 64)
        # kernels has size (BATCH_SIZE, 10, 5, 5)
        # output is a list of length BATCH_SIZE with arrays with size (11, 3, 64, 64)
        
        ans = []
        imgs = torch.unbind(imgs, dim = 0)
        kernels = torch.unbind(kernels, dim = 0)
        bg = torch.unbind(bg, dim = 0)
        for b in range(BATCH_SIZE):
            # We pretend that the batch is the 3 input channels.
            transformed_images = F.conv2d(torch.unsqueeze(imgs[b], dim = 1), torch.unsqueeze(kernels[b], dim = 1), padding = 2) # 3 x 10 x 64 x 64
            transformed_images = torch.transpose(transformed_images, 0, 1) # 10 x 3 x 64 x 64
            # append static background
            transformed_images = torch.cat((transformed_images, torch.unsqueeze(bg[b], dim = 0)), 0) # 11 x 3 x 64 x 64
            ans.append(transformed_images)
        return ans

    @staticmethod
    def expected_pixel(options, masks):
        # options is the output of apply_kernels
        # masks has size (BATCH_SIZE, 11, 64, 64)
        # output has size (BATCH_SIZE, 3, 64, 64)
        ans = []
        masks = torch.unbind(masks, dim = 0)
        for b in range(BATCH_SIZE):
            here = []
            for c in range(3):
                prediction = torch.sum(options[b][:, c, :, :] * masks[b], dim = 0)
                here.append(prediction)
            ans.append(torch.stack(here))
        return torch.stack(ans)

    @staticmethod
    def slow_expected_pixel(options, masks):
        ans = torch.zeros(BATCH_SIZE, 3, 64, 64)
        
        for b in range(BATCH_SIZE):
            for c in range(3):
                for i in range(64):
                    for j in range(64):
                        for q in range(11):
                            ans[b][c][i][j] += options.data[b, q, c, i, j] * masks.data[b, q, i, j]
        return ans

    def make_predictions(self, bg, videos, stactions, training):
        # NOTE: The variable [videos] has already been unbound in dimension 1, i.e. videos[t] has size BATCH_SIZE x 3 x 64 x 64.
        
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

        tiled = torch.FloatTensor(tiled)
        # Each frame will now be processed separately
        tiled = torch.unbind(tiled, dim = 1)
        
        hidden = self.rnn.initHidden(BATCH_SIZE)
        cell = self.rnn.initCell(BATCH_SIZE)

        ans = []
        for t in range(TRAIN_LEN - 1 if training else 5): # TODO: Should be a 20
            # If testing, give it only 2 frames to work with
            # Can also insert scheduled sampling here pretty easily, if desired
            video_input = videos[t] if training or t <= 1 else ans[-1].data
            masks, kernels, hidden, cell = self.rnn(Variable(video_input), Variable(tiled[t]), hidden, cell)

            transformed_images = Trainer.apply_kernels(Variable(bg), Variable(videos[t]), kernels)

            predictions = Trainer.expected_pixel(transformed_images, masks)
            ans.append(predictions)
        return ans
    
    def train(self):
        self.epoch += 1
        bg, videos, states, actions = self.sess.run(self.data_getter)
        small_videos = Trainer.normalize_and_downsample(videos)
        
        # bg has size (BATCH_SIZE, 1, 512, 640, 3)
        small_bg = torch.squeeze(Trainer.normalize_and_downsample(bg))
        del videos
        del bg
        # Each frame will now be processed separately
        videos = torch.unbind(small_videos, dim = 1)
        
        stactions = np.concatenate([states, actions], axis = 2)
        
        loss = 0
        
        self.optimizer.zero_grad()
        self.state_predict_optimizer.zero_grad()

        state_prediction_loss = 0
        
        predictions = self.make_predictions(small_bg, videos, stactions, training = True)
        
        for t in range(TRAIN_LEN - 1):
            loss += self.loss_fn(predictions[t], Variable(videos[t + 1], requires_grad = False))
            
            predicted_state = self.state_predictor(Variable(torch.FloatTensor(stactions[:, t, :])))

            state_prediction_loss += self.loss_fn(predicted_state, Variable(torch.FloatTensor(states[:, t + 1, :]), requires_grad = False))

        loss.backward()
        state_prediction_loss.backward()
        self.optimizer.step()
        self.state_predict_optimizer.step()
        self.writer.add_scalar('state_prediction_loss', state_prediction_loss.data.numpy(), self.epoch)
        self.writer.add_scalar('loss', loss.data.numpy(), self.epoch)
        self.writer.add_scalar('log_loss', np.log(loss.data.numpy()), self.epoch)
        return loss

    def test(self):
        bg, videos, states, actions = self.sess.run(self.test_data)
        assert(np.shape(states)[1] == 20) # TEST_LEN

        small_videos = Trainer.normalize_and_downsample(videos)
        small_bg = torch.squeeze(Trainer.normalize_and_downsample(bg)) # "abuse of notation"
        del videos
        del bg
        
        # p.5: We only get the agent's internal state at the beginning.
        # For the rest, we must predict it.
        predicted_states = [states[:, 0, :]]
        for i in range(1, np.shape(states)[1]):
            prev_staction = np.concatenate((predicted_states[i-1], actions[:, i-1, :]), axis = 1)
            next_state = self.state_predictor(Variable(torch.FloatTensor(prev_staction))).data.numpy()
            predicted_states.append(next_state)
        states = np.stack(predicted_states, axis = 1)
    
        # Each frame will now be processed separately
        videos = torch.unbind(small_videos, dim = 1)
        
        stactions = np.concatenate([states, actions], axis = 2)
        
        predictions = self.make_predictions(small_bg, videos, stactions, training = False)
        
        # re-group by original batch
        predictions = torch.stack(predictions, dim = 1)
        predictions = torch.unbind(predictions, dim = 0)

        for b in range(BATCH_SIZE):
            # send it to range (0,1)
            seq = vutils.make_grid(predictions[b].data + 0.5, nrow = 1)
            self.writer.add_image('test ' + str(b), seq, self.epoch)
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

    run_expected_pixel = True
    if run_expected_pixel or run_all:
        options = Variable(torch.FloatTensor(np.random.randn(BATCH_SIZE, 11, 3, 64, 64)))
        masks = Variable(torch.FloatTensor(np.random.randn(BATCH_SIZE, 11, 64, 64)))
        ans = Trainer.expected_pixel(options, masks)
        ans2 = Trainer.slow_expected_pixel(options, masks)
        diff = F.mse_loss(ans, ans2)
        print(diff)
        print(ans.size())
    
    
if __name__ == '__main__' and not run_tests:
    rnn = CDNA()
    # On page 11: "The next state is predicted linearly from the current state and action."
    state_predictor = nn.Linear(10, 5)
    trainer = Trainer(rnn, state_predictor)

    
    for i in range(200):
        print("HELLO!")
        print(trainer.train())
        if trainer.epoch % 10 == 1:
            trainer.test()
