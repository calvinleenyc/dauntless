import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from grab_train_images import build_image_input, BATCH_SIZE, TRAIN_LEN


class ConvLSTM(nn.Module):
    def __init__(self, sq_side, input_ch, hidden_ch, use_cuda):
        super(ConvLSTM, self).__init__()
        self.hidden_ch = hidden_ch
        self.sq_side = sq_side
        self.use_cuda = use_cuda

        self.i2F = nn.Conv2d(input_ch + hidden_ch, hidden_ch, kernel_size = 5, padding = 2)
        self.i2I = nn.Conv2d(input_ch + hidden_ch, hidden_ch, kernel_size = 5, padding = 2)
        self.i2O = nn.Conv2d(input_ch + hidden_ch, hidden_ch, kernel_size = 5, padding = 2)
        self.i2C = nn.Conv2d(input_ch + hidden_ch, hidden_ch, kernel_size = 5, padding = 2)
        
    def forward(self, input, hidden, cell):
        combined = torch.cat((input, hidden), 1)
        FG = F.sigmoid(self.i2F(combined))
        I = F.sigmoid(self.i2I(combined))
        O = F.sigmoid(self.i2O(combined))
        C = FG * cell + I * F.tanh(self.i2C(combined))
        H = O * F.tanh(C)
        return H, C

    def initHidden(self, batch_size):
        if self.use_cuda:
            return Variable(torch.zeros(batch_size, self.hidden_ch, self.sq_side, self.sq_side).cuda())
        else:
            return Variable(torch.zeros(batch_size, self.hidden_ch, self.sq_side, self.sq_side))

    def initCell(self, batch_size):
        if self.use_cuda:
            return Variable(torch.zeros(batch_size, self.hidden_ch, self.sq_side, self.sq_side).cuda())
        else:
            return Variable(torch.zeros(batch_size, self.hidden_ch, self.sq_side, self.sq_side))

def apply_kernels(bg, imgs, kernels):
    # should rename imgs to img
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

    
class CDNA(nn.Module):
    def __init__(self, use_cuda):
        super(CDNA, self).__init__()
        self.use_cuda = use_cuda
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 5, stride = 2, padding = 2)
        self.lstm1 = ConvLSTM(32, 32, 32, use_cuda = use_cuda)
        self.lstm2 = ConvLSTM(32, 32, 32, use_cuda = use_cuda)
        self.downsample23 = nn.Conv2d(32, 64, 2, stride = 2)
        self.lstm3 = ConvLSTM(16, 64, 64, use_cuda = use_cuda)
        self.lstm4 = ConvLSTM(16, 64, 64, use_cuda = use_cuda)
        self.downsample45 = nn.Conv2d(64, 128, 2, stride = 2)
        self.lstm5 = ConvLSTM(8, 138, 138, use_cuda = use_cuda)
        self.to_kernels = nn.Linear(138 * 8 * 8, 10 * 5 * 5)
        self.upsample56 = nn.ConvTranspose2d(138, 64, 2, stride = 2)
        self.lstm6 = ConvLSTM(16, 64, 64, use_cuda = use_cuda)
        self.upsample67 = nn.ConvTranspose2d(64 + 64, 32, 2, stride = 2)
        self.lstm7 = ConvLSTM(32, 32, 32, use_cuda = use_cuda)
        # the end of the diagram is ambiguous
        self.last_upsample = nn.ConvTranspose2d(32 + 32, 32, 2, stride = 2) 
        self.conv2 = nn.Conv2d(32, 11, kernel_size = 1)

        # For some reason, F.softmax(x, dim = 2) doesn't work on my machine,
        # so I use this instead: given a 4D tensor, it softmaxes dimension 1.
        self.softmax = nn.Softmax2d()
        
    def forward(self, bg, img, tiled, hiddens, cells):
        # input is preprocessed with numpy (at least for now)
        layer0 = self.conv1(img)
        hidden1, cell1 = self.lstm1(layer0, hiddens[1], cells[1])
        hidden2, cell2 = self.lstm2(hidden1, hiddens[2], cells[2])
        hidden3, cell3 = self.lstm3(self.downsample23(hidden2), hiddens[3], cells[3])
        hidden4, cell4 = self.lstm4(hidden3, hiddens[4], cells[4])
        
        input5 = torch.cat((self.downsample45(hidden4), tiled), 1)
        hidden5, cell5 = self.lstm5(input5, hiddens[5], cells[5])

        kernels = self.to_kernels(hidden5.view([-1, 138 * 8 * 8])).view([-1, 25, 10, 1])
        # NOT a channel softmax, but a spatial one
        normalized_kernels = torch.transpose(self.softmax(kernels), 1, 2)
        normalized_kernels = torch.stack(torch.split(torch.squeeze(normalized_kernels), 5, dim = 2), dim = -2)
        # We will wait to transform the images until we compute the loss.

        hidden6, cell6 = self.lstm6(self.upsample56(hidden5), hiddens[6], cells[6])
        input7 = self.upsample67(torch.cat((hidden6, hidden3), 1))
        hidden7, cell7 = self.lstm7(input7, hiddens[7], cells[7])

        input_out = self.last_upsample(torch.cat((hidden7, hidden1), 1))
        masks = self.softmax(self.conv2(input_out)) # channel softmax

        transformed_images = apply_kernels(bg, img, normalized_kernels)

        return expected_pixel(transformed_images, masks), [None, hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7],\
            [None, cell1, cell2, cell3, cell4, cell5, cell6, cell7]

    def initHidden(self, batch_size = 1):
        # The first entry is just so that the indexing aligns with the semantics
        return [None,
                self.lstm1.initHidden(batch_size),
                self.lstm2.initHidden(batch_size),
                self.lstm3.initHidden(batch_size),
                self.lstm4.initHidden(batch_size),
                self.lstm5.initHidden(batch_size),
                self.lstm6.initHidden(batch_size),
                self.lstm7.initHidden(batch_size),
        ]

    def initCell(self, batch_size = 1):
        return [None,
                self.lstm1.initCell(batch_size),
                self.lstm2.initCell(batch_size),
                self.lstm3.initCell(batch_size),
                self.lstm4.initCell(batch_size),
                self.lstm5.initCell(batch_size),
                self.lstm6.initCell(batch_size),
                self.lstm7.initCell(batch_size),
        ]
