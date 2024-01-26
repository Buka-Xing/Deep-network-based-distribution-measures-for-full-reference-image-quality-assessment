# Copyright (C) <2022> Xingran Liao
# @ City University of Hong Kong

# Permission is hereby granted, free of charge, to any person obtaining a copy of this code and
# associated documentation files (the "code"), to deal in the code without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the code,
# and to permit persons to whom the code is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the code.

# THE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE Xingran Liao BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE code OR THE USE OR OTHER DEALINGS IN THE code.

#================================================
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from utils import downsample

class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()

def js_div(p_output, q_output, get_softmax=True):
    KLDivLoss = nn.KLDivLoss(reduction='none')
    if get_softmax:
        p_output = F.softmax(p_output,dim=0)
        q_output = F.softmax(q_output,dim=0)
    log_mean_output = ((p_output + q_output + 1e-17)/2).log()
    part1 = KLDivLoss(log_mean_output, p_output).sum(dim=0)
    part2 = KLDivLoss(log_mean_output, q_output).sum(dim=0)
    return (part1+part2)/2

def js_distance(X, Y, win=8):
    chn_num = X.shape[1]

    X_patch = torch.reshape(X, [win, win, chn_num, -1])
    Y_patch = torch.reshape(Y, [win, win, chn_num, -1])
    patch_num = (X.shape[2] // win) * (X.shape[3] // win)

    X_1D = torch.reshape(X_patch, [-1, chn_num * patch_num])
    Y_1D = torch.reshape(Y_patch, [-1, chn_num * patch_num])

    X_pdf = X_1D
    Y_pdf = Y_1D

    jsd = js_div(X_pdf, Y_pdf)

    L2 = ((X_1D - Y_1D) ** 2).sum(dim=0)
    w = (1 / (torch.sqrt(torch.exp((- 1 / (jsd + 10)))) * (jsd + 10) ** 2))

    final = jsd + L2 * w
    # final = jsd

    return final.mean()


class DeepJSD_squeeze(torch.nn.Module):
    def __init__(self, channels=3):
        assert channels == 3
        super(DeepJSD_squeeze, self).__init__()
        SquNet_pretrained_features = models.squeezenet1_0(pretrained=True).features

        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()

        for x in range(0, 2):
            self.stage1.add_module(str(x), SquNet_pretrained_features[x])
        self.stage2.add_module(str(2), SquNet_pretrained_features[2])
        for x in range(3, 6):
            self.stage2.add_module(str(x), SquNet_pretrained_features[x])
        self.stage3.add_module(str(6), SquNet_pretrained_features[6])
        for x in range(7, 11):
            self.stage3.add_module(str(x), SquNet_pretrained_features[x])
        self.stage4.add_module(str(11), SquNet_pretrained_features[11])
        self.stage4.add_module(str(12), SquNet_pretrained_features[12])

        for param in self.parameters():
            param.requires_grad = False

        self.chns = [3, 96, 128, 256, 256]

    def forward_once(self, x):
        h = x
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]

    def forward(self, x, y, as_loss=True, resize=True):
        assert x.shape == y.shape
        if resize:
            x, y,_ ,_ = downsample(x, y)
        if as_loss:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
        score = 0
        layer_score=[]
        window = 8
        for k in range(len(self.chns)):
            row_padding = round(feats0[k].size(2) / window) * window - feats0[k].size(2)
            column_padding = round(feats0[k].size(3) / window) * window - feats0[k].size(3)

            pad = nn.ZeroPad2d((column_padding, 0, 0, row_padding))
            feats0_k = pad(feats0[k])
            feats1_k = pad(feats1[k])

            tmp = js_distance(feats0_k, feats1_k, win=window)
            layer_score.append(torch.log(tmp + 1))
            score = score + tmp
        score = score / (k+1)


        if as_loss:
            return score
        else:
            return torch.log(score + 1) ** 0.25
if __name__ == '__main__':
    from PIL import Image
    import argparse
    from utils import prepare_image

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='images/I47.png')
    parser.add_argument('--dist', type=str, default='images/I47_03_05.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref = prepare_image(Image.open(args.ref).convert("RGB")).to(device)
    dist = prepare_image(Image.open(args.dist).convert("RGB")).to(device)

    model = DeepJSD_squeeze().to(device)

    score = model(ref, dist, as_loss=False) # 尚不清楚as_loss为true会带来什么影响.
    print('score: %.4f' % score.item())
