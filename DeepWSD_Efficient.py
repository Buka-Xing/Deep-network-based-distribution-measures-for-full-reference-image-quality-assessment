import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from utils import downsample
from ot.lp import wasserstein_1d

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

def ws_distance(X,Y,P=1,win=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chn_num = X.shape[1]

    X_patch   = torch.reshape(X,[win,win,chn_num,-1])
    Y_patch   = torch.reshape(Y,[win,win,chn_num,-1])
    patch_num = (X.shape[2]//win) * (X.shape[3]//win)

    X_1D = torch.reshape(X_patch,[-1,chn_num*patch_num])
    Y_1D = torch.reshape(Y_patch,[-1,chn_num*patch_num])

    interval = np.arange(0, X_1D.shape[0], 1)
    all_samples = torch.from_numpy(interval).to(device).repeat([patch_num*chn_num,1]).t()

    X_pdf = F.softmax(X_1D, dim=-1)
    Y_pdf = F.softmax(Y_1D, dim=-1)

    wsd   = wasserstein_1d(all_samples, all_samples, X_pdf, Y_pdf, P)

    L2 = ((X_1D - Y_1D) ** 2).sum(dim=0)
    w  =  (1 / ( torch.sqrt(torch.exp( (- 1/(wsd+10) ))) * (wsd+10)**2))

    final = wsd + L2 * w
    # final = wsd

    return final.mean()


class DeepWSD_eff(torch.nn.Module):

    def __init__(self, channels=3):
        assert channels == 3
        super(DeepWSD_eff, self).__init__()

        effb7_pretrained_features = models.efficientnet_b7(pretrained=True).features
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        self.stage6 = torch.nn.Sequential()
        self.stage7 = torch.nn.Sequential()

        for x in range(0, 2):
            self.stage1.add_module(str(x), effb7_pretrained_features[x])
        for x in range(2, 3):
            self.stage2.add_module(str(x), effb7_pretrained_features[x])
        for x in range(3, 4):
            self.stage3.add_module(str(x), effb7_pretrained_features[x])
        for x in range(4, 5):
            self.stage4.add_module(str(x), effb7_pretrained_features[x])
        for x in range(5, 6):
            self.stage5.add_module(str(x), effb7_pretrained_features[x])
        for x in range(6, 7):
            self.stage6.add_module(str(x), effb7_pretrained_features[x])
        for x in range(7, 8):
            self.stage7.add_module(str(x), effb7_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.chns = [3, 16, 24, 40, 80, 112, 192, 320]

    def forward_once(self, x):
        h = x
        h = self.stage1(h)
        h_1 = h
        h = self.stage2(h)
        h_2 = h
        h = self.stage3(h)
        h_3 = h
        h = self.stage4(h)
        h_4 = h
        h = self.stage5(h)
        h_5 = h
        h = self.stage6(h)
        h_6 = h
        h = self.stage7(h)
        h_7 = h

        return [x, h_1, h_2, h_3, h_4, h_5, h_6, h_7]

    def forward(self, x, y, as_loss=True, resize=True):
        assert x.shape == y.shape
        if resize:
            x, y, _ , _ = downsample(x, y)
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

            tmp = ws_distance(feats0_k, feats1_k, win=window)
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

    model = DeepWSD_eff().to(device)

    score = model(ref, dist, as_loss=False)
    print('score: %.4f' % score.item())
    # score: 0.3347