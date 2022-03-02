# Reference from https://github.com/liujiawei2333/Compressed-sensing-CSNet/
import torch.nn as nn
import numpy as np
import torch


class ResBlockbase(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True):
        super(ResBlockbase, self).__init__()

        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=1, bias=bias)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out += x
        out = self.relu2(self.conv2(out))
        return out


class CsNet(nn.Module):
    def __init__(self, sensing_rate):
        super(CsNet, self).__init__()

        self.measurement = int(sensing_rate * 1024)
        self.base = 64

        self.sample = nn.Conv2d(1, self.measurement, kernel_size=32, padding=0, stride=32, bias=False)
        self.initial = nn.Conv2d(self.measurement, 1024, kernel_size=1, padding=0, stride=1, bias=False)

        self.conv1 = nn.Conv2d(1, self.base, kernel_size=3, padding=1, stride=1, bias=False)
        self.res1 = ResBlockbase(self.base, 3)
        self.res2 = ResBlockbase(self.base, 3)
        self.res3 = ResBlockbase(self.base, 3)
        self.res4 = ResBlockbase(self.base, 3)
        self.res5 = ResBlockbase(self.base, 3)
        self.res6 = ResBlockbase(self.base, 3)
        self.res7 = ResBlockbase(self.base, 3)
        self.res8 = ResBlockbase(self.base, 3)
        self.conv2 = nn.Conv2d(self.base, 1, kernel_size=3, padding=1, stride=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        [b, c, h, w] = x.shape

        x = self.sample(x)
        """
        noise = np.random.normal(loc=0, scale=0.1, size=x.shape)
        noise = torch.from_numpy(noise)
        noise = noise.type(torch.FloatTensor).cuda()

        x = x + noise
        """
        x = self.initial(x)
        initial = nn.PixelShuffle(32)(x)
        out = self.relu(self.conv1(initial))
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.res6(out)
        out = self.res7(out)
        out = self.res8(out)
        out = self.conv2(out)
        return out + initial, initial


if __name__ == '__main__':
    image = torch.rand(4, 1, 96, 96).cpu()
    net = CsNet(0.5).cpu()
    x = net(image)
