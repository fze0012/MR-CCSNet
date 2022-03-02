import torch
from torch import nn


class ResLayerPool(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResLayerPool, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=2, bias=False),
        )

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(x)
        res = self.pool(x)

        return out + res


class MRCCSNet(nn.Module):
    def __init__(self, sensing_rate):
        super(MRCCSNet, self).__init__()

        self.sensing_rate = sensing_rate
        self.measurement = int(sensing_rate * 1024)
        self.base = 64
        self.blocksize = 32
        n_feats_hsc = 32

        if sensing_rate == 0.5:
            self.hsc = nn.Sequential(
                nn.Conv2d(1, n_feats_hsc, kernel_size=3, padding=1, stride=1, bias=False),
                ResLayerPool(n_feats_hsc, n_feats_hsc),
                nn.Conv2d(n_feats_hsc, 2, kernel_size=1, padding=0, stride=1, bias=False),
            )

            self.initial = nn.Conv2d(2, 4, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 2
        elif sensing_rate == 0.25:
            self.hsc = nn.Sequential(
                nn.Conv2d(1, n_feats_hsc, kernel_size=3, padding=1, stride=1, bias=False),
                ResLayerPool(n_feats_hsc, n_feats_hsc),
                ResLayerPool(n_feats_hsc, n_feats_hsc),
                nn.Conv2d(n_feats_hsc, 4, kernel_size=1, padding=0, stride=1, bias=False)
            )
            self.initial = nn.Conv2d(4, 16, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 4
        elif sensing_rate == 0.125:
            self.hsc = nn.Sequential(
                nn.Conv2d(1, n_feats_hsc, kernel_size=3, padding=1, stride=1, bias=False),
                ResLayerPool(n_feats_hsc, n_feats_hsc),
                ResLayerPool(n_feats_hsc, n_feats_hsc),
                nn.Conv2d(n_feats_hsc, 2, kernel_size=1, padding=0, stride=1, bias=False)
            )
            self.initial = nn.Conv2d(2, 16, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 4
        elif sensing_rate == 0.0625:
            self.hsc = nn.Sequential(
                nn.Conv2d(1, n_feats_hsc, kernel_size=3, padding=1, stride=1, bias=False),
                ResLayerPool(n_feats_hsc, n_feats_hsc),
                ResLayerPool(n_feats_hsc, n_feats_hsc),
                ResLayerPool(n_feats_hsc, n_feats_hsc),
                nn.Conv2d(n_feats_hsc, 4, kernel_size=1, padding=0, stride=1, bias=False),

            )
            self.initial = nn.Conv2d(4, 64, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 8
        elif sensing_rate == 0.03125:
            self.hsc = nn.Sequential(
                nn.Conv2d(1, n_feats_hsc, kernel_size=3, padding=1, stride=1, bias=False),
                ResLayerPool(n_feats_hsc, n_feats_hsc),
                ResLayerPool(n_feats_hsc, n_feats_hsc),
                ResLayerPool(n_feats_hsc, n_feats_hsc),
                nn.Conv2d(n_feats_hsc, 2, kernel_size=1, padding=0, stride=1, bias=False),
            )
            self.initial = nn.Conv2d(2, 64, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 8
        elif sensing_rate == 0.015625:
            self.hsc = nn.Sequential(
                nn.Conv2d(1, n_feats_hsc, kernel_size=3, padding=1, stride=1, bias=False),
                ResLayerPool(n_feats_hsc, n_feats_hsc),
                ResLayerPool(n_feats_hsc, n_feats_hsc),
                ResLayerPool(n_feats_hsc, n_feats_hsc),
                ResLayerPool(n_feats_hsc, n_feats_hsc),

                nn.Conv2d(n_feats_hsc, 4, kernel_size=1, padding=0, stride=1, bias=False),
            )
            self.initial = nn.Conv2d(4, 256, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 16

        self.conv1 = nn.Conv2d(1, self.base, kernel_size=3, padding=1, stride=1, bias=True)
        if sensing_rate == 0.5:
            self.res1 = MRB2()
            self.res2 = MRB2()
            self.res3 = MRB2()
            self.res4 = MRB2()
            self.res5 = MRB2()
            self.res6 = MRB2()
            self.res7 = MRB2()
            self.res8 = MRB2()
        elif sensing_rate == 0.25:
            self.res1 = MRB4()
            self.res2 = MRB4()
            self.res3 = MRB4()
            self.res4 = MRB4()
            self.res5 = MRB4()
            self.res6 = MRB4()
            self.res7 = MRB4()
            self.res8 = MRB4()
        elif sensing_rate == 0.125:
            self.res1 = MRB8()
            self.res2 = MRB8()
            self.res3 = MRB8()
            self.res4 = MRB8()
            self.res5 = MRB8()
            self.res6 = MRB8()
            self.res7 = MRB8()
            self.res8 = MRB8()
        elif sensing_rate == 0.0625:
            self.res1 = MRB16()
            self.res2 = MRB16()
            self.res3 = MRB16()
            self.res4 = MRB16()
            self.res5 = MRB16()
            self.res6 = MRB16()
            self.res7 = MRB16()
            self.res8 = MRB16()
        elif sensing_rate == 0.03125:
            self.res1 = MRB32()
            self.res2 = MRB32()
            self.res3 = MRB32()
            self.res4 = MRB32()
            self.res5 = MRB32()
            self.res6 = MRB32()
            self.res7 = MRB32()
            self.res8 = MRB32()
        elif sensing_rate == 0.015625:
            self.res1 = MRB64()
            self.res2 = MRB64()
            self.res3 = MRB64()
            self.res4 = MRB64()
            self.res5 = MRB64()
            self.res6 = MRB64()
            self.res7 = MRB64()
            self.res8 = MRB64()

        self.conv2 = nn.Conv2d(self.base, 1, kernel_size=3, padding=1, stride=1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        xsample = self.hsc(x)
        x = self.initial(xsample)
        initial = nn.PixelShuffle(self.m)(x)

        out = self.relu(self.conv1(initial))
        out = self.res1(out, xsample)
        out = self.res2(out, xsample)
        out = self.res3(out, xsample)
        out = self.res4(out, xsample)
        out = self.res5(out, xsample)
        out = self.res6(out, xsample)
        out = self.res7(out, xsample)
        out = self.res8(out, xsample)

        out = self.conv2(out)

        return out + initial, initial


class MRB64(nn.Module):
    def __init__(self):
        super(MRB64, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),

        )

        self.mix1 = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )
        self.mix2 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),

        )
        self.mix3 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(4),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )
        self.mix4 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(8),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )
        self.mix5 = nn.Sequential(
            nn.Conv2d(4, 256, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(16),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )

    def forward(self, x, sample):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x5 = self.conv5(torch.cat([x4, self.mix1(sample)], dim=1))
        x6 = self.conv6(torch.cat([x5, x4], dim=1))
        x7 = self.conv7(torch.cat([x6, self.mix2(sample)], dim=1))
        x8 = self.conv8(torch.cat([x7, x3], dim=1))
        x9 = self.conv9(torch.cat([x8, self.mix3(sample)], dim=1))
        x10 = self.conv10(torch.cat([x9, x2], dim=1))
        x11 = self.conv11(torch.cat([x10, self.mix4(sample)], dim=1))
        x12 = self.conv12(torch.cat([x11, x1], dim=1))
        x13 = self.conv13(torch.cat([x12, self.mix5(sample)], dim=1))
        x14 = self.conv14(torch.cat([x13, x], dim=1))
        return x14


class MRB32(nn.Module):
    def __init__(self):
        super(MRB32, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )

        self.mix1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )
        self.mix2 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),

        )
        self.mix3 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(4),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )
        self.mix4 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(8),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )

    def forward(self, x, sample):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(torch.cat([x3, self.mix1(sample)], dim=1))
        x5 = self.conv5(torch.cat([x4, x3], dim=1))
        x6 = self.conv6(torch.cat([x5, self.mix2(sample)], dim=1))
        x7 = self.conv7(torch.cat([x6, x2], dim=1))
        x8 = self.conv8(torch.cat([x7, self.mix3(sample)], dim=1))
        x9 = self.conv9(torch.cat([x8, x1], dim=1))
        x10 = self.conv10(torch.cat([x9, self.mix4(sample)], dim=1))
        x11 = self.conv11(torch.cat([x10, x], dim=1))
        return x11


class MRB16(nn.Module):
    def __init__(self):
        super(MRB16, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )

        self.mix1 = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )
        self.mix2 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),

        )
        self.mix3 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(4),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )
        self.mix4 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(8),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )

    def forward(self, x, sample):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(torch.cat([x3, self.mix1(sample)], dim=1))
        x5 = self.conv5(torch.cat([x4, x3], dim=1))
        x6 = self.conv6(torch.cat([x5, self.mix2(sample)], dim=1))
        x7 = self.conv7(torch.cat([x6, x2], dim=1))
        x8 = self.conv8(torch.cat([x7, self.mix3(sample)], dim=1))
        x9 = self.conv9(torch.cat([x8, x1], dim=1))
        x10 = self.conv10(torch.cat([x9, self.mix4(sample)], dim=1))
        x11 = self.conv11(torch.cat([x10, x], dim=1))

        return x11


class MRB8(nn.Module):
    def __init__(self):
        super(MRB8, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),

        )
        self.mix1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),

        )
        self.mix2 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),

        )
        self.mix3 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(4),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),

        )

    def forward(self, x, sample):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x2, self.mix1(sample)], dim=1))
        x4 = self.conv4(torch.cat([x3, self.mix2(sample)], dim=1))
        x5 = self.conv5(torch.cat([x1, x4], dim=1))
        x6 = self.conv6(torch.cat([x5, self.mix3(sample)], dim=1))
        x7 = self.conv7(torch.cat([x6, x], dim=1))
        return x7


class MRB4(nn.Module):
    def __init__(self):
        super(MRB4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),

        )
        self.mix1 = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),

        )
        self.mix2 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),

        )
        self.mix3 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(4),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),

        )

    def forward(self, x, sample):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x2, self.mix1(sample)], dim=1))
        x4 = self.conv4(torch.cat([x3, self.mix2(sample)], dim=1))
        x5 = self.conv5(torch.cat([x1, x4], dim=1))
        x6 = self.conv6(torch.cat([x5, self.mix3(sample)], dim=1))
        x7 = self.conv7(torch.cat([x6, x], dim=1))
        return x7


class MRB2(nn.Module):
    def __init__(self):
        super(MRB2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU()
        )
        self.mix1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
        )
        self.mix2 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU()
        )

    def forward(self, x, sample):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat([x1, self.mix1(sample)], dim=1))
        x3 = self.conv3(torch.cat([x2, x1], dim=1))
        x4 = self.conv4(torch.cat([x3, self.mix2(sample)], dim=1))
        x5 = self.conv5(torch.cat([x4, x], dim=1))

        return x5


if __name__ == '__main__':
    image = torch.rand(1, 1, 96, 96).cpu()
    net = MRCCSNet(0.5).cpu()
    x = net(image)
