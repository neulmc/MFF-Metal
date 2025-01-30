import torch
import torch.nn as nn
import torch.nn.functional as F


def Split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2

def Split3(x):
    c = int(x.size()[1])
    c1 = round(c / 3)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:int(c1*2), :, :].contiguous()
    x3 = x[:, int(c1*2):, :, :].contiguous()
    return x1, x2, x3

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output

class conv3x3_resume(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.conv3x3 = Conv(nIn // 2, nIn // 2, kSize, 1, padding=1, bn_acti=True)
        self.conv1x1_resume = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.conv3x3(input)
        output = self.conv1x1_resume(output)
        return output

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class Init_Block(nn.Module):
    def __init__(self, tiny_c = 32):
        super(Init_Block, self).__init__()
        self.init_conv = nn.Sequential(
            Conv(3, tiny_c, 3, 1, padding=1, bn_acti=True),
            #Conv(tiny_c, tiny_c, 3, 1, padding=1, bn_acti=True),
            #Conv(tiny_c, tiny_c, 3, 1, padding=1, bn_acti=True),
            Conv(tiny_c, tiny_c, 3, 1, padding=2, dilation=(2, 2), bn_acti=True),
            Conv(tiny_c, tiny_c, 3, 1, padding=2, dilation=(2, 2), bn_acti=True),
        )

    def forward(self, x):
        o = self.init_conv(x)
        return o

class SEM_B(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)

        self.dconv_left = Conv(nIn // 4, nIn // 4, (dkSize, dkSize), 1,
                               padding=(1, 1), groups=nIn // 4, bn_acti=True)

        self.dconv_right = Conv(nIn // 4, nIn // 4, (dkSize, dkSize), 1,
                                padding=(1 * d, 1 * d), groups=nIn // 4, dilation=(d, d), bn_acti=True)


        self.bn_relu_1 = BNPReLU(nIn)

        self.conv3x3_resume = conv3x3_resume(nIn , nIn , (dkSize, dkSize), 1,
                                padding=(1 , 1 ),  bn_acti=True)

    def forward(self, input):

        output = self.conv3x3(input)

        x1, x2 = Split(output)

        letf = self.dconv_left(x1)

        right = self.dconv_right(x2)

        output = torch.cat((letf, right), 1)
        output = self.conv3x3_resume(output)

        return self.bn_relu_1(output + input)

class SEM_B3(nn.Module):
    def __init__(self, nIn, d=[1, 2], kSize=3, dkSize=3):
        super().__init__()

        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)

        self.dconv_left = Conv(nIn // 6, nIn // 6, (dkSize, dkSize), 1,
                               padding=(1, 1), groups=nIn // 6, bn_acti=True)
        self.dconv_middle = Conv(nIn // 6, nIn // 6, (dkSize, dkSize), 1,
                                padding=(1 * d[0], 1 * d[0]), groups=nIn // 6, dilation=(d[0], d[0]), bn_acti=True)
        self.dconv_right = Conv(nIn // 6, nIn // 6, (dkSize, dkSize), 1,
                                padding=(1 * d[1], 1 * d[1]), groups=nIn // 6, dilation=(d[1], d[1]), bn_acti=True)


        self.bn_relu_1 = BNPReLU(nIn)

        self.conv3x3_resume = conv3x3_resume(nIn , nIn , (dkSize, dkSize), 1,
                                padding=(1 , 1 ),  bn_acti=True)

    def forward(self, input):

        output = self.conv3x3(input)

        x1, x2, x3 = Split3(output)

        letf = self.dconv_left(x1)
        middle = self.dconv_middle(x2)
        right = self.dconv_right(x3)

        output = torch.cat((letf, middle, right), 1)
        output = self.conv3x3_resume(output)

        return self.bn_relu_1(output + input)


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        #self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.conv3x3 = Conv(nIn, nConv, kSize=2, stride=2, padding=0)  # fix
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output


class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)

        return input


class SENet_Block(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SENet_Block, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class PMCA(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(PMCA, self).__init__()

        self.partition_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.conv2x2 = Conv(ch_in, ch_in, 2, 1, padding=(0, 0), groups=ch_in, bn_acti=False)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.SE_Block = SENet_Block(ch_in=ch_in, reduction=reduction)

    def forward(self, x):
        o1 = self.partition_pool(x)

        o1 = self.conv2x2(o1)

        o2 = self.global_pool(x)

        o_sum = o1 + o2
        w = self.SE_Block(o_sum)
        o = w * x

        return o


class FFM_A(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(FFM_A, self).__init__()
        self.bn_prelu = BNPReLU(ch_in)
        self.conv1x1 = Conv(ch_in, ch_out, 1, 1, padding=0, bn_acti=False)

    def forward(self, x):
        x1, x2 = x
        o = self.bn_prelu(torch.cat([x1, x2], 1))
        o = self.conv1x1(o)
        return o


class FFM_B(nn.Module):
    def __init__(self, ch_in, ch_pmca):
        super(FFM_B, self).__init__()
        self.PMCA = PMCA(ch_in=ch_pmca, reduction=8)
        self.bn_prelu = BNPReLU(ch_in)
        self.conv1x1 = Conv(ch_in, ch_in, 1, 1, padding=0, bn_acti=False)

    def forward(self, x):
        x1, x2, x3 = x
        x2 = self.PMCA(x2)
        o = self.bn_prelu(torch.cat([x1, x2, x3], 1))
        o = self.conv1x1(o)
        return o

class FFM_B0(nn.Module):
    def __init__(self, ch_in, ch_out, ch_pmca):
        super(FFM_B0, self).__init__()
        self.PMCA = PMCA(ch_in=ch_pmca, reduction=8)
        self.bn_prelu = BNPReLU(ch_in)
        self.conv1x1 = Conv(ch_in, ch_out, 1, 1, padding=0, bn_acti=False)

    def forward(self, x):
        x1, x2, x3, x4 = x
        x34 = self.PMCA(torch.cat([x3,x4],1))
        o = self.bn_prelu(torch.cat([x1, x2, x34], 1))
        o = self.conv1x1(o)
        return o

class FFM_B1(nn.Module):
    def __init__(self, ch_in, ch_out, ch_pmca):
        super(FFM_B1, self).__init__()
        self.PMCA = PMCA(ch_in=ch_pmca, reduction=8)
        self.bn_prelu = BNPReLU(ch_in)
        self.conv1x1 = Conv(ch_in, ch_out, 1, 1, padding=0, bn_acti=False)

    def forward(self, x):
        x1, x2, x3, x4, x5 = x
        x345 = self.PMCA(torch.cat([x3,x4,x5],1))
        o = self.bn_prelu(torch.cat([x1, x2, x345], 1))
        o = self.conv1x1(o)
        return o

class FFM_B2(nn.Module):
    def __init__(self, ch_in, ch_out, ch_pmca):
        super(FFM_B2, self).__init__()
        self.PMCA = PMCA(ch_in=ch_pmca, reduction=8)
        self.bn_prelu = BNPReLU(ch_in)
        self.conv1x1 = Conv(ch_in, ch_out, 1, 1, padding=0, bn_acti=False)

    def forward(self, x):
        x1, x2, x3, x4, x5, x6 = x
        #x2 = self.PMCA(x2)
        x3456 = self.PMCA(torch.cat([x3,x4,x5,x6],1))
        o = self.bn_prelu(torch.cat([x1, x2, x3456], 1))
        o = self.conv1x1(o)
        return o

class SEM_B_Block(nn.Module):
    def __init__(self, num_channels, num_block, dilation, flag):
        super(SEM_B_Block, self).__init__()
        self.SEM_B_Block = nn.Sequential()
        for i in range(0, num_block):
            #self.SEM_B_Block.add_module("SEM_Block_" + str(flag) + str(i), SEM_B(num_channels, d=dilation[i]))
            self.SEM_B_Block.add_module("SEM_Block_" + str(flag) + str(i), SEM_B3(num_channels, d=dilation[i]))
    def forward(self, x):
        o = self.SEM_B_Block(x)
        return o

class max_re(nn.Module):
    def __init__(self, c1=16, c2=32, classes=19, input_c1 = 30, input_c2 = 60):
        super(max_re, self).__init__()
        self.c1, self.c2 = c1, c2

        self.mid_layer_1x1 = Conv(input_c1, c1, 1, 1, padding=0, bn_acti=False)
        self.deep_layer_1x1 = Conv(input_c2, c2, 1, 1, padding=0, bn_acti=False)

        self.DwConv1 = Conv(self.c1 + self.c2, self.c1 + self.c2, (3, 3), 1, padding=(1, 1),
                            groups=self.c1 + self.c2, bn_acti=True)
        self.PwConv1 = Conv(self.c1 + self.c2, classes, 1, 1, padding=0, bn_acti=False)

        self.scale_atten = Conv(self.c1 + self.c2, 1, 1, 1, padding=0, bn_acti=False)
        self.gp = nn.AdaptiveAvgPool2d(1)

        self.DwConv2 = Conv(input_c2, input_c2, (3, 3), 1, padding=(1, 1), groups=input_c2, bn_acti=True)
        self.PwConv2 = Conv(input_c2, classes, 1, 1, padding=0, bn_acti=False)

    def forward(self, x1, x2):
        x2_size = x2.size()[2:]

        x1_ = self.mid_layer_1x1(x1)
        x2_ = self.deep_layer_1x1(x2)

        x2_ = F.interpolate(x2_, [x2_size[0] * 2, x2_size[1] * 2], mode='bilinear', align_corners=False)

        x1_x2_cat = torch.cat([x1_, x2_], 1)

        x1_x2_cat_b = self.scale_atten(x1_x2_cat)
        x1_x2_cat_b = self.gp(x1_x2_cat_b)

        x1_x2_cat_a = self.DwConv1(x1_x2_cat)
        x1_x2_cat_a = self.PwConv1(x1_x2_cat_a)
        x1_x2_cat_att = torch.sigmoid(x1_x2_cat_a)

        o = self.DwConv2(x2)
        o = self.PwConv2(o)
        o = F.interpolate(o, [x2_size[0] * 2, x2_size[1] * 2], mode='bilinear', align_corners=False)

        o = o * x1_x2_cat_att
        return o, x1_x2_cat_b

class MAD(nn.Module):
    def __init__(self, c1=16, c2=24, classes=19, input_c=[60,90,120]):
        super(MAD, self).__init__()
        self.max_re48 = max_re(c1=c1, c2=c2, classes=classes, input_c1 = input_c[1],input_c2 = input_c[2])
        self.max_re24 = max_re(c1=c1, c2=c2, classes=classes, input_c1 = input_c[0],input_c2 = input_c[1])
        self.softmax = nn.Softmax()

    def forward(self, x):
        x0, x1, x2 = x
        x2_size = x2.size()[2:]

        re48, gp48 = self.max_re48(x1,x2)
        o48 = F.interpolate(re48, [x2_size[0] * 8, x2_size[1] * 8], mode='bilinear', align_corners=False)
        re24, gp24 = self.max_re24(x0,x1)
        o24 = F.interpolate(re24, [x2_size[0] * 8, x2_size[1] * 8], mode='bilinear', align_corners=False)

        gp = torch.cat([gp48, gp24], dim=1)
        gp = self.softmax(gp)

        o = o48 * gp[:,0:1,:,:] + o24 * gp[:,1:2,:,:]
        return o, [o48,o24], gp


class self_net(nn.Module):
    def __init__(self, classes=4, block_0=2, block_1=2, block_2=6, channel=[30,60,90,120]):
        super().__init__()
        self.classes = classes
        self.block_0 = block_0
        self.block_1 = block_1
        self.block_2 = block_2
        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        self.down_2 = InputInjection(2)  # down-sample the image 2 times
        self.down_3 = InputInjection(3)  # down-sample the image 3 times
        self.c1 = channel[0]
        self.c2 = channel[1]
        self.c3 = channel[2]
        self.c4 = channel[3]

        self.Init_Block = Init_Block(self.c1)

        self.downsample_0 = DownSamplingBlock(self.c1, self.c2)
        self.SEM_B_Block0 = SEM_B_Block(num_channels=self.c2, num_block=self.block_0, dilation=[[2,3], [2,3]], flag=0)
        self.FFM_B0 = FFM_B0(ch_in=self.c2 + self.c2 + self.c1 + 3, ch_out = self.c2, ch_pmca=self.c2 +self.c1 )

        self.downsample_1 = DownSamplingBlock(self.c2, self.c3)
        self.SEM_B_Block1 = SEM_B_Block(num_channels=self.c3, num_block=self.block_1, dilation=[[2,3], [2,3], [4,6], [4,6]], flag=1)
        self.FFM_B1 = FFM_B1(ch_in=self.c3 + self.c1 + self.c2 + self.c3 + 3, ch_out = self.c3, ch_pmca=self.c1 + self.c2+  self.c3)

        self.downsample_2 = DownSamplingBlock(self.c3, self.c4)
        self.SEM_B_Block2 = SEM_B_Block(num_channels=self.c4, num_block=self.block_2, dilation=[[2,3], [2,3], [4,6], [4,6], [8,12], [8,12]], flag=2)
        self.FFM_B2 = FFM_B2(ch_in=self.c4 + self.c1 + self.c2 + self.c3 + self.c4 + 3, ch_out = self.c4, ch_pmca=self.c1 + self.c2 + self.c3 + self.c4)

        self.MAD = MAD(classes=self.classes, input_c=[self.c2, self.c3, self.c4])
        self.softmax = nn.Softmax()

    def forward(self, input, T_KD=False, multi_show=False):
        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)

        # stage1
        out_init_block = self.Init_Block(input)

        # stage2
        out_downsample_0 = self.downsample_0(out_init_block)
        out_sem_block0 = self.SEM_B_Block0(out_downsample_0)
        input_ffm_a = out_sem_block0, down_1, out_downsample_0, self.down_1(out_init_block)
        out_ffm_a = self.FFM_B0(input_ffm_a)

        # stage3
        out_downsample_1 = self.downsample_1(out_ffm_a)
        out_sem_block1 = self.SEM_B_Block1(out_downsample_1)
        input_sem1_pmca1 = out_sem_block1, down_2, out_downsample_1, self.down_1(out_ffm_a), self.down_2(out_init_block)
        out_ffm_b1 = self.FFM_B1(input_sem1_pmca1)

        # stage4
        out_downsample_2 = self.downsample_2(out_ffm_b1)
        out_se_block2 = self.SEM_B_Block2(out_downsample_2)
        input_sem2_pmca2 = out_se_block2, down_3, out_downsample_2, self.down_1(out_ffm_b1), self.down_2(out_ffm_a), self.down_3(out_init_block)
        out_ffm_b2 = self.FFM_B2(input_sem2_pmca2)

        # Decoder
        input_ffmb1_ffmb2 = out_ffm_a, out_ffm_b1, out_ffm_b2
        out_mad, [o48,o24], gp = self.MAD(input_ffmb1_ffmb2)

        # output
        if T_KD:
            return out_mad
        elif multi_show:
            return self.softmax(out_mad), [self.softmax(o48), self.softmax(o24)], gp
        else:
            return self.softmax(out_mad)