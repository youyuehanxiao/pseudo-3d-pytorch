from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = ['P3D', 'P3D63', 'P3D131', 'P3D199']

#空间卷积核（1×3×3）
def conv_S(in_planes,out_planes,stride=1,padding=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=1, padding=padding, bias=False)

#时间卷积核（3×1×1）
def conv_T(in_planes,out_planes,stride=1,padding=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=1, padding=padding, bias=False)

#为了保证残差连接的尺寸相同，对输入进行变换操作（将原resnet的点卷积操作变成了池化操作，因为做一次3D卷积的升维操作需要花费很大计算量）
def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride) #对输入数据进行池化，使其尺寸与卷积后的尺寸相同（当前没有改变通道数）

    #生成一个全0张量（通道数为输出张量与输入张量通道数之差，其它尺寸与输入相同），用于拼接到输入张量通道维度的末尾，使其与卷积后的特征通道数相同
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).zero_()

    #将cpu张量转换成GPU张量
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    #将生成的全0张量按通道维度拼接到输入数据末尾，并且将张量封装到Variable——用于梯度计算，包含data和grad
    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_s=0, depth_3d=47, ST_struc=('A','B','C')):
        '''
        :param inplanes and planes: 输入和输出通道数
        :param n_s: 当前层数
        :param depth_3d: 使用3D卷积的层数（超过depth_3d的层使用2D卷积）
        :param ST_struc: 对应P3D的三种结构
        '''
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.depth_3d = depth_3d
        self.ST_struc = ST_struc
        self.len_ST = len(self.ST_struc)

        stride_p = stride #步长

        #顶层卷积，通道降维，时间维度不变，空间维度在不同层衔接时（第0层不算）下采样（缩小1/2）
        if not self.downsample == None: #需要进行下采样（缩小空间尺寸，增加通道数）
            stride_p = (1, 2, 2)
        if n_s < self.depth_3d: #3D卷积层
            if n_s == 0: #3D卷积层顶层
                stride_p = 1
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False, stride=stride_p) #3D卷积通道降维
            #print(n_s, '层，步长：', stride_p, stride)
            #self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False, stride=(stride_p, stride, stride))  # 3D卷积通道降维
            self.bn1 = nn.BatchNorm3d(planes)
        else: #2D卷积层
            # if n_s == self.depth_3d: #2D卷积层顶层
            #     stride_p = 2
            # else:
            #     stride_p = 1
            # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride_p) #2D卷积通道降维
            print(n_s, '层，步长：', stride_p, stride)
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)  # 2D卷积通道降维
            self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        self.id = n_s #当前层数
        self.ST = list(self.ST_struc)[self.id % self.len_ST] #判断当前层使用哪种结构（结构A、B、C交替使用）

        #中间层卷积，通道数、时间、空间维度均不变
        if self.id < self.depth_3d: #3D卷积
            self.conv2 = conv_S(planes, planes, stride=1, padding=(0, 1, 1)) #进行空间卷积（k = 1 × 3 × 3）
            self.bn2 = nn.BatchNorm3d(planes)

            self.conv3 = conv_T(planes, planes, stride=1, padding=(1, 0, 0)) #进行时间卷积（k = 3 × 1 × 1）
            self.bn3 = nn.BatchNorm3d(planes)
        else: #2D卷积
            self.conv_normal = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn_normal = nn.BatchNorm2d(planes)

        #底层卷积，通道升维（升到planes * 4），时间、空间维度均不变
        if n_s < self.depth_3d: #3D卷积
            self.conv4 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False) #3D卷积，通道升维
            self.bn4 = nn.BatchNorm3d(planes * 4)
        else: #2D卷积
            self.conv4 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False) #2D卷积，通道降维
            self.bn4 = nn.BatchNorm2d(planes * 4)

        #激活操作
        self.relu = nn.ReLU(inplace=True)

        #self.stride = stride

    #P3D-A: T(S(x))
    def ST_A(self, x):
        x = self.conv2(x) #对输入进行空间卷积
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x) #对空间卷积结果进行时间卷积
        x = self.bn3(x)
        x = self.relu(x)

        return x  #对原始输入进行时空卷积的结果

    #P3D-B: S(x) + T(x)
    def ST_B(self, x):
        tmp_x = self.conv2(x) #对输入进行空间卷积
        tmp_x = self.bn2(tmp_x)
        tmp_x = self.relu(tmp_x)

        x = self.conv3(x) #对输入进行时间卷积
        x = self.bn3(x)
        x = self.relu(x)

        return x+tmp_x  #对原始输入进行空间卷积结果 + 对原始输入进行空间卷积结果

    #P3D-C: T(S(x)) + S(x)
    def ST_C(self, x):
        x = self.conv2(x) #对输入进行空间卷积
        x = self.bn2(x)
        x = self.relu(x)

        tmp_x = self.conv3(x) #对空间卷积后的结果进行时间卷积
        tmp_x = self.bn3(tmp_x)
        tmp_x = self.relu(tmp_x)

        return x+tmp_x  #对原始输入进行空间卷积结果 + 时空卷积结果

    def forward(self, x):
        residual = x
        #输出2D-3D转折时的特征尺寸
        if self.id == self.depth_3d:
            print('2D卷积最第一层输入尺寸：', x.shape)

        out = self.conv1(x) #点卷积，通道降维（衔接层，空间维度缩小1/2，时间维度不变；其它层各维度均不变）
        out = self.bn1(out)
        out = self.relu(out)

        if self.id == self.depth_3d:
            print('2D卷积第一层输出尺寸：', out.shape)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)

        #中间卷积，通道不变，各维度不变
        if self.id < self.depth_3d: #C3D parts:
            if self.ST == 'A':
                out = self.ST_A(out)
            elif self.ST == 'B':
                out = self.ST_B(out)
            elif self.ST == 'C':
                out = self.ST_C(out)
        else:
            out = self.conv_normal(out)   # normal is res5 part, C2D all.
            out = self.bn_normal(out)
            out = self.relu(out)

        #点卷积，通道升维，各维度不变
        out = self.conv4(out)
        out = self.bn4(out)

        #残差值
        if self.downsample is not None:
            residual = self.downsample(x)

        #加上残差值
        out += residual
        out = self.relu(out)

        return out

#P3D基本单元
class P3D(nn.Module):
    def __init__(self, block, layers, modality='RGB', shortcut_type='B', num_classes=400, dropout=0.5, ST_struc=('A','B','C')):
        super(P3D, self).__init__()
        self.inplanes = 64  #卷积层的初始输入通道数目
        # self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
        #                        padding=(3, 3, 3), bias=False)

        #两种输入数据，图像帧和光流
        self.input_channel = 3 if modality == 'RGB' else 2  # 2 is for flow

        self.ST_struc = ST_struc #P3D基本结构

        #顶层，3D卷积，进行空间下采样（缩小1/2），通道升维（初始化核心卷积层的输入数据）
        self.conv1_custom = nn.Conv3d(self.input_channel, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                                      padding=(0, 3, 3), bias=False)

        #前3层进行3D卷积，最后一层进行2D卷积（3D卷积结束后，时间维度为1，变成2D卷积）
        self.depth_3d = sum(layers[:3])# C3D layers are only (res2,res3,res4),  res5 is C2D

        #顶层批量归一化
        self.bn1 = nn.BatchNorm3d(64) # bn1 is followed by conv1

        self.cnt = 0 #当前层数

        self.relu = nn.ReLU(inplace=True)

        #顶层池化，3D最大池化，空间下采样（缩小1/2）
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=2, padding=(0, 1, 1))       # pooling layer for conv1.

        #核心3D池化层，时间下采样（缩小1/2）
        self.maxpool_2 = nn.MaxPool3d(kernel_size=(2, 1, 1), padding=0, stride=(2, 1, 1))   # pooling layer for res2, 3, 4.

        #核心3D卷积层构建
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)

        #核心2D卷积层构建
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        #核心2D池化层，平均池化，空间下采样（缩小为n-4）
        self.avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=1)                              # pooling layer for res5.

        #随机失活
        self.dropout = nn.Dropout(p=dropout)

        #尾层，全连接层
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        #3D卷积层和批归一化层的权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) #HE初始化，符合(0,2/n)的正态分布，n为输入个数？
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        #输入数据的尺寸、均值、标准差
        # some private attribute
        self.input_size = (self.input_channel, 16, 160, 160)       # input of the network
        self.input_mean = [0.485, 0.456, 0.406] if modality=='RGB' else [0.5]
        self.input_std = [0.229, 0.224, 0.225] if modality=='RGB' else [np.mean([0.229, 0.224, 0.225])]


    @property
    def scale_size(self):
        return self.input_size[2] * 256 // 160   # asume that raw images are resized (340,256).

    @property
    def temporal_length(self):
        return self.input_size[1]

    @property
    def crop_size(self):
        return self.input_size[2]

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        stride_p = stride #especially for downsample branch.

        if self.cnt < self.depth_3d: #3D卷积层
            if self.cnt == 0:
                stride_p = 1 #卷积顶层，步长为1
            else:
                stride_p = (1, 2, 2) #非顶层，空间下采样（缩小1/2）

            #不同层之间的衔接部分，需要对输入x进行变形
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A': #如果当前层使用A结构，使用池化和0值填充达到下采样和通道升维效果

                    #对shortcut_type结构使用池化操作代替点卷积操作。partial为偏函数，用于固定downsample_basic_block函数的参数。
                    downsample = partial(downsample_basic_block,
                                         planes=planes * block.expansion,
                                         stride=stride) #时间和空间均进行下采样（缩小1/2），通道升维（用0填充）
                else: #对其它结构使用点卷积，时间维度不变，空间维度下采样（缩小1/2）
                    downsample = nn.Sequential(
                        nn.Conv3d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride_p, bias=False),
                        nn.BatchNorm3d(planes * block.expansion)
                    )

        else: #2D卷积层
            #不同层之间的衔接部分，需要对输入x进行变形
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A': #应该可以省略
                    downsample = partial(downsample_basic_block,
                                         planes=planes * block.expansion,
                                         stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=2, bias=False), #stride=stride
                        nn.BatchNorm2d(planes * block.expansion)
                    )

        #创建当前层，堆叠Bottleneck块（每一个Bottleneck块的最后输出通道数都是planes * block.expansion）
        layers = []
        #本层首个Bottleneck块
        layers.append(block(self.inplanes, planes, stride, downsample, n_s=self.cnt,
                            depth_3d=self.depth_3d, ST_struc=self.ST_struc))
        self.cnt += 1 #当前层数加1

        self.inplanes = planes * block.expansion #上一块的输出通道数目，也是下一块的输入通道数目

        #本层其它Bottleneck块
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, n_s=self.cnt, depth_3d=self.depth_3d, ST_struc=self.ST_struc))
            self.cnt += 1

        #将本层所有块顺序放入Sequential容器中
        return nn.Sequential(*layers)

    def forward(self, x):
        print('第一层输入尺寸：', x.size())

        #顶层卷积，通道维度升到初始设定值64，空间维度缩小1/2（160->80)，时间维度不变
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)

        #顶层池化，时间维度缩小1/2（16->8），空间维度缩小1/2（80->40）
        x = self.maxpool(x)
        print('第一层输出尺寸：', x.size())

        #第2层卷积+池化，时间维度缩小1/2（8->4池化时缩小，卷积时不变），空间维度不变
        x = self.maxpool_2(self.layer1(x))  #  Part Res2
        print('第二层输出尺寸：', x.size())

        ##第3层卷积+池化，时间维度缩小1/2（4->2池化时缩小，卷积时不变），空间维度缩小1/2（40->20卷积时缩小，池化时不变）
        x = self.maxpool_2(self.layer2(x))  #  Part Res3
        print('第三层输出尺寸：', x.size())

        # 第3层卷积+池化，时间维度缩小1/2（2->1池化时缩小，卷积时不变），空间维度缩小1/2（20->10卷积时缩小，池化时不变）
        x = self.maxpool_2(self.layer3(x))  #  Part Res4
        print('第四层输出尺寸：', x.size())

        sizes = x.size()
        print('3D卷积最后一层输出尺寸：', sizes)

        #将5维的张量变形为4维（此方法会将不同样本的图像整合成同一个样本的不同通道，造成数据混乱；可以根据通道，依次采取每个样本的所有通道作为一个新的样本数据；
        # 但是，如果当前时间维度为1，则不会受影响，所以要根据数据样本尺寸设计合理的网络模型，使得在进行转换时的样本时间维度是1）
        x = x.view(-1, sizes[1], sizes[3], sizes[4])  #  Part Res5，将x从5维变成4维，保留原来的通道数和空间维度
        print('2D卷积第一层输入尺寸：', x.size())

        #最后一层卷积（2D卷积），空间维度缩小1/2（10->5）
        x = self.layer4(x)

        #最后一层池化，空间维度缩小为 1×1
        x = self.avgpool(x)

        #第0维自动计算，第1维变成最后的输入特征数目（512 * 4）
        x = x.view(-1, self.fc.in_features)

        #全连接，随机失活
        x = self.fc(self.dropout(x))

        return x


def P3D63(**kwargs):
    """
    Construct a P3D63 modelbased on a ResNet-50-3D model.
    """
    model = P3D(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def P3D131(**kwargs):
    """
    Construct a P3D131 model based on a ResNet-101-3D model.
    """
    model = P3D(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def P3D199(pretrained=False, modality='RGB', **kwargs):
    """
    construct a P3D199 model based on a ResNet-152-3D model.
    """
    model = P3D(Bottleneck, [3, 8, 36, 3], modality=modality, **kwargs)
    if pretrained == True:
        if modality == 'RGB':
            pretrained_file = 'p3d_rgb_199.checkpoint.pth.tar'
        elif modality == 'Flow':
            pretrained_file = 'p3d_flow_199.checkpoint.pth.tar'
        weights = torch.load(pretrained_file)['state_dict']
        model.load_state_dict(weights)
    return model

# custom operation
#为不同层设置不同的优化策略
def get_optim_policies(model=None, modality='RGB', enable_pbn=True):
    '''
    first conv:         weight --> conv weight
                        bias   --> conv bias
    normal action:      weight --> non-first conv + fc weight
                        bias   --> non-first conv + fc bias
    bn:                 the first bn2, and many all bn3.

    '''
    first_conv_weight = []
    first_conv_bias = []
    normal_weight = []
    normal_bias = []
    bn = []

    if model == None:
        # log.l.info('no model!')
        exit()

    conv_cnt = 0
    bn_cnt = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Conv2d):
            ps = list(m.parameters())
            conv_cnt += 1
            if conv_cnt == 1:
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            else:
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
        elif isinstance(m, torch.nn.Linear):
            ps = list(m.parameters())
            normal_weight.append(ps[0])
            if len(ps) == 2:
                normal_bias.append(ps[1])
              
        elif isinstance(m, torch.nn.BatchNorm3d):
            bn_cnt += 1
            # later BN's are frozen
            if not enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
        elif isinstance(m, torch.nn.BatchNorm2d):
            bn.extend(list(m.parameters()))
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    slow_rate = 0.7
    n_fore = int(len(normal_weight) * slow_rate)
    slow_feat = normal_weight[:n_fore] #finetune slowly.
    slow_bias = normal_bias[:n_fore]
    normal_feat = normal_weight[n_fore:]
    normal_bias = normal_bias[n_fore:]

    return [
        {'params': first_conv_weight, 'lr_mult': 5 if modality == 'Flow' else 1, 'decay_mult': 1,
         'name': "first_conv_weight"},
        {'params': first_conv_bias, 'lr_mult': 10 if modality == 'Flow' else 2, 'decay_mult': 0,
         'name': "first_conv_bias"},
        {'params': slow_feat, 'lr_mult': 1, 'decay_mult': 1,
         'name': "slow_feat"},
        {'params': slow_bias, 'lr_mult': 2, 'decay_mult': 0,
         'name': "slow_bias"},
        {'params': normal_feat, 'lr_mult': 1 , 'decay_mult': 1,
         'name': "normal_feat"},
        {'params': normal_bias, 'lr_mult': 2, 'decay_mult':0,
         'name': "normal_bias"},
        {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
         'name': "BN scale/shift"},
    ]



if __name__ == '__main__':

    model = P3D199(pretrained=False, num_classes=400)
    model = model.cuda()
    data = torch.autograd.Variable(torch.rand(2, 3, 16, 160, 160)).cuda()   # if modality=='Flow', please change the 2nd dimension 3==>2
    out = model(data)
    print(out.size())
