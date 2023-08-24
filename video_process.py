import os
from pathlib import Path

import random

import numpy as np
import pickle as pk
import cv2
from tqdm import tqdm
from PIL import Image

import torchvision.transforms as transforms
import torch

from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset


class VideoDataset(Dataset):

    def __init__(self, directory_list, local_rank=0, enable_GPUs_num=0, distributed_load=False, resize_shape=[224, 224],
                 mode='train', clip_len=32, crop_size=168):

        self.clip_len = clip_len #剪辑后的视频帧数
        self.crop_size = crop_size #每一帧图像裁剪后的尺寸
        self.resize_shape = resize_shape #重置每一帧图像大小
        self.mode = mode #数据种类（train/test/validate）

        self.fnames, labels = [], [] #存储视频数据和类别标签
        # get the directory of the specified split

        #获取文件路径和对应标签
        for directory in directory_list:
            folder = Path(directory) #获取数据根目录对象
            print("Load dataset from folder : ", folder)
            for label in sorted(os.listdir(folder)): #获取标签名（标签名为根目录下的子目录名）
                #获取每个子文件夹下的文件路径和对应标签
                for fname in os.listdir(os.path.join(folder, label)) if mode == "train" else os.listdir(
                        os.path.join(folder, label))[:10]:
                    self.fnames.append(os.path.join(folder, label, fname)) #当前文件路径
                    labels.append(label) #当前文件标签

        random_list = list(zip(self.fnames, labels)) #将文件路径—标签映射成元组列表[(文件路径1, 标签1), (文件路径2, 标签2), ...]
        random.shuffle(random_list) #打乱顺序
        self.fnames[:], labels[:] = zip(*random_list) #重新打包

        # self.fnames = self.fnames[:240]

        #分布式加载训练集
        if mode == 'train' and distributed_load:
            single_num_ = len(self.fnames) // enable_GPUs_num
            self.fnames = self.fnames[local_rank * single_num_:((local_rank + 1) * single_num_)]
            labels = labels[local_rank * single_num_:((local_rank + 1) * single_num_)]

        # prepare a mapping between the label names (strings) and indices (ints)
        #创建将标签名-数字编号的映射（101个标签对应0-100）
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}

        # convert the list of label names into an array of label indices
        #根据标签名列表labels创建数字编号列表
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

    #遍历DataLoader时，获取对应数据函数
    def __getitem__(self, index):
        '''
        :param index: 要获取的数据索引
        '''
        # loading and preprocessing. TODO move them to transform classess
        #将索引为index的数据加载到buffer中，buffer是np.array类型数组(c × t × h × w)
        buffer = self.loadvideo(self.fnames[index])

        #每一帧图片随机裁剪，剪裁范围和大小为（height_index ~ height_index + self.crop_size, width_index ~ width_index + self.crop_size）
        height_index = np.random.randint(buffer.shape[2] - self.crop_size)
        width_index = np.random.randint(buffer.shape[3] - self.crop_size)

        #返回裁剪后的数据、对应标签数字编号
        return buffer[:, :, height_index:height_index + self.crop_size, width_index:width_index + self.crop_size], \
        self.label_array[index]

    #获取数据总量
    def __len__(self):
        return len(self.fnames)

    #将视频fname加载成3维张量数据
    def loadvideo(self, fname):
        '''
        :param fname: 视频文件路径
        '''

        # initialize a VideoCapture object to read video data into a numpy array
        #数据处理集合
        self.transform = transforms.Compose([
            transforms.Resize([self.resize_shape[0], self.resize_shape[1]]), #重置每一帧图像大小
            transforms.ToTensor(), #转换成张量（数据范围：0.0-1.0）
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) #标准化
        ])

        flip = True #是否对每一帧图像进行翻转
        #翻转方式：1——水平翻转，0——垂直翻转，-1——水平垂直翻转
        flipCode = random.choice([-1, 0, 1]) if np.random.random() < 0.5 and self.mode == "train" else 0

        #加载视频流，创建cv2.VideoCapture实例对象
        try:
            video_stream = cv2.VideoCapture(fname) #创建一个视频对象（加载视频）
            frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT)) #获取视频帧数
        except RuntimeError: #如果指定视频加载错误，重新随机选择另一个视频进行加载
            index = np.random.randint(self.__len__())
            video_stream = cv2.VideoCapture(self.fnames[index])
            frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        #如果视频帧数小于或等于（剪辑长度+1），则重新随机选择另一个视频进行加载
        while frame_count < self.clip_len + 2:
            index = np.random.randint(self.__len__())
            video_stream = cv2.VideoCapture(self.fnames[index])
            frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))

        #对时间维度(帧）进行裁剪，裁剪范围（start_idx ~ end_idx），裁剪大小为self.clip_len帧。
        speed_rate = np.random.randint(1, 3) if frame_count > self.clip_len * 2 + 2 else 1 #控制裁剪范围
        time_index = np.random.randint(frame_count - self.clip_len * speed_rate) #裁剪开始的帧id（从哪一帧开始裁剪）

        start_idx = time_index #裁剪开始帧id
        end_idx = time_index + (self.clip_len * speed_rate) #裁剪结束帧id
        final_idx = frame_count - 1 #视频最后一帧id（没有用，可删除）

        #count——当前遍历的帧id，sample_count——已经选择的帧数目，retaining——当前帧是否读取成功
        count, sample_count, retaining = 0, 0, True

        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        #创建一个空数组，用于存储处理后的数据
        buffer = np.empty((self.clip_len, 3, self.resize_shape[0], self.resize_shape[1]), np.dtype('float16'))

        #数据读取与处理（时间裁剪和空间放缩）
        while (count <= end_idx and retaining):
            retaining, frame = video_stream.read() #读取当前帧
            if count < start_idx: #start_idx之前的帧都不用
                count += 1
                continue
            #从一帧开始，每隔speed_rate帧进行一次帧选取
            if count % speed_rate == speed_rate - 1 and count >= start_idx and sample_count < self.clip_len:
                if flip:
                    frame = cv2.flip(frame, flipCode=flipCode) #对选取的帧进行旋转
                try:
                    #转换色彩空间、将numpy数组转换image数组、处理当前帧（transform）、转换为tensor张量
                    buffer[sample_count] = self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                except cv2.error as err:
                    continue

                sample_count += 1 #已选择的帧数加1
            count += 1 #已读取的帧数加1

        video_stream.release() #释放cv2.VideoCapture对象空间

        #维度调换（t × c × h × w -> c × t × h × w）
        return buffer.transpose((1, 0, 2, 3))


if __name__ == '__main__':

    datapath = ['/root/data1/datasets/UCF-101']

    #从本地加载数据，封装到Dataset对象中
    dataset = VideoDataset(datapath,
                           resize_shape=[224, 224],
                           mode='validation')

    #将Dataset数据加载到DataLoader对象
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=24, pin_memory=True)

    #创建进度条
    bar = tqdm(total=len(dataloader), ncols=80)

    #从DataLoader对象中数据加载
    for step, (buffer, labels) in enumerate(dataloader):
        print(buffer.shape)
        print("label: ", labels)
        bar.update(1)


    # prefetcher = DataPrefetcher(BackgroundGenerator(dataloader), 0)
    # batch = prefetcher.next()
    # iter_id = 0
    # while batch is not None:
    #     iter_id += 1
    #     bar.update(1)
    #     if iter_id >= len(dataloader):
    #         break
    #
    #     batch = prefetcher.next()
    #     print(batch[0].shape)
    #     print("label: ", batch[1])

    # for step, (buffer, labels) in enumerate(BackgroundGenerator(dataloader)):
    #     print(buffer.shape)
    #     print("label: ", labels)
    #     bar.update(1)

