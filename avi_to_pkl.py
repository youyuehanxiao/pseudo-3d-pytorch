import os
from pathlib import Path
import random
import cv2
import numpy as np
import pickle as pk
from tqdm import tqdm
from PIL import Image

import multiprocessing
import time

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class VideoDataset(Dataset):

    def __init__(self, directory, local_rank, num_local_rank, resize_shape=[168, 168], mode='val', clip_len=8,
                 frame_sample_rate=2):
        folder = Path(directory)  # get the directory of the specified split
        print("Load dataset from folder : ", folder)
        self.clip_len = clip_len
        self.resize_shape = resize_shape

        self.frame_sample_rate = frame_sample_rate
        self.mode = mode

        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder))[:200]:
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)
        '''
        random_list = list(zip(self.fnames, labels))
        random.shuffle(random_list)
        self.fnames[:], labels[:] = zip(*random_list)
        '''
        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        label_file = str(len(os.listdir(folder))) + 'class_labels.txt'
        with open(label_file, 'w') as f:
            for id, label in enumerate(sorted(self.label2index)):
                f.writelines(str(id + 1) + ' ' + label + '\n')
        if mode == 'train' or 'val' and num_local_rank > 1:
            single_num_ = len(self.fnames) // 24
            self.fnames = self.fnames[local_rank * single_num_:((local_rank + 1) * single_num_)]
            labels = labels[local_rank * single_num_:((local_rank + 1) * single_num_)]

        for file in tqdm(self.fnames, ncols=80):
            fname = file.split("/")
            self.directory = '/root/dataset/{}/{}'.format(fname[-3], fname[-2])

            if os.path.exists('{}/{}.pkl'.format(self.directory, fname[-1])):
                continue
            else:
                capture = cv2.VideoCapture(file)
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count > self.clip_len:
                    buffer = self.loadvideo(capture, frame_count, file)
                else:
                    while frame_count < self.clip_len:
                        index = np.random.randint(self.__len__())
                        capture = cv2.VideoCapture(self.fnames[index])
                        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                        buffer = self.loadvideo(capture, frame_count, file)

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        return index

    def __len__(self):
        return len(self.fnames)

    def loadvideo(self, capture, frame_count, fname):
        # initialize a VideoCapture object to read video data into a numpy array
        self.transform_nor = transforms.Compose([
            transforms.Resize([224, 224]),
        ])

        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        start_idx = 0
        end_idx = frame_count - 1
        frame_count_sample = frame_count // self.frame_sample_rate - 1
        if frame_count > 300:
            end_idx = np.random.randint(300, frame_count)
            start_idx = end_idx - 300
            frame_count_sample = 301 // self.frame_sample_rate - 1
        buffer_normal = np.empty((frame_count_sample, 224, 224, 3), np.dtype('uint8'))

        count = 0
        retaining = True
        sample_count = 0

        # read in each frame, one at a time into the numpy buffer array
        while (count <= end_idx and retaining):
            retaining, frame = capture.read()
            if count < start_idx:
                count += 1
                continue

            if retaining is False or count > end_idx:
                break

            if count % self.frame_sample_rate == (self.frame_sample_rate - 1) and sample_count < frame_count_sample:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                buffer_normal[sample_count] = self.transform_nor(frame)

                sample_count += 1
            count += 1

        fname = fname.split("/")
        self.directory = '/root/dataset/{}/{}'.format(fname[-3], fname[-2])
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        # Save tensor to .pkl file
        with open('{}/{}.pkl'.format(self.directory, fname[-1]), 'wb') as Normal_writer:
            pk.dump(buffer_normal, Normal_writer)

        capture.release()

        return buffer_normal


if __name__ == '__main__':

    datapath = '/root/dataset/UCF101'
    process_num = 24

    for i in range(process_num):
        p = multiprocessing.Process(target=VideoDataset, args=(datapath, i, process_num))
        p.start()

    print('CPU core number:' + str(multiprocessing.cpu_count()))

    for p in multiprocessing.active_children():
        print('子进程' + p.name + ' id: ' + str(p.pid))
    print('all done')
