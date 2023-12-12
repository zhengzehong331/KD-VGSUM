import io
import subprocess
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, T5Tokenizer
from tqdm import tqdm
import pytorch_lightning as pl
import torch
import numpy as np
from mmcv.parallel import DataContainer as DC
from data_preprocess.video_preprocess import Collect, ColorJitter, DecordDecode, DecordInit, Flip, FormatShape, GrayScale, MultiScaleCrop, Normalize, Resize, SampleFrames, ToTensor
from utils.utils import pad_sents, get_mask
import csv
from mmcv.fileio import FileClient
from zhconv import convert
import os

# 配置图像归一化参数
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


class OurDataset(Dataset):
    """Summarization dataset"""
    """Summarization dataset"""

    def __init__(self, args, mode):
        self.args = args
        self.sample_duration=8
        if 't5' in self.args.model:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        else:
            self.tokenizer = BartTokenizer.from_pretrained(
                'facebook/bart-base')
        if mode == 'train':
            src_path = args.train_src_path
            tgt_path = args.train_tgt_path
        if mode == 'val':
            src_path = args.val_src_path
            tgt_path = args.val_tgt_path
        if mode == 'test':
            src_path = args.test_src_path
            tgt_path = args.test_tgt_path
        self.src = self.file_reader(src_path)
        self.tgt = self.file_reader(tgt_path)
        self.data_id = [item.split()[0] for item in self.tgt]
        self.video_paths = [self.args.image_feature_path+ item.split()[0]+'.mp4' for item in self.tgt]
        self.src = [" ".join(item.split()[1:]) for item in self.src]
        self.tgt = [" ".join(item.split()[1:]) for item in self.tgt]
        # get tokenized test
        print('==================== Tokening {} set ======================'.format(mode))
        self.src_ids = self.tokenize(self.src)
        self.tgt_ids = self.tokenize(self.tgt)
        if self.args.model == 'multi_modal_bart':
            print('==================== Video Prepreocess {} set ======================'.format(mode))
            self.visual_ids = self.visual_pre()
        else:
            self.visual_ids = None

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src_ids[idx], self.tgt_ids[idx], self.data_id[idx]

    def tokenize(self, data):
        tokenized_text = [self.tokenizer.encode(
            i, add_special_tokens=False) for i in tqdm(data)]
        return tokenized_text

    def file_reader(self, file_path):
        file = open(file_path, 'r')
        lines = [item.strip('\n') for item in file.readlines()]
        return lines

    def visual_pre(self):
        """视频预处理
        """
        scale_resize = (224, 224)
        vis_feature = []
        for line in tqdm(self.video_paths):
            result = {}
            result["filepath"] = line
            result["start_index"] = 1
            result["modality"] = 'RGB'

            # 初始化video_reader
            vedio_reader = DecordInit()
            vedio_reader(results=result)
            # 采样帧处理
            # sample_frames = SampleFrames(
            #     clip_len=1, frame_interval=1, num_clips=8, test_mode=False)

            sample_frames = SampleFrames(clip_len=1, frame_interval=self.args.frame_interval, test_mode=False)
            sample_frames(results=result)
            # 使用decord对视频进行解码准备
            decord_decode = DecordDecode()
            decord_decode(results=result)
            # 重新统一图像尺寸 这里将1280*720缩放为265*455（可以通过动态调整长宽缩放，从而调整缩放比例）
            resize = Resize(scale=scale_resize, keep_ratio=False)
            resize(results=result)
            # 用一个随机选择的比例列表来裁剪图像。
            # scale_crop = MultiScaleCrop(input_size=224,
            #                             scales=(1, 0.875, 0.75, 0.66),
            #                             random_crop=False,
            #                             max_wh_scale_gap=1)
            # scale_crop(results=result)
            # 以一定的概率翻转输入的图像
            filp = Flip(flip_ratio=0.5)
            filp(results=result)
            # ColorJitter是PyTorch中的一种数据增强方法，可用于对图像进行随机颜色扭曲，从而增强模型的鲁棒性和泛化能力。
            colorJitter = ColorJitter(p=0.9)
            colorJitter(results=result)
            # 将彩色图像转换为灰度图像
            gary_scale = GrayScale(p=0.2)
            gary_scale(results=result)
            # 正则化
            noprmalize = Normalize(**img_norm_cfg)
            noprmalize(results=result)
            # 将图像格式变换为最终的输入格式
            format_shape = FormatShape(input_format='NCHW')
            format_shape(results=result)
            # 可以理解成是一个监控器,可以监控读取的信息,在key中填入所需要的key
            collect = Collect(keys=['imgs'])
            collect(results=result)
            to_tensor = ToTensor(keys=['imgs'])
            to_tensor(results=result)
            vis_feature.append(result['imgs'])

        return vis_feature

    def collate_fn(self, data):
        if self.args.model == 'text_only_bart':
            # rebuild the raw text and truncate to max length
            max_input_len = self.args.max_input_len
            max_output_len = self.args.max_output_len
            raw_src = [pair[0] for pair in data]
            raw_tgt = [pair[1] for pair in data]
            raw_src = [i[:max_input_len-1] for i in raw_src]
            raw_tgt = [i[:max_output_len-1] for i in raw_tgt]
            src = []
            tgt = []
            # remove blank data
            for i in range(len(raw_src)):
                src.append(raw_src[i])
                tgt.append(raw_tgt[i])
            # make input mask
            mask = torch.tensor(get_mask(src, max_len=max_input_len))
            # make input ids
            src_ids = torch.tensor(pad_sents(src, 1, max_len=max_input_len)[0])
            # make output ids
            decoder_ids = [[0]+i for i in tgt]
            # make output labels
            label_ids = [i+[2] for i in tgt]
            decoder_ids = torch.tensor(
                pad_sents(decoder_ids, 1, max_len=max_output_len)[0])
            label_ids = torch.tensor(
                pad_sents(label_ids, -100, max_len=max_output_len)[0])

            return src_ids, decoder_ids, mask, label_ids

        elif self.args.model == 'multi_modal_bart':
            # rebuild the raw text and truncate to max length
            max_input_len = self.args.max_input_len
            max_output_len = self.args.max_output_len
            max_img_len = self.args.max_img_len
            raw_src = [pair[0] for pair in data]
            raw_tgt = [pair[1] for pair in data]
            data_id = [pair[2] for pair in data]
            raw_src = [i[:max_input_len-1] for i in raw_src]
            raw_tgt = [i[:max_output_len-1] for i in raw_tgt]
            src = []
            tgt = []
            img = np.zeros([len(raw_src),self.args.max_img_len,3,224,224])
            img_len = []
            # remove blank data
            for i in range(len(raw_src)):
                src.append(raw_src[i])
                tgt.append(raw_tgt[i])
                img[i][:self.visual_ids[i].shape[0]] = self.visual_ids[i][:self.args.max_img_len]
                img_len.append(self.visual_ids[0])

            # shape : NCHW
            # img = self.visual_ids
            # for f_visual in img:
            #     img_len.append(f_visual.shape[0])

            # img_batched_tensor = torch.stack(img)
            # img_batched_tensor = torch.cat([t.unsqueeze(0) for t in img], dim=0)
            
            # img = img[:,:max(img_len)]

            # make input mask
            mask = torch.tensor(get_mask(src, max_len=max_input_len))
            # make input ids
            src_ids = torch.tensor(pad_sents(src, 1, max_len=max_input_len)[0])
            # make output ids
            decoder_ids = [[0]+i for i in tgt]
            # make output labels
            label_ids = [i+[2] for i in tgt]
            decoder_ids = torch.tensor(
                pad_sents(decoder_ids, 1, max_len=max_output_len)[0])
            label_ids = torch.tensor(
                pad_sents(label_ids, -100, max_len=max_output_len)[0])
            return src_ids, decoder_ids, mask, label_ids, torch.tensor(img), img_len

        elif self.args.model == 'text_only_t5':
            # rebuild the raw text and truncate to max length
            max_input_len = self.args.max_input_len
            max_output_len = self.args.max_output_len
            raw_src = [pair[0] for pair in data]
            raw_tgt = [pair[1] for pair in data]
            raw_src = [i[:max_input_len-1] for i in raw_src]
            raw_tgt = [i[:max_output_len-1] for i in raw_tgt]
            src = []
            tgt = []
            # remove blank data
            for i in range(len(raw_src)):
                src.append(raw_src[i])
                tgt.append(raw_tgt[i])
            # make input mask
            mask = torch.tensor(get_mask(src, max_len=max_input_len))
            # make input ids
            src_ids = torch.tensor(pad_sents(src, 0, max_len=max_input_len)[0])
            # make output ids
            decoder_ids = [[0]+i for i in tgt]
            # make output labels
            label_ids = [i+[1] for i in tgt]
            decoder_ids = torch.tensor(
                pad_sents(decoder_ids, 0, max_len=max_output_len)[0])
            label_ids = torch.tensor(
                pad_sents(label_ids, -100, max_len=max_output_len)[0])

            return src_ids, decoder_ids, mask, label_ids

        elif self.args.model == 'multi_modal_t5':
            # rebuild the raw text and truncate to max length
            max_input_len = self.args.max_input_len
            max_output_len = self.args.max_output_len
            max_img_len = self.args.max_img_len
            raw_src = [pair[0] for pair in data]
            raw_tgt = [pair[1] for pair in data]
            data_id = [pair[2] for pair in data]
            raw_src = [i[:max_input_len-1] for i in raw_src]
            raw_tgt = [i[:max_output_len-1] for i in raw_tgt]
            src = []
            tgt = []
            img = np.zeros([len(raw_src), self.args.max_img_len, 2048])
            img_len = []
            # remove blank data
            for i in range(len(raw_src)):
                src.append(raw_src[i])
                tgt.append(raw_tgt[i])
                if self.args.vision_use_noise:
                    image_feature = np.load(
                        self.args.image_feature_path + data_id[i] + '_noise.npy')[:max_img_len]
                else:
                    image_feature = np.load(
                        self.args.image_feature_path + data_id[i] + '.npy')[:max_img_len]
                # image_feature = np.load(self.args.image_feature_path + data_id[i]+ '.npy')[:max_img_len]
                img[i][:image_feature.shape[0]] = image_feature
                img_len.append(image_feature.shape[0])
            img = img[:, :max(img_len)]

            # make input mask
            mask = torch.tensor(get_mask(src, max_len=max_input_len))
            # make input ids
            src_ids = torch.tensor(pad_sents(src, 0, max_len=max_input_len)[0])
            # make output ids
            decoder_ids = [[0]+i for i in tgt]
            # make output labels
            label_ids = [i+[1] for i in tgt]
            decoder_ids = torch.tensor(
                pad_sents(decoder_ids, 0, max_len=max_output_len)[0])
            label_ids = torch.tensor(
                pad_sents(label_ids, -100, max_len=max_output_len)[0])
            return src_ids, decoder_ids, mask, label_ids, torch.tensor(img), img_len

        else:
            raise ValueError("Invalid model")

# Create a dataloading module as per the PyTorch Lightning Docs


class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        # scale_resize = int(256 / 224 * args.input_size)
        train_set = OurDataset(args, 'train')
        val_set = OurDataset(args, 'val')
        test_set = OurDataset(args, 'test')
        self.train_loader = DataLoader(dataset=train_set,
                                       batch_size=args.batch_size,
                                       num_workers=3,
                                       shuffle=True,
                                       collate_fn=train_set.collate_fn)
        self.val_loader = DataLoader(dataset=val_set,
                                     batch_size=args.batch_size,
                                     num_workers=3,
                                     shuffle=False,
                                     collate_fn=val_set.collate_fn)
        self.test_loader = DataLoader(dataset=test_set,
                                      batch_size=args.batch_size,
                                      num_workers=3,
                                      shuffle=False,
                                      collate_fn=test_set.collate_fn)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
