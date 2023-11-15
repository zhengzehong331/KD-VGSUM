import io
import subprocess
import whisper
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


# from .pipeline import Compose

# 配置图像归一化参数
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


class OurDataset(Dataset):
    """Summarization dataset"""

    def __init__(self, args, mode):

        self.args = args
        self.start_index = 1
        self.whisper_model = whisper.load_model("base")

        # self.pipeline = Compose(pipeline)

        # initial tokenizer and text
        if 't5' in self.args.model:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        else:
            self.tokenizer = BartTokenizer.from_pretrained(
                'facebook/bart-base')
        if mode == 'train':
            src_path = args.train_src_path
            # tgt_path = args.train_tgt_path
        if mode == 'val':
            src_path = args.val_src_path
            # tgt_path = args.val_tgt_path
        if mode == 'test':
            src_path = args.test_src_path
            # tgt_path = args.test_tgt_path
        self.lines = self.file_reader(src_path)
        # 除去错误数据
        self.lines = [line for line in self.lines if os.path.exists("data/"+line["video_path"])]

        # 文件检查损坏检查
        # print("================Check Broken Video====================")
        # self.lines = [line for line in tqdm(self.lines) if not self.is_video_corrupted("data/"+line["video_path"])]

        self.tgt = [line["tgt"] for line in self.lines]
        self.data_id = [line["data_id"] for line in self.lines]
        self.src = [line["transport"] for line in self.lines]
        # print('==================== Transcription {} video ======================'.format(mode))
        # self.src = self.transcription(self.lines)
        print('==================== Tokening {} set ======================'.format(mode))
        self.src_ids = self.tokenize(self.src)
        self.tgt_ids = self.tokenize(self.tgt)

        if self.args.model == 'multi_modal_bart':
            print(
                '==================== Video Prepreocess {} set ======================'.format(mode))
            self.visual_ids = self.visual_pre(self.lines)
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
        lines = []
        with open(file_path, 'r') as fin:
            csv_reader = csv.reader(fin)
            next(csv_reader, None)
            for line in csv_reader:
                if line is not None:
                    data_id = line[0]
                    tgt = line[1]
                    video_path = line[2]
                    transport = line[3]
                lines.append(dict(data_id=data_id,
                                  tgt=tgt, video_path=video_path,transport=transport))
        return lines
    
    # 检查文件是否损坏函数
    def is_video_corrupted(self,file_path):
        try:
            # 使用ffprobe的命令行工具进行文件检测
            subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries','stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1', file_path],stderr=subprocess.STDOUT)
            return False  # 如果没有抛出异常，文件没有损坏
        except subprocess.CalledProcessError as e:
            return True  # 文件损坏

    
    def transcription(self, path):
        """进行视频语音提取操作，并生成字幕
        """
        src = []
        for line in tqdm(self.lines):
            path = "data/"+line["video_path"]
            tran = self.whisper_model.transcribe(
                path, fp16=False, language='Chinese')
            ch_transcription = convert(tran["text"], 'zh-cn')
            src.append(ch_transcription)
        return src

    def visual_pre(self, path):
        """视频预处理
        """
        scale_resize = (224, 224)
        vis_feature = []
        for line in tqdm(self.lines):
            result = {}
            result["filepath"] = line["video_path"]
            result["start_index"] = 1
            result["modality"] = 'RGB'

            # 初始化video_reader
            vedio_reader = DecordInit()
            vedio_reader(results=result)
            # 采样帧处理
            sample_frames = SampleFrames(
                clip_len=1, frame_interval=1, num_clips=8, test_mode=False)
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
            # self.src_ids[idx], self.tgt_ids[idx], self.data_id[idx]
            raw_src = [pair[0] for pair in data]
            raw_tgt = [pair[1] for pair in data]
            data_id = [pair[2] for pair in data]
            raw_src = [i[:max_input_len-1] for i in raw_src]
            raw_tgt = [i[:max_output_len-1] for i in raw_tgt]
            src = []
            tgt = []
            # img = np.zeros([len(raw_src), self.args.max_img_len, 2048])
            img_len = []
            # remove blank data
            for i in range(len(raw_src)):
                src.append(raw_src[i])
                tgt.append(raw_tgt[i])
                # 这里固定只取8帧，我们也可以设置为每8帧获取一次图像
                img_len.append(8)
                # image_feature = np.load(
                #     self.args.image_feature_path + data_id[i] + '.npy')[:max_img_len]
                # img[i][:image_feature.shape[0]] = image_feature
                # img_len.append(image_feature.shape[0])
                # img_len.append()
            # img = img[:, :max(img_len)]

            # shape : NCHW
            img = self.visual_ids
            # img_len = [i.shape[1]*i.shape[2]*i.shape[3] for i in img]
            # img_width_len = max([i.shape[3] for i in img])
            # img_height_len = max([i.shape[2] for i in img])
            # for i in range(len(img)):
            #     img[i] = img[i][:, :, :img_height_len, :img_width_len]
            img_batched_tensor = torch.stack(img)
            img_batched_tensor = torch.cat(
                [t.unsqueeze(0) for t in img], dim=0)

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
            # return src_ids, decoder_ids, mask, label_ids, torch.tensor(img), img_len
            return src_ids, decoder_ids, mask, label_ids, img_batched_tensor, img_len

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
