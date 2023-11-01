import io
import os
import os.path as osp
import shutil
import warnings
from collections.abc import Sequence
import cv2
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import Dataset
import copy
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
import mmcv
import numpy as np
import torch
import tarfile
from torch.nn.modules.utils import _pair
import random
import torchvision
from PIL import Image
from .rand_augment import rand_augment_transform
from torchvision import transforms
from mmcv.fileio import FileClient
import whisper
from mmcv.parallel import DataContainer as DC

PIPELINES = Registry('pipeline')


def _init_lazy_if_proper(results, lazy):
    """初始化懒加载

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if 'img_shape' not in results:
        results['img_shape'] = results['imgs'][0].shape[:2]
    if lazy:
        if 'lazy' not in results:
            img_h, img_w = results['img_shape']
            lazyop = dict()
            lazyop['original_shape'] = results['img_shape']
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            lazyop['flip'] = False
            lazyop['flip_direction'] = None
            lazyop['interpolation'] = None
            results['lazy'] = lazyop
    else:
        assert 'lazy' not in results, 'Use Fuse after lazy operations'


def _pil_interp(method):
    if method == "bicubic":
        return Image.BICUBIC
    elif method == "lanczos":
        return Image.LANCZOS
    elif method == "hamming":
        return Image.HAMMING
    else:
        return Image.BILINEAR


class Compose:
    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


@PIPELINES.register_module()
class DecordInit:
    '''
    Decord: https://github.com/dmlc/decord
    使用 decord 初始化video_reader
    所需键值为 "fileId"
    添加与修改的key为 video_reader 和 total_frames
    '''

    def __init__(self, io_backend='disk', num_threads=1, **kwargs):
        self.io_backend = io_backend
        self.num_threads = num_threads
        self.kwargs = kwargs
        self.file_client = None
        self.tarfile = None

    def __call__(self, results):
        '''
        执行Decord初始化
        Args:
        results (dict): 将修改得到的 dict 给管道中的下一个转换。
        '''
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')
        # 两种形式的处理
        # 1. 直接处理tar
        # 2. 直接处理视频文件
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        file_obj = io.BytesIO(self.file_client.get(
            "data/"+results['filepath']))
        # else:
        #     if self.tarfile is None:
        #         data_root = os.path.dirname(results['fileId']) + '.tar'
        #         self.tarfile = tarfile.open(data_root)
        #     video_name = results['fileId'].split('/')[-1]
        #     iob = self.tarfile.extractfile(video_name)
        #     iob = iob.read()
        #     file_obj = io.BytesIO(iob)
        container = decord.VideoReader(file_obj, num_threads=self.num_threads)
        results['video_reader'] = container
        results['total_frames'] = len(container)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'num_threads={self.num_threads})')
        return repr_str


@PIPELINES.register_module()
class SampleFrames:
    """
    采样视频帧
    输入参数为为 "total_frames（总帧数）"、"start_index（起始索引）"，添加或修改的键为
    是 "frame_inds"、"frame_interval "和 "num_clips"。

    参数：
     clip_len （int）： 每个采样输出片段的帧数。
        frame_interval（int）帧间距： 相邻采样帧的时间间隔。
            默认值：1。
        num_clips (int)： 要采样的片段数量。默认值：1。
        temporal_jitter (bool): 时间抖动： 是否应用时间抖动。
            默认值：1： 假。
        twice_sample（bool）两次采样： 测试时是否使用两次采样。
            如果设置为 "True"，它将对有固定偏移和无固定偏移的帧进行采样、
            这通常用于 TSM 模型的测试。默认值为 True： 假。
        out_of_bound_opt（str）： 处理越界帧
            索引的方式。可用选项包括 "loop（循环）"、"repeat_last（重复最后一次）"。
            默认值："loop"（循环）。
        test_mode (bool)： 构建测试或验证数据集时存储为 True。
            默认值："test_mode"： 假。
        start_index（无）： 该参数已被弃用，并移至数据集
            类（``BaseDataset``, ``VideoDatset``, ``RawframeDataset``等）、
            请参见：https://github.com/open-mmlab/mmaction2/pull/89.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None,
                 frame_uniform=False,
                 multiview=1):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.frame_uniform = frame_uniform
        self.multiview = multiview
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_clips(self, num_frames):
        """在训练阶段下获取clip偏移。


        它将计算所选帧的平均间隔、并在 [0, avg_interval] 之间的偏移范围内随机移动它们。如果帧的总数小于 clips_num 或 原始帧的长度，它将返回所有零索引。

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: 生成采样帧的索引
            例如：需要获取8个clip的索引，那么输出为array([ 169,  546,  837, 1246, 1407, 1761, 2232, 2538])这样一个np.ndarray
        """
        # 每个采样输出片段的帧数 x 采样帧的间距
        # 传入参数为clip_len=1,frame_intervala=1说明没有设置帧间距，片段长度为1，即一帧为一个片段
        ori_clip_len = self.clip_len * self.frame_interval
        # 计算平均间隔，
        # 例如：共有2617帧，传入的总片段为8，所以共有8个采样片段，那么平均间隔为327
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            # 获取碎解采样
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)

        return clip_offsets

    def _sample_clips(self, num_frames):
        """为指定模式下的视频选择片段偏移。

        Args:
            num_frames (int): 视频总帧数

        Returns:
            np.ndarray: 采样帧的索引
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            if self.multiview == 1:
                clip_offsets = self._get_train_clips(num_frames)
            else:
                clip_offsets = np.concatenate(
                    [self._get_train_clips(num_frames) for _ in range(self.multiview)])

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        return clip_offsets

    def get_seq_frames(self, num_frames):
        """
        Modified from https://github.com/facebookresearch/SlowFast/blob/64abcc90ccfdcbb11cf91d6e525bed60e92a8796/slowfast/datasets/ssv2.py#L159
        给定视频索引，返回采样帧索引列表
        Args:
            num_frames (int):  视频总帧数
        Returns:
            seq (list):  是从视频中采样的帧的索引
        """
        seg_size = float(num_frames - 1) / self.clip_len
        seq = []
        for i in range(self.clip_len):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if not self.test_mode:
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)

        return np.array(seq)

    def __call__(self, results):
        """执行样本帧加载。
        参数： result字典，由上一个pipline接受并且传给下一个pipline

        Returns:
        frame_inds
        clip_len
        frame_interval
        num_clips
        """
        total_frames = results['total_frames']
        if self.frame_uniform:  # sthv2 sampling strategy
            assert results['start_index'] == 0
            frame_inds = self.get_seq_frames(total_frames)
        else:
            clip_offsets = self._sample_clips(total_frames)
            frame_inds = clip_offsets[:, None] + np.arange(
                self.clip_len)[None, :] * self.frame_interval
            frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        # 将数组reshape为一个二维数组，其中每一行包含 self.clip_len 个帧索引。
        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index

        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results


@PIPELINES.register_module()
class DecordDecode():
    """
    使用decord对视频进行解码准备，
    Decord: https://github.com/dmlc/decord

    需要使用"video_reader","filename"和"frame_inds"参数
    添加与修改`result`中的`img`与`origin_shape`、`img_shape`参数
    img: 指定帧的W*H*C数据
    origin_shape: 视频帧的维度
    img_shape: 图像的维度
    """

    def __call__(self, results):
        """
        执行Decord解码
        """
        # 使用前面init的解码容器
        container = results['video_reader']
        # .ndim 属性用于获取数组的维度数，判断该数组是维度是否为1
        if results['frame_inds'].ndim != 1:
            # 若不为1，则将该数组将降维
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        # 按顺序生成帧索引映射
        frame_dict = {
            idx: container[idx].asnumpy()
            for idx in np.unique(frame_inds)
        }

        # 获取了指定帧的数据，以weibo数据为例，其是一个(720, 1280, 3)的numpy数组
        # 三个参数分别是W*H*channel(颜色通道数)
        imgs = [frame_dict[idx] for idx in frame_inds]

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@PIPELINES.register_module()
class Resize:
    """
    将图像调整到特定大小

    Args:
        scale:  如果 keep_ratio 为 True，它将作为缩放系数或最大尺寸。如果是个浮点数，图像将按该系数重新缩放
        否则，如果它是一个包含 2 个整数的元组，图像将在缩放范围内尽可能大地重新缩放。将在缩放范围内尽可能大地重新缩放，否则，它将作为输出大小的（w，h）。

        keep_ratio （bool）： 如果设置为 True，将在不改变纵横比的情况下调整图像大小。否则，它将把图像调整为 定的大小。默认值为 True： 默认为 True。

        interpolation (str): interpolation的算法，参数可填nearest" | "bilinear". Default: "bilinear".

        lazy （bool）： 决定是否应用懒惰操作。默认值为：False
    """

    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear',
                 lazy=False):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(
                    f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # 将 np.inf 赋值给长边，以便稍后重新缩放短边
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.lazy = lazy

    def _resize_imgs(self, imgs, new_w, new_h):
        return [
            # mmcv.imresize 函数将每个输入图像 img 调整到指定的新尺寸 (new_w, new_h)，并返回调整大小后的图像。
            mmcv.imresize(
                img, (new_w, new_h), interpolation=self.interpolation)
            for img in imgs
        ]

    @staticmethod
    def _resize_kps(kps, scale_factor):
        return kps * scale_factor

    @staticmethod
    def _box_resize(box, scale_factor):
        """Rescale the bounding boxes according to the scale_factor.

        Args:
            box (np.ndarray): The bounding boxes.
            scale_factor (np.ndarray): The scale factor used for rescaling.
        """
        assert len(scale_factor) == 2
        scale_factor = np.concatenate([scale_factor, scale_factor])
        return box * scale_factor

    def __call__(self, results):
        """
        执行Resize操作
        """
        # 是否执行懒加载
        _init_lazy_if_proper(results, self.lazy)

        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')
        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)

        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            # 使用 mmcv 库中的 rescale_size 函数来根据给定的缩放比例（只能按照原始比例缩放）
            # https://mmcv.readthedocs.io/zh_CN/latest/api/generated/mmcv.image.imresize.html#mmcv.image.imresize
            new_w, new_h = mmcv.rescale_size(
                (img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale
        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor']*self.scale_factor
        if not self.lazy:
            if 'imgs' in results:
                results['imgs'] = self._resize_imgs(results['imgs'], new_w,
                                                    new_h)
            if 'keypoint' in results:
                results['keypoint'] = self._resize_kps(results['keypoint'],
                                                       self.scale_factor)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')
            lazyop['interpolation'] = self.interpolation

        if 'gt_bboxes' in results:
            assert not self.lazy
            results['gt_bboxes'] = self._box_resize(results['gt_bboxes'],
                                                    self.scale_factor)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_resize(
                    results['proposals'], self.scale_factor)
        # 展示图片
        # for i, image_np in enumerate(results['imgs']):
        #     result_dir = '/home/zehong/Desktop/NLP/VG-SUM/result/image'
        #     save_path = os.path.join(result_dir, f'image_{i + 1}.jpg')
        #     # 保存图像到指定路径
        #     image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGRA)
        #     cv2.imwrite(save_path, image_np)
        #     print(f'Image {i + 1} saved to: {save_path}')
        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'flip_label_map={self.flip_label_map}, lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class Whisper:
    """进行视频语音提取操作，并生成字幕
    """

    def __init__(self, lazy=False, io_backend='disk', **kwargs):
        self.lazy = lazy
        self.kwargs = kwargs
        self.whisper_model = whisper.load_model("base")
        self.file_client = None
        self.io_backend = io_backend

    def __call__(self, results):
        from zhconv import convert
        if results['tar'] is False:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend, **self.kwargs)
            # file_obj = io.BytesIO(self.file_client.get(
            #     "data/"+results['filepath']))
            file_path = "data/"+results['filepath']
        else:
            if self.tarfile is None:
                data_root = os.path.dirname(results['fileId']) + '.tar'
                self.tarfile = tarfile.open(data_root)
        subtitle = self.whisper_model.transcribe(
            file_path, fp16=False, language='Chinese')
        ch_sub = convert(subtitle["text"], 'zh-cn')
        results["subtitle"] = ch_sub
        return results


@PIPELINES.register_module()
class RandomCrop:
    '''直接进行随机裁剪，特定输出尺寸
    结果中的必填键是 "img_shape"、"keypoint"（可选）、"imgs"（可选），
    添加或修改的键是 "keypoint"、"imgs"、"lazy"；

    "lazy "中的必填键是 "flip"、"crop_bbox"，
    添加或修改的键是 "crop_bbox"。  

    Args:
    size (int): 输出维度.
    lazy (bool):是否使用懒加载. Default: False.     
    '''

    def __init__(self, size, lazy=False):
        if not isinstance(size, int):
            raise TypeError(f'Size must be an int, but got {type(size)}')
        self.size = size
        self.lazy = lazy

    @staticmethod
    def _crop_kps(kps, crop_bbox):
        return kps - crop_bbox[:2]

    @staticmethod
    def _crop_imgs(imgs, crop_bbox):
        x1, y1, x2, y2 = crop_bbox
        return [img[y1:y2, x1:x2] for img in imgs]

    @staticmethod
    def _box_crop(box, crop_bbox):
        """根据 crop_bbox 裁剪边框。
        Args:
            box (np.ndarray): 绑定 boxes.
            crop_bbox(np.ndarray):用于裁剪原始图像的 bbox
        """

        x1, y1, x2, y2 = crop_bbox
        img_w, img_h = x2 - x1, y2 - y1

        box_ = box.copy()
        box_[..., 0::2] = np.clip(box[..., 0::2] - x1, 0, img_w - 1)
        box_[..., 1::2] = np.clip(box[..., 1::2] - y1, 0, img_h - 1)
        return box_

    def _all_box_crop(self, results, crop_bbox):
        """根据 crop_bbox 裁剪 结果集中的gt_bboxes和proposals

        Args:
            results (dict): 采样帧中的所有信息, 包括
                'gt_bboxes' 和 'proposals' (optional).
            crop_bbox(np.ndarray): 用于裁剪原始图像的 bbox。
        """
        results['gt_bboxes'] = self._box_crop(results['gt_bboxes'], crop_bbox)
        if 'proposals' in results and results['proposals'] is not None:
            assert results['proposals'].shape[1] == 4
            results['proposals'] = self._box_crop(results['proposals'],
                                                  crop_bbox)
        return results

    def __call__(self, results):
        """执行 RandomCrop 操作.

        Args:
            results (dict): 结果集
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = self.size / img_w, self.size / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_x_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        new_h, new_w = self.size, self.size

        crop_bbox = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['crop_bbox'] = crop_bbox

        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        # Process entity boxes
        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(size={self.size}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class MultiScaleCrop(RandomCrop):
    '''使用随机选择的比例列表裁剪图像。
    从刻度列表中随机选择 w 和 h 刻度。比例为 1 表示基本尺寸，即图像宽度和高度的最小值。
    w 和 h 的缩放级别会被控制为小于某个值，以防止纵横比过大或过小。

    必填键为 "img_shape"、"imgs"（可选）、"keypoint"（可选），添加或修改的键为 "imgs"、"crop_bbox"、"img_shape"、"lazy "和 "scales"。
    懒加载模式中的必填键是 "crop_bbox"，添加或修改的键是 "crop_bbox"。

    Args:
    input_size (int | tuple[int]): (w, h) 神经网络的输入
    scales (tuple[float]): 选择宽度和高度比例。（随机选择其中一个）
    max_wh_scale_gap (int): w 级和 h 级刻度的最大间距。
        Default: 1.
    random_crop (bool): 如果设置为 "True"，裁剪 bbox 将随机采样，否则将从固定区域采样。采样，否则将从固定区域采样。
        Default: False.
    num_fixed_crops (int): 如果设置为 5，裁剪 bbox 将保留 5 个
        基本固定区域： 左上"、"右上"、"左下"、"右下"、"中间"、
        "右下"、"中央"。如果设置为 13，裁剪方框将附加 8 个固定区域： "左中"、"右中
        "下居中"、"上居中"、"左上四分之一"、
        "右上四分之一"、"左下四分之一"、"右下四分之一"。
        Default: 5.
    lazy (bool):是否采用懒加载. Default: False.
    '''

    def __init__(self,
                 input_size,
                 scales=(1, ),
                 max_wh_scale_gap=1,
                 random_crop=False,
                 num_fixed_crops=5,
                 lazy=False):
        self.input_size = _pair(input_size)
        if not mmcv.is_tuple_of(self.input_size, int):
            raise TypeError(f'Input_size must be int or tuple of int, '
                            f'but got {type(input_size)}')

        if not isinstance(scales, tuple):
            raise TypeError(f'Scales must be tuple, but got {type(scales)}')

        if num_fixed_crops not in [5, 13]:
            raise ValueError(f'Num_fix_crops must be in {[5, 13]}, '
                             f'but got {num_fixed_crops}')

        self.scales = scales
        self.max_wh_scale_gap = max_wh_scale_gap
        self.random_crop = random_crop
        self.num_fixed_crops = num_fixed_crops
        self.lazy = lazy

    def __call__(self, results):
        """执行 MultiScaleCrop 操作
        Args:
            results (dict): 将修改得到的dict给管道中的下一个变换。

        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']
        base_size = min(img_h, img_w)
        crop_sizes = [int(base_size * s) for s in self.scales]

        candidate_sizes = []
        for i, h in enumerate(crop_sizes):
            for j, w in enumerate(crop_sizes):
                if abs(i - j) <= self.max_wh_scale_gap:
                    candidate_sizes.append([w, h])

        # 随机选择一个裁剪尺寸
        crop_size = random.choice(candidate_sizes)
        for i in range(2):
            if abs(crop_size[i] - self.input_size[i]) < 3:
                crop_size[i] = self.input_size[i]

        crop_w, crop_h = crop_size

        if self.random_crop:
            x_offset = random.randint(0, img_w - crop_w)
            y_offset = random.randint(0, img_h - crop_h)
        else:
            w_step = (img_w - crop_w) // 4
            h_step = (img_h - crop_h) // 4
            candidate_offsets = [
                (0, 0),  # upper left
                (4 * w_step, 0),  # upper right
                (0, 4 * h_step),  # lower left
                (4 * w_step, 4 * h_step),  # lower right
                (2 * w_step, 2 * h_step),  # center
            ]
            if self.num_fixed_crops == 13:
                extra_candidate_offsets = [
                    (0, 2 * h_step),  # center left
                    (4 * w_step, 2 * h_step),  # center right
                    (2 * w_step, 4 * h_step),  # lower center
                    (2 * w_step, 0 * h_step),  # upper center
                    (1 * w_step, 1 * h_step),  # upper left quarter
                    (3 * w_step, 1 * h_step),  # upper right quarter
                    (1 * w_step, 3 * h_step),  # lower left quarter
                    (3 * w_step, 3 * h_step)  # lower right quarter
                ]
                candidate_offsets.extend(extra_candidate_offsets)
            x_offset, y_offset = random.choice(candidate_offsets)

        new_h, new_w = crop_h, crop_w

        # 手动调整偏移量
        crop_bbox = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        # crop_bbox为裁剪框的四个坐标
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)
        # scales参数：操作根据指定的比例列表scales，在不同的尺度上对原始图像进行裁剪。例如，如果scales为(0.8, 1.0, 1.2)，则操作会分别以0.8倍、1.0倍和1.2倍的尺度对图像进行裁剪，生成三种不同尺度的子图像。
        results['scales'] = self.scales
        # 表示裁剪框在原始图像中的归一化位置和大小。
        # 如第一个值（0.43076923）表示裁剪框的左上角 x 坐标在原始图像宽度方向上的位置，归一化到 [0, 1] 范围内。在这个例子中，它占据了原始图像宽度的约 43.08% 处。
        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_x_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'input_size={self.input_size}, scales={self.scales}, '
                    f'max_wh_scale_gap={self.max_wh_scale_gap}, '
                    f'random_crop={self.random_crop}, '
                    f'num_fixed_crops={self.num_fixed_crops}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class Flip:
    """按照一定概率翻转图片
    将给定 imgs 中的元素按特定方向反序排列。 imgs 的形状会保留，但元素会重新排序。


    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "keypoint", "lazy" and
    "flip_direction". Required keys in "lazy" is None, added or modified key
    are "flip" and "flip_direction". 

    翻转扩增应 应放在任何裁剪/重塑增强之后，以确保 crop_quadruple 计算正确。

    Args:
        flip_ratio (float): 翻转图片的概率. Default: 0.5.
        direction (str): 水平或垂直翻转图像.选项包括
            "horizontal" | "vertical". Default: "horizontal".
        flip_label_map (Dict[int, int] | None): 用特定标签转换翻转图像的标签. Default: None.
        left_kp (list[int]): 左侧关键点的索引，用于翻转关键点。
            Default: None.
        right_kp (list[ind]): 右侧关键点的索引，用于翻转关键点。. Default: None.
        lazy (bool): 是否使用懒加载. Default: False.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal',
                 flip_label_map=None,
                 left_kp=None,
                 right_kp=None,
                 lazy=False):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.flip_label_map = flip_label_map
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.lazy = lazy

    def _flip_imgs(self, imgs, modality):
        _ = [mmcv.imflip_(img, self.direction) for img in imgs]
        lt = len(imgs)
        if modality == 'Flow':
            # The 1st frame of each 2 frames is flow-x
            for i in range(0, lt, 2):
                imgs[i] = mmcv.iminvert(imgs[i])
        return imgs

    def _flip_kps(self, kps, kpscores, img_width):
        kp_x = kps[..., 0]
        kp_x[kp_x != 0] = img_width - kp_x[kp_x != 0]
        new_order = list(range(kps.shape[2]))
        if self.left_kp is not None and self.right_kp is not None:
            for left, right in zip(self.left_kp, self.right_kp):
                new_order[left] = right
                new_order[right] = left
        kps = kps[:, :, new_order]
        if kpscores is not None:
            kpscores = kpscores[:, :, new_order]
        return kps, kpscores

    @staticmethod
    def _box_flip(box, img_width):
        """根据图像宽度翻转包围盒.

        Args:
            box (np.ndarray): 翻转边框
            img_width (int): 图像宽度.
        """
        box_ = box.copy()
        box_[..., 0::4] = img_width - box[..., 2::4]
        box_[..., 2::4] = img_width - box[..., 0::4]
        return box_

    def __call__(self, results):
        """实行 Flip 操作.

        Args:
            results (dict): 修改后的 dict 给流水线中的下一个变换
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')
            assert self.direction == 'horizontal', (
                'Only horizontal flips are'
                'supported for human keypoints')

        modality = results['modality']
        if modality == 'Flow':
            assert self.direction == 'horizontal'

        flip = np.random.rand() < self.flip_ratio

        results['flip'] = flip
        results['flip_direction'] = self.direction
        img_width = results['img_shape'][1]

        if self.flip_label_map is not None and flip:
            results['label'] = self.flip_label_map.get(results['label'],
                                                       results['label'])

        if not self.lazy:
            if flip:
                if 'imgs' in results:
                    results['imgs'] = self._flip_imgs(results['imgs'],
                                                      modality)
                if 'keypoint' in results:
                    kp = results['keypoint']
                    kpscore = results.get('keypoint_score', None)
                    kp, kpscore = self._flip_kps(kp, kpscore, img_width)
                    results['keypoint'] = kp
                    if 'keypoint_score' in results:
                        results['keypoint_score'] = kpscore
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Use one Flip please')
            lazyop['flip'] = flip
            lazyop['flip_direction'] = self.direction

        if 'gt_bboxes' in results and flip:
            assert not self.lazy and self.direction == 'horizontal'
            width = results['img_shape'][1]
            results['gt_bboxes'] = self._box_flip(results['gt_bboxes'], width)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_flip(results['proposals'],
                                                      width)
        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'flip_label_map={self.flip_label_map}, lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class ColorJitter:
    def __init__(self, p=0.8, p_gray=0.2, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        self.p = p
        self.p_gray = p_gray
        self.worker = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, results):
        imgs = results['imgs']
        v = random.random()
        if v < self.p:
            imgs = [np.asarray(self.worker(Image.fromarray(img)))
                    for img in imgs]

        results['imgs'] = imgs
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}')
        return repr_str


@PIPELINES.register_module()
class GrayScale:
    # 随机灰度，有些灰有些不灰
    def __init__(self, p=0.2):
        self.p = p
        self.worker_gray = torchvision.transforms.Grayscale(
            num_output_channels=3)

    def __call__(self, results):
        imgs = results['imgs']
        v = random.random()
        if v < self.p:
            imgs = [np.asarray(self.worker_gray(Image.fromarray(img)))
                    for img in imgs]

        results['imgs'] = imgs
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}')
        return repr_str


@PIPELINES.register_module()
class Normalize:
    """用给定的平均值和 std 值对图像进行归一化处理.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required

    Args:
        mean (Sequence[float]): 不同channels平均值。
        std (Sequence[float]): 不同channels的标准差.
        to_bgr (bool): 是否将通道从 RGB 转换为 BGR.
            Default: False.
        adjust_magnitude (bool): I当模式为 "Flow "时，指示是否在 "scale_factor "上调整流量大小. Default: False.
    """

    def __init__(self, mean, std, to_bgr=False, adjust_magnitude=False):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}'
            )

        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_bgr = to_bgr
        self.adjust_magnitude = adjust_magnitude

    def __call__(self, results):
        modality = results['modality']

        if modality == 'RGB':
            n = len(results['imgs'])
            h, w, c = results['imgs'][0].shape
            imgs = np.empty((n, h, w, c), dtype=np.float32)
            for i, img in enumerate(results['imgs']):
                imgs[i] = img

            for img in imgs:
                mmcv.imnormalize_(img, self.mean, self.std, self.to_bgr)
            results['imgs'] = imgs
            results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_bgr=self.to_bgr)

            return results
        if modality == 'Flow':
            num_imgs = len(results['imgs'])
            assert num_imgs % 2 == 0
            assert self.mean.shape[0] == 2
            assert self.std.shape[0] == 2
            n = num_imgs // 2
            h, w = results['imgs'][0].shape
            x_flow = np.empty((n, h, w), dtype=np.float32)
            y_flow = np.empty((n, h, w), dtype=np.float32)
            for i in range(n):
                x_flow[i] = results['imgs'][2 * i]
                y_flow[i] = results['imgs'][2 * i + 1]
            x_flow = (x_flow - self.mean[0]) / self.std[0]
            y_flow = (y_flow - self.mean[1]) / self.std[1]
            if self.adjust_magnitude:
                x_flow = x_flow * results['scale_factor'][0]
                y_flow = y_flow * results['scale_factor'][1]
            imgs = np.stack([x_flow, y_flow], axis=-1)
            results['imgs'] = imgs
            args = dict(
                mean=self.mean,
                std=self.std,
                to_bgr=self.to_bgr,
                adjust_magnitude=self.adjust_magnitude)
            results['img_norm_cfg'] = args
            return results
        raise NotImplementedError

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mean={self.mean}, '
                    f'std={self.std}, '
                    f'to_bgr={self.to_bgr}, '
                    f'adjust_magnitude={self.adjust_magnitude})')
        return repr_str


@PIPELINES.register_module()
class FormatShape:
    """将最后的图像转化为神经网络的输入模式

    Required keys are "imgs", "num_clips" and "clip_len", added or modified
    keys are "imgs" and "input_shape".

    Args:
        input_format (str): 定义最终的输入模式
        collapse (bool): To collpase input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Default: False.
    """

    def __init__(self, input_format, collapse=False):
        self.input_format = input_format
        self.collapse = collapse
        if self.input_format not in ['NCTHW', 'NCHW', 'NCHW_Flow', 'NPTCHW']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def __call__(self, results):
        """Performs the FormatShape formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['imgs'], np.ndarray):
            results['imgs'] = np.array(results['imgs'])
        imgs = results['imgs']
        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * L
        if self.collapse:
            assert results['num_clips'] == 1

        if self.input_format == 'NCTHW':
            num_clips = results['num_clips']
            clip_len = results['clip_len']

            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x L x H x W x C
            imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
            # N_crops x N_clips x C x L x H x W
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
            # M' x C x L x H x W
            # M' = N_crops x N_clips
        elif self.input_format == 'NCHW':
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            # M x C x H x W
        elif self.input_format == 'NCHW_Flow':
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x L x H x W x C
            imgs = np.transpose(imgs, (0, 1, 2, 5, 3, 4))
            # N_crops x N_clips x L x C x H x W
            imgs = imgs.reshape((-1, imgs.shape[2] * imgs.shape[3]) +
                                imgs.shape[4:])
            # M' x C' x H x W
            # M' = N_crops x N_clips
            # C' = L x C
        elif self.input_format == 'NPTCHW':
            num_proposals = results['num_proposals']
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((num_proposals, num_clips * clip_len) +
                                imgs.shape[1:])
            # P x M x H x W x C
            # M = N_clips x L
            imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
            # P x M x C x H x W
        if self.collapse:
            assert imgs.shape[0] == 1
            imgs = imgs.squeeze(0)

        results['imgs'] = imgs
        results['input_shape'] = imgs.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str


@PIPELINES.register_module()
class Tokenlization():
    """使用bart-base 进行tokenlizer"""

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, results):

        lm_tokenized_article = {
            "input_ids": torch.zeros(1, self.config.TOKEN.SENT_TOKEN_LEN, dtype=torch.int32),
            "attention_mask": torch.zeros(1, self.config.TOKEN.SENT_TOKEN_LEN, dtype=torch.int32),
        }

        sent_len = 0
        cls_position = []
        article = []

        # 进行分句子
        import re
        # 定义包含中文文本的字符串
        chinese_text = results["subtitle"]
        # 使用正则表达式查找句号或句点作为分句标志
        sentences = re.split('(。|！|\!|\.|？|\?)', chinese_text)         # 保留分割符
        # 去除空句子（如果有的话）
        sentences = [sentence.strip()
                     for sentence in sentences if sentence.strip()]

        for i in range(len(sentences)):
            tokenized_sent = self.tokenizer(
                sentences,
                add_special_tokens=self.config.TOKEN.ADD_SPECIAL_TOKENS,
                return_tensors=self.config.TOKEN.RETURN_TENSOR,
            )
            # add <s> and <\s>
            if sent_len + tokenized_sent["input_ids"].size()[1] <= self.config.TOKEN.SENT_TOKEN_LEN:
                cls_position.append(sent_len)
                article.append(sentences[i])
                sent_len += tokenized_sent["input_ids"].size()[1]

                lm_tokenized_article["input_ids"][:, cls_position[-1]:sent_len] = tokenized_sent["input_ids"]
                lm_tokenized_article["attention_mask"][:, cls_position[-1]: sent_len] = tokenized_sent[
                    "attention_mask"]
            else:
                break
        results['input_ids'] = tokenized_sent['input_ids']
        results['attention_mask'] = tokenized_sent['attention_mask']
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str


@PIPELINES.register_module()
class Collect:
    """收集相关任务的数据.其实就是将前面步骤中的result中需要使用的属性转化为data

    This keeps the items in ``keys`` as it is, and collect items in
    ``meta_keys`` into a meta item called ``meta_name``.This is usually
    the last stage of the data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str]): Required keys to be collected.
        meta_name (str): The name of the key that contains meta infomation.
            This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str]): Keys that are collected under meta_name.
            The contents of the ``meta_name`` dictionary depends on
            ``meta_keys``.
            By default this includes:

            - "filename": 文件路径
            - "videosum": 视频摘要
            - "original_shape": 原始视频维度
                (h, w, c)
                (h, w, c).  Note that images may be zero padded on the
                bottom/right, if the batch tensor is larger than this shape.
            - "pad_shape": image shape after padding
            - "flip_direction": a str in ("horiziontal", "vertival") to
                indicate if the image is fliped horizontally or vertically.
            - "img_norm_cfg": a dict of normalization information:
                - mean - per channel mean subtraction
                - std - per channel std divisor
                - to_rgb - bool indicating if bgr was converted to rgb
        nested (bool): If set as True, will apply data[x] = [data[x]] to all
            items in data. The arg is added for compatibility. Default: False.
    """

    # def __init__(self,
    #              keys,
    #              meta_keys=('filepath', 'original_shape', 'img_shape',
    #                         'pad_shape', 'flip_direction', 'img_norm_cfg'),
    #              meta_name='img_metas',
    #              nested=False):
    def __init__(self,
                 keys,
                 meta_keys=('imgs', 'original_shape', 'img_shape'),
                 meta_name='img_metas',
                 nested=False):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name
        self.nested = nested

    def __call__(self, results):
        """Performs the Collect formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        data = {}
        for key in self.keys:
            data[key] = results[key]

        if len(self.meta_keys) != 0:
            meta = {}
            for key in self.meta_keys:
                meta[key] = results[key]
            data[self.meta_name] = DC(meta, cpu_only=True)
        if self.nested:
            for k in data:
                data[k] = [data[k]]

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys}, '
                f'nested={self.nested})')


def to_tensor(data):
    """将数据转化为`torch.Tensor`.

    转化类型: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    if isinstance(data, int):
        return torch.LongTensor([data])
    if isinstance(data, float):
        return torch.FloatTensor([data])
    raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PIPELINES.register_module()
class ToTensor:
    """在pipline将result中的 dict某些值转换为 `torch.Tensor` 

    Args:
        keys (Sequence[str]): Required keys to be converted.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Performs the ToTensor formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'


@PIPELINES.register_module()
class RandAugment:
    def __init__(self, auto_augment, input_size=224, interpolation='bicubic', level='video'):
        if isinstance(input_size, tuple):
            img_size = input_size[-2:]
        else:
            img_size = input_size

        if auto_augment:
            assert isinstance(auto_augment, str)
            if isinstance(img_size, tuple):
                img_size_min = min(img_size)
            else:
                img_size_min = img_size
            aa_params = {"translate_const": int(img_size_min * 0.45)}
            if interpolation and interpolation != "random":
                aa_params["interpolation"] = _pil_interp(interpolation)
        self.auto_augment = auto_augment
        self.aa_params = aa_params
        self.level = level

    def do_ops(self, ops, buf):
        for op in ops:
            buf = op(buf)
        return buf

    def get_ops(self, ra_ops, num_layers, choice_weights):
        return np.random.choice(
            ra_ops,
            num_layers,
            replace=choice_weights is None,
            p=choice_weights,
        )

    def __call__(self, results):
        if self.auto_augment.startswith("rand"):
            ra_ops, num_layers, choice_weights = rand_augment_transform(
                self.auto_augment, self.aa_params)

        assert results['modality'] == 'RGB', 'Imgaug only support RGB images.'
        in_type = results['imgs'][0].dtype.type

        if self.level == 'video':
            ops = self.get_ops(ra_ops, num_layers, choice_weights)
            buffer = [
                transforms.ToPILImage()(frame) for frame in results['imgs']
            ]
            results['imgs'] = [
                np.asarray(self.do_ops(ops, buf)) for buf in buffer
            ]

        elif self.level == 'image':
            buffer = [
                transforms.ToPILImage()(frame) for frame in results['imgs']
            ]
            results['imgs'] = []
            for buf in buffer:
                ops = self.get_ops(ra_ops, num_layers, choice_weights)
                buf = self.do_ops(ops, buf)
                results['imgs'].append(np.asarray(buf))
        else:
            assert False, 'Unknown RandAugment config section'

        img_h, img_w, _ = results['imgs'][0].shape
        out_type = results['imgs'][0].dtype.type
        assert in_type == out_type, \
            ('Imgaug input dtype and output dtype are not the same. ',
             f'Convert from {in_type} to {out_type}')

        results['img_shape'] = (img_h, img_w)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.aug})'
        return repr_str
