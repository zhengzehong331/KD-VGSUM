from transformers.utils import logging
import math
import copy
from typing import Optional, List
import torch
from torch import nn
from models import xsum
from models.clip.model import LayerNorm
from models.mit import MultiframeIntegrationTransformer
# from models.modeling_bart import BartLearnedPositionalEmbedding

logger = logging.get_logger(__name__)


class ImageTransformerEncoder(nn.Module):
    """"
    input:
        d_model:input dimension
        num_layers: num layer
        num_heads: Multiple Attention Mechanisms head number
        dim_feedforward:feed-forward neural network input dimension
    """

    def __init__(self, d_model, num_layers, num_heads, dim_feedforward=2048, backbone=None):
        super(ImageTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.multimodal_model, _ = xsum.load(None, 'ViT-B/32',
                                             device="cpu", jit=False,
                                             T=8,
                                             droppath=0,
                                             use_checkpoint=False,
                                             use_cache=True,
                                             logger=logger
                                             )

        self.prompts_visual_ln = LayerNorm(self.multimodal_model.vision_width)
        self.prompts_visual_proj = nn.Parameter(
            torch.randn(self.multimodal_model.vision_width, self.multimodal_model.embed_dim))
        self.mit = self.multimodal_model.mit

        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward)
        # self.encoder = _TransformerEncoder(
        #     encoder_layer, num_layers=num_layers)

        # 加载CLIP模型,只保留视觉部分

        # # Multimodal Model: image-text match model
        # if backbone in ['LinProj', 'CLIP-RN50', 'CLIP-ViT']:
        #     # multimodal_model, self.image_preprocess = clip.load(
        #     #     self.args["clip_path"], device=self.device, jit=False)
        #     multimodal_model, self.image_preprocess = clip.load(
        #         'ViT-B/32', device='cpu', jit=False)
        #     self.visual = copy.deepcopy(multimodal_model.visual)
        #     del multimodal_model

        # # 这里选择投影方式
        # if backbone == 'LinProj':
        #     # 只对图像进行投影嵌入，删除其他没用的参数
        #     del self.visual.ln_pre
        #     del self.visual.transformer
        #     del self.visual.ln_post
        #     del self.visual.proj

        # elif backbone.endswith('RN50'):
        #     del self.visual.attnpool
        #     for param in self.visual.parameters():
        #         param.requires_grad = False

        #     # Image Post Linear Proj
        #     self.image_post_linproj = torch.nn.Sequential(
        #         torch.nn.LayerNorm(2048),
        #         torch.nn.Linear(2048, 768),
        #     )

        #     # Image [CLS] Embedding
        #     self.visual_class_embedding = torch.nn.Parameter(
        #         torch.zeros(self.generator.config.d_model, device=self.device))
        #     self.visual_class_embedding.data.normal_(mean=0.0, std=0.02)

        # elif backbone.endswith('ViT'):
        #     del self.visual.ln_post
        #     del self.visual.proj
        #     for param in self.visual.parameters():
        #         param.requires_grad = False

        #     # Image Post Linear Proj
        #     self.image_post_linproj = torch.nn.Sequential(
        #         torch.nn.LayerNorm(self.generator.config.hidden_size),
        #         torch.nn.Linear(self.generator.config.hidden_size,
        #                         self.generator.config.hidden_size),
        #         # torch.nn.Sigmoid(),
        #     )

        # # Image Position Embedding 对图像位置进行嵌入
        # # self.visual_embed_positions = BartLearnedPositionalEmbedding(
        # #     self.generator.config.max_position_embeddings,
        # #     # 这里的d_model是transformer模型中的超参数，表示模型中嵌入和层的layer的大小
        # #     self.generator.config.d_model,
        # # )
        # # self.visual_embed_positions.weight.data.normal_(mean=0.0, std=0.02)

        # # if self.args['use_image_score']:
        # #     # Image Extractor Head  图像特征抽取
        # #     self.image_classifier_head = torch.nn.Sequential(
        # #         torch.nn.LayerNorm(self.generator.config.hidden_size),
        # #         torch.nn.Linear(self.generator.config.hidden_size, 1),
        # #     )
        # #     for n, p in self.image_classifier_head.named_parameters():
        # #         if 'weight' in n:
        # #             p.data.normal_(mean=0.0, std=0.02)
        # #         elif 'bias' in n:
        # #             p.data.zero_()
        # # else:
        # #     self.image_preprocess = None

        # self.pos_encoder = PositionalEncoding(d_model, dropout=0.1)
        # # self.clip, state_dict = xsum.load(None, 'ViT-B/32',
        # #                                         device="cpu", jit=False,
        # #                                         T=8,
        # #                                         droppath=0,
        # #                                         use_checkpoint=False,
        # #                                         use_cache=True,
        # #                                   )

    def forward(self, image: torch.Tensor, lens: Optional[List[int]] = None):
        b = image.shape[0]
        video_features, img_features = self.encode_video(image)
        img_features = img_features.mean(dim=1, keepdim=False)
        video_features = video_features / \
            video_features.norm(dim=-1, keepdim=True)
        return video_features

        # if lens is not None:
        #     max_len = max(lens)

        #     mask = [([False] * l + [True] * (max_len - l)) for l in lens]
        #     mask = torch.tensor(mask).to(device=inputs.device)
        # else:
        #     mask = None

        # # input = batch
        # inputs = inputs.permute(1, 0, 2)

        # inputs = inputs * math.sqrt(self.d_model)
        # inputs = self.pos_encoder(inputs)

        # # (seq_len, bs, dim)
        # outputs = self.clip(image=inputs)

        # outputs = self.encoder(src=inputs, src_key_padding_mask=mask)
        # # outputs = self.clip(image=inputs)

        # return [o.permute(1, 0, 2) for o in outputs]

        # [batch_size, channel, W, H]
        # VICR

    def encode_image(self, image):
        return self.multimodal_model.visual(image)

    def encode_video(self, image):
        # b:batch-size t:time-step c:channels h:height w:width
        b, t, c, h, w = image.size()

        # 合并前两个维度
        image = image.reshape(-1, c, h, w)

        # cls这里表示类别特征
        cls_features, img_features = self.encode_image(
            image)  # (16,512)  (16,49,769)
        img_features = self.prompts_visual_ln(img_features)
        img_features = img_features @ self.prompts_visual_proj  # (16,49,512)

        cls_features = cls_features.view(b, t, -1)  # (2,8,512)
        img_features = img_features.view(
            b, t, -1, cls_features.shape[-1])  # (2,8,49,512)
        # 这里shape从[2,8,512]->[16,512]
        video_features = self.multimodal_model.mit(cls_features)  # 2,512

        return video_features, img_features


def padTensor(t: torch.Tensor, targetLen: int) -> torch.Tensor:
    oriLen, dim = t.size()
    return torch.cat((t, torch.zeros(targetLen - oriLen, dim).to(t.device)), dim=0)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class _TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = [src]

        for mod in self.layers:
            output = mod(outputs[-1], src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)
            outputs.append(output)

        if self.norm is not None:
            outputs[-1] = self.norm(outputs[-1])

        return outputs[1:]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
