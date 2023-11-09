import copy
from distutils import config
from macpath import join
# from clip.model import CLIP, LayerNorm, Transformer
from typing import Tuple, Union
import torch
from torch import nn
from torch import optim
import numpy as np

from models.clip.model import CLIP, LayerNorm, Transformer
from .mit import MultiframeIntegrationTransformer
from .cct import CrossFrameCommunicationTransformer
# from .btm import UniMS, VisualGuideBartModel
# from .metrics import ROUGEScore
# from torchmetrics import RetrievalPrecision
import sys
import warnings
import hashlib
import os
import urllib
from datasets import load_metric

from tqdm import tqdm
from transformers.utils import logging
logger = logging.get_logger(__name__)
sys.path.append("../")


class XSUM(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # video
                 T=8,
                 droppath=0.,
                 mit_layers=1,
                 # prompt
                 prompts_alpha=1e-4,
                 prompts_layers=1,
                 # other
                 use_cache=True,
                 use_checkpoint=False,
                 ):
        super().__init__(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )

        self.use_cache = use_cache
        self.mit = MultiframeIntegrationTransformer(
            T=T, embed_dim=embed_dim, layers=mit_layers,)
        self.vision_width = vision_width
        self.embed_dim = embed_dim
        # self.prompts_visual_ln = LayerNorm(vision_width)
        # self.prompts_visual_proj = nn.Parameter(
        #     torch.randn(vision_width, embed_dim))

        dpr = [x.item() for x in torch.linspace(
            0, droppath, vision_layers)] if droppath > 0. else None

        vision_heads = vision_width // 64

        self.visual = CrossFrameCommunicationTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            droppath=dpr,
            T=T,
            use_checkpoint=use_checkpoint,
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)
        eos_indx = text.argmax(dim=-1)
        K, N1, C = x.shape

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(K, -1)
        return x

    def encode_video(self, image):
        b, t, c, h, w = image.size()
        image = image.reshape(-1, c, h, w)
        cls_features = self.encode_image(image) # 15,512
        # cls_features, img_features = self.encode_image(image)
        # img_features = self.prompts_visual_ln(img_features)
        # img_features = img_features @ self.prompts_visual_proj

        cls_features = cls_features.view(b, t, -1) # 2,8,512
        # img_features = img_features.contiguous().view(
        #     b, t, -1, cls_features.shape[-1])

        video_features = self.mit(cls_features)    #(2,8,512)

        # return video_features, img_features
        return video_features

    def forward(self, image):
        b = image.shape[0]
        video_features = self.encode_video(image)    # 2,8,512
        # video_features, img_features = self.encode_video(image)
        # img_features = img_features.mean(dim=1, keepdim=False)
        video_features = video_features / \
            video_features.norm(dim=-1, keepdim=True)  # 2,8,512
        return video_features


def build_model(state_dict: dict, T=8, droppath=0., use_checkpoint=False, logger=logger, prompts_alpha=1e-1, prompts_layers=2, use_cache=True, mit_layers=4,):
    """
    Args:
        state_dict: 模型架构设置
    """
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith(
            "visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1 .weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(
            f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)

        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + \
            1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(
        k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = XSUM(
        embed_dim=embed_dim,
        image_resolution=image_resolution, vision_layers=vision_layers, vision_width=vision_width, vision_patch_size=vision_patch_size,
        context_length=context_length, vocab_size=vocab_size, transformer_width=transformer_width, transformer_heads=transformer_heads, transformer_layers=transformer_layers,
        T=T, droppath=droppath, mit_layers=mit_layers,
        prompts_alpha=prompts_alpha, prompts_layers=prompts_layers,
        use_checkpoint=use_checkpoint, use_cache=use_cache,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f"load pretrained CLIP: {msg}")

    return model.eval()


_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def load(model_path, name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         jit=True, T=8, droppath=0., use_checkpoint=False, logger=logger, use_cache=True, prompts_alpha=1e-1, prompts_layers=2, mit_layers=1,
         ):
    if model_path is None:
        model_path = _download(_MODELS[name])
    try:
        # 载入jit模型
        model = torch.jit.load(
            model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(
                f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(state_dict or model.state_dict(), T=T, droppath=droppath,
                        use_checkpoint=use_checkpoint, logger=logger,
                        prompts_alpha=prompts_alpha,
                        prompts_layers=prompts_layers,
                        use_cache=use_cache,
                        mit_layers=mit_layers,
                        )
    if str(device) == "cpu":
        model.float()
    return model, model.state_dict()


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(
            f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target
