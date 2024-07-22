# Merge image encoder and fuse module to create an ID Encoder
# send multiple ID images, we can directly obtain the updated text encoder containing a stacked ID embedding

import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
from transformers.models.clip.configuration_clip import CLIPVisionConfig

from .resampler import FacePerceiverResampler

VISION_CONFIG_DICT = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768
}


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class QFormerPerceiver(nn.Module):
    def __init__(self, id_embeddings_dim, cross_attention_dim, num_tokens, embedding_dim=1024, use_residual=True, ratio=4):
        super().__init__()

        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.use_residual = use_residual
        print(cross_attention_dim*num_tokens)
        self.token_proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim*ratio),
            nn.GELU(),
            nn.Linear(id_embeddings_dim*ratio, cross_attention_dim*num_tokens),
        )
        self.token_norm = nn.LayerNorm(cross_attention_dim)
        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=128,
            heads=cross_attention_dim // 128,
            embedding_dim=embedding_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )

    def forward(self, x, last_hidden_state):
        x = self.token_proj(x)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.token_norm(x) # cls token
        out = self.perceiver_resampler(x, last_hidden_state) # retrieve from patch tokens
        if self.use_residual: # TODO: if use_residual is not true
            out = x + 1.0 * out 
        return out


class FuseModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, prompt_embeds, id_embeds):
        stacked_id_embeds = torch.cat([prompt_embeds, id_embeds], dim=-1)
        stacked_id_embeds = self.mlp1(stacked_id_embeds) + prompt_embeds
        stacked_id_embeds = self.mlp2(stacked_id_embeds)
        stacked_id_embeds = self.layer_norm(stacked_id_embeds)
        return stacked_id_embeds

    def forward(
        self,
        prompt_embeds,
        id_embeds,
        class_tokens_mask,
    ) -> torch.Tensor:
        # id_embeds shape: [b, max_num_inputs, 1, 2048]
        id_embeds = id_embeds.to(prompt_embeds.dtype)
        num_inputs = class_tokens_mask.sum().unsqueeze(0) # TODO: check for training case
        batch_size, max_num_inputs = id_embeds.shape[:2]
        # seq_length: 77
        seq_length = prompt_embeds.shape[1]
        # flat_id_embeds shape: [b*max_num_inputs, 1, 2048]
        flat_id_embeds = id_embeds.view(
            -1, id_embeds.shape[-2], id_embeds.shape[-1]
        )
        # valid_id_mask [b*max_num_inputs]
        valid_id_mask = (
            torch.arange(max_num_inputs, device=flat_id_embeds.device)[None, :]
            < num_inputs[:, None]
        )
        valid_id_embeds = flat_id_embeds[valid_id_mask.flatten()]

        prompt_embeds = prompt_embeds.view(-1, prompt_embeds.shape[-1])
        class_tokens_mask = class_tokens_mask.view(-1)
        valid_id_embeds = valid_id_embeds.view(-1, valid_id_embeds.shape[-1])
        # slice out the image token embeddings
        image_token_embeds = prompt_embeds[class_tokens_mask]
        stacked_id_embeds = self.fuse_fn(image_token_embeds, valid_id_embeds)
        assert class_tokens_mask.sum() == stacked_id_embeds.shape[0], f"{class_tokens_mask.sum()} != {stacked_id_embeds.shape[0]}"
        prompt_embeds.masked_scatter_(class_tokens_mask[:, None], stacked_id_embeds.to(prompt_embeds.dtype))
        updated_prompt_embeds = prompt_embeds.view(batch_size, seq_length, -1)
        return updated_prompt_embeds


class PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken(CLIPVisionModelWithProjection):
    def __init__(self, id_embeddings_dim=512):
        super().__init__(CLIPVisionConfig(**VISION_CONFIG_DICT))
        self.fuse_module = FuseModule(2048)
        self.visual_projection_2 = nn.Linear(1024, 1280, bias=False)

        cross_attention_dim = 2048
        # projection
        self.num_tokens = 2
        self.cross_attention_dim = cross_attention_dim
        self.qformer_perceiver = QFormerPerceiver(
                                    id_embeddings_dim, 
                                    cross_attention_dim, 
                                    self.num_tokens,
                                )

    def forward(self, id_pixel_values, prompt_embeds, class_tokens_mask, id_embeds):
        b, num_inputs, c, h, w = id_pixel_values.shape
        id_pixel_values = id_pixel_values.view(b * num_inputs, c, h, w)

        last_hidden_state = self.vision_model(id_pixel_values)[0]
        id_embeds = id_embeds.view(b * num_inputs, -1)

        id_embeds = self.qformer_perceiver(id_embeds, last_hidden_state)
        id_embeds = id_embeds.view(b, num_inputs, self.num_tokens, -1)
        updated_prompt_embeds = self.fuse_module(prompt_embeds, id_embeds, class_tokens_mask)

        return updated_prompt_embeds

if __name__ == "__main__":
    PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken()
