"""JAX implementation of the inspected Diffusers UNet2DModel checkpoint."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp


def conv2d(x, weight, bias=None, stride=1, padding=1):
    result = jax.lax.conv_general_dilated(
        x,
        jnp.asarray(weight, dtype=jnp.float32),
        window_strides=(stride, stride),
        padding=((padding, padding), (padding, padding)) if isinstance(padding, int) else padding,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return result if bias is None else result + jnp.asarray(bias, dtype=jnp.float32)[None, :, None, None]


def group_norm(x, weight, bias, groups=32, eps=1e-6):
    batch, channels, height, width = x.shape
    if channels % groups:
        raise ValueError(f"channels={channels} is not divisible by groups={groups}")
    grouped = x.reshape(batch, groups, channels // groups, height, width)
    mean = grouped.mean(axis=(2, 3, 4), keepdims=True)
    variance = jnp.mean(jnp.square(grouped - mean), axis=(2, 3, 4), keepdims=True)
    normalized = ((grouped - mean) * jax.lax.rsqrt(variance + eps)).reshape(x.shape)
    return normalized * jnp.asarray(weight, dtype=jnp.float32)[None, :, None, None] + jnp.asarray(bias, dtype=jnp.float32)[None, :, None, None]


def timestep_embedding(timesteps, dimension=128, freq_shift=1, flip_sin_to_cos=False):
    timesteps = jnp.asarray(timesteps, dtype=jnp.float32).reshape(-1)
    half = dimension // 2
    exponent = -jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / (half - freq_shift)
    frequencies = jnp.exp(exponent)
    angles = timesteps[:, None] * frequencies[None, :]
    embedding = jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=1)
    if flip_sin_to_cos:
        embedding = jnp.concatenate((embedding[:, half:], embedding[:, :half]), axis=1)
    return embedding


class CifarUNet2D:
    def __init__(self, parameters: Mapping[str, object], config: Mapping[str, object]):
        self.parameters = parameters
        self.config = config
        self.used_names: set[str] = set()
        self.norm_groups = int(config["norm_num_groups"])
        self.norm_eps = float(config["norm_eps"])

    def p(self, name: str):
        self.used_names.add(name)
        return self.parameters[name]

    def resnet(self, x, temb, prefix: str):
        input_x = x
        x = group_norm(x, self.p(f"{prefix}.norm1.weight"), self.p(f"{prefix}.norm1.bias"), self.norm_groups, self.norm_eps)
        x = jax.nn.silu(x)
        x = conv2d(x, self.p(f"{prefix}.conv1.weight"), self.p(f"{prefix}.conv1.bias"))
        temb_value = jax.nn.silu(temb)
        temb_value = temb_value @ jnp.asarray(self.p(f"{prefix}.time_emb_proj.weight"), dtype=jnp.float32).T + jnp.asarray(self.p(f"{prefix}.time_emb_proj.bias"), dtype=jnp.float32)
        x = x + temb_value[:, :, None, None]
        x = group_norm(x, self.p(f"{prefix}.norm2.weight"), self.p(f"{prefix}.norm2.bias"), self.norm_groups, self.norm_eps)
        x = jax.nn.silu(x)
        x = conv2d(x, self.p(f"{prefix}.conv2.weight"), self.p(f"{prefix}.conv2.bias"))
        shortcut_name = f"{prefix}.conv_shortcut.weight"
        if shortcut_name in self.parameters:
            shortcut = conv2d(input_x, self.p(shortcut_name), self.p(f"{prefix}.conv_shortcut.bias"), padding=0)
        else:
            shortcut = input_x
        return (shortcut + x) / jnp.sqrt(jnp.asarray(1.0, dtype=jnp.float32))

    def attention(self, x, prefix: str):
        residual = x
        x = group_norm(x, self.p(f"{prefix}.group_norm.weight"), self.p(f"{prefix}.group_norm.bias"), self.norm_groups, self.norm_eps)
        batch, channels, height, width = x.shape
        tokens = jnp.transpose(x, (0, 2, 3, 1)).reshape(batch, height * width, channels)
        query = tokens @ jnp.asarray(self.p(f"{prefix}.query.weight"), dtype=jnp.float32).T + jnp.asarray(self.p(f"{prefix}.query.bias"), dtype=jnp.float32)
        key = tokens @ jnp.asarray(self.p(f"{prefix}.key.weight"), dtype=jnp.float32).T + jnp.asarray(self.p(f"{prefix}.key.bias"), dtype=jnp.float32)
        value = tokens @ jnp.asarray(self.p(f"{prefix}.value.weight"), dtype=jnp.float32).T + jnp.asarray(self.p(f"{prefix}.value.bias"), dtype=jnp.float32)
        heads = 32
        head_dim = channels // heads
        query = query.reshape(batch, -1, heads, head_dim).transpose(0, 2, 1, 3)
        key = key.reshape(batch, -1, heads, head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(batch, -1, heads, head_dim).transpose(0, 2, 1, 3)
        scores = jnp.einsum("bhqd,bhkd->bhqk", query, key) / jnp.sqrt(jnp.asarray(head_dim, dtype=jnp.float32))
        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum("bhqk,bhkd->bhqd", weights, value).transpose(0, 2, 1, 3).reshape(batch, height * width, channels)
        attended = attended @ jnp.asarray(self.p(f"{prefix}.proj_attn.weight"), dtype=jnp.float32).T + jnp.asarray(self.p(f"{prefix}.proj_attn.bias"), dtype=jnp.float32)
        attended = attended.reshape(batch, height, width, channels).transpose(0, 3, 1, 2)
        return residual + attended

    def upsample(self, x, prefix: str, target_shape):
        x = jax.image.resize(
            x,
            (x.shape[0], x.shape[1], target_shape[0], target_shape[1]),
            method="nearest",
        )
        return conv2d(x, self.p(f"{prefix}.conv.weight"), self.p(f"{prefix}.conv.bias"))

    def __call__(self, images, timesteps):
        x = jnp.asarray(images, dtype=jnp.float32)
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected NCHW input (B,3,H,W), got {x.shape}")
        t = jnp.asarray(timesteps, dtype=jnp.float32).reshape(-1)
        if t.shape[0] == 1:
            t = jnp.broadcast_to(t, (x.shape[0],))
        # The range check is kept for ordinary eager calls. During compiled
        # sampling, t is a tracer and its range is guaranteed by the caller.
        if t.shape[0] != x.shape[0]:
            raise ValueError("timesteps must be mathematical levels 1 through 1000")
        if not isinstance(t, jax.core.Tracer) and (bool(jnp.any(t < 1)) or bool(jnp.any(t > 1000))):
            raise ValueError("timesteps must be mathematical levels 1 through 1000")
        x = conv2d(x, self.p("conv_in.weight"), self.p("conv_in.bias"))
        # The checkpoint was trained with zero-based Diffusers indices; the course uses t=1,...,1000.
        temb = timestep_embedding(t - 1.0, 128, int(self.config["freq_shift"]), bool(self.config["flip_sin_to_cos"]))
        temb = temb @ jnp.asarray(self.p("time_embedding.linear_1.weight"), dtype=jnp.float32).T + jnp.asarray(self.p("time_embedding.linear_1.bias"), dtype=jnp.float32)
        temb = jax.nn.silu(temb)
        temb = temb @ jnp.asarray(self.p("time_embedding.linear_2.weight"), dtype=jnp.float32).T + jnp.asarray(self.p("time_embedding.linear_2.bias"), dtype=jnp.float32)
        skips = [x]
        down_types = self.config["down_block_types"]
        for block_index, block_type in enumerate(down_types):
            for resnet_index in range(int(self.config["layers_per_block"])):
                prefix = f"down_blocks.{block_index}.resnets.{resnet_index}"
                x = self.resnet(x, temb, prefix)
                if block_type == "AttnDownBlock2D":
                    x = self.attention(x, f"down_blocks.{block_index}.attentions.{resnet_index}")
                skips.append(x)
            if block_index < len(down_types) - 1:
                x = conv2d(x, self.p(f"down_blocks.{block_index}.downsamplers.0.conv.weight"), self.p(f"down_blocks.{block_index}.downsamplers.0.conv.bias"), stride=2, padding=int(self.config["downsample_padding"]))
                skips.append(x)
        x = self.resnet(x, temb, "mid_block.resnets.0")
        x = self.attention(x, "mid_block.attentions.0")
        x = self.resnet(x, temb, "mid_block.resnets.1")
        up_types = self.config["up_block_types"]
        for block_index, block_type in enumerate(up_types):
            for resnet_index in range(3):
                skip = skips.pop()
                x = jnp.concatenate((x, skip), axis=1)
                prefix = f"up_blocks.{block_index}.resnets.{resnet_index}"
                x = self.resnet(x, temb, prefix)
                if block_type == "AttnUpBlock2D":
                    x = self.attention(x, f"up_blocks.{block_index}.attentions.{resnet_index}")
            if block_index < len(up_types) - 1:
                x = self.upsample(
                    x,
                    f"up_blocks.{block_index}.upsamplers.0",
                    skips[-1].shape[2:],
                )
        x = group_norm(x, self.p("conv_norm_out.weight"), self.p("conv_norm_out.bias"), self.norm_groups, self.norm_eps)
        x = jax.nn.silu(x)
        return conv2d(x, self.p("conv_out.weight"), self.p("conv_out.bias"))
