# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed._functional_collectives as ft_c
from packaging.version import parse
from torch.distributed.tensor.experimental._attention import _cp_options

import sglang.multimodal_gen.envs as envs
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_world_size,
)

# Notes on experimental comm paths:
# - envs.SGLANG_USP_ASYNC_ALLTOALL overlaps Q/K/V by fusing them into one collective.
# - envs.SGLANG_USP_FP8_COMM additionally compresses all_to_all payloads to FP8 + per-token scales.
#   This may reduce bandwidth but adds quant/dequant overhead and may affect numerics.

_cp_options.enable_load_balance = False

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
        AttentionImpl,
    )

logger = logging.getLogger(__name__)


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _get_fp8_dtype_and_max() -> tuple[torch.dtype, float]:
    """Return (fp8_dtype, fp8_max) for current platform.

    We keep this local to avoid importing heavy quant modules into diffusion runtime.
    """
    # Default (CUDA, most platforms)
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    fp8_fnuz = getattr(torch, "float8_e4m3fnuz", None)
    if fp8_dtype is None:
        raise RuntimeError("torch does not support float8 on this build.")

    # ROCm MI300 uses e4m3fnuz; mimic sglang.srt.layers.quantization.fp8_kernel.is_fp8_fnuz
    try:
        if torch.version.hip is not None and fp8_fnuz is not None:
            props = torch.cuda.get_device_properties(0)
            if hasattr(props, "gcnArchName") and "gfx94" in str(props.gcnArchName):
                fp8_dtype = fp8_fnuz
                # fp8_max for fnuz follows ONNX spec; sglang uses 224.0
                return fp8_dtype, 224.0
    except Exception:
        # If we fail to query device properties, fall back to e4m3fn.
        pass

    fp8_max = float(torch.finfo(fp8_dtype).max)
    return fp8_dtype, fp8_max


def _per_token_quant_fp8(
    x_2d: torch.Tensor, fp8_dtype: torch.dtype, fp8_max: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize x_2d (M,K) -> (x_q_fp8 (M,K), x_s_f16 (M,1)).

    Quantizer selection:
    - If envs.SGLANG_USP_FP8_COMM_USE_LIGHTX2V: use LightX2V's quant_fp8_vllm
      (vLLM scaled_fp8_quant) to isolate quantizer effects.
    - Else: prefer sgl-kernel fast path when available, otherwise fall back to a torch impl.
    """
    assert x_2d.ndim == 2 and x_2d.is_contiguous()

    # Option A: vLLM quantizer (custom op), used to A/B against sgl-kernel.
    if envs.SGLANG_USP_FP8_COMM_USE_VLLM:
        try:
            from vllm import _custom_ops as ops  # type: ignore

            # Match LightX2V's quant_fp8_vllm behavior:
            # ops.scaled_fp8_quant(x, scale=None, scale_ub=None, use_per_token_if_dynamic=True)
            x_q, x_s = ops.scaled_fp8_quant(
                x_2d, scale=None, scale_ub=None, use_per_token_if_dynamic=True
            )
            # Normalize scale dtype to fp16 for comm size consistency.
            return x_q.contiguous(), x_s.to(torch.float16).contiguous()
        except Exception as e:
            logger.warning(
                "Requested vLLM FP8 quant (scaled_fp8_quant) but failed (%s); falling back to sgl-kernel/torch.",
                e,
            )

    # Fast path: sgl-kernel per-token quant (CUDA/ROCm builds with custom ops).
    try:
        from sgl_kernel import sgl_per_token_quant_fp8  # type: ignore

        x_q = torch.empty_like(x_2d, device=x_2d.device, dtype=fp8_dtype)
        x_s = torch.empty((x_2d.shape[0], 1), device=x_2d.device, dtype=torch.float32)
        sgl_per_token_quant_fp8(x_2d, x_q, x_s)
        return x_q, x_s.to(torch.float16)
    except Exception:
        pass

    # Fallback: pure torch per-row absmax quantization.
    x_f = x_2d.float()
    amax = x_f.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = (fp8_max / amax).to(torch.float32)
    x_q = (x_f * scale).clamp(min=-fp8_max, max=fp8_max).to(fp8_dtype).contiguous()
    x_s = (1.0 / scale).to(torch.float16).contiguous()
    return x_q, x_s


def _usp_all_to_all_single_fp8(x: torch.Tensor) -> torch.Tensor:
    """FP8-compressed all_to_all_single variant.

    It preserves the original tensor shape/dtype, but transmits FP8 + per-token scales.
    """
    ulysses_pg = get_sp_group().ulysses_group
    assert ulysses_pg is not None, "Ulysses process group is not initialized."

    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    if x.ndim == 0 or x.shape[-1] == 0:
        return x

    # We rely on the underlying all_to_all_single splitting the flattened buffer
    # evenly. For correct packing, each per-rank chunk must align to whole rows
    # of the (M, D) view (i.e., elem_per_rank % D == 0).
    d = int(x.shape[-1])
    if x.numel() % world_size != 0:
        return _usp_all_to_all_single(x)  # fallback to original
    elem_per_rank = x.numel() // world_size
    if elem_per_rank % d != 0:
        return _usp_all_to_all_single(x)  # fallback to original

    rows_per_rank = elem_per_rank // d
    m = x.numel() // d
    assert m == rows_per_rank * world_size

    fp8_dtype, fp8_max = _get_fp8_dtype_and_max()
    orig_dtype = x.dtype

    x_2d = x.contiguous().reshape(m, d)
    x_q, x_s16 = _per_token_quant_fp8(x_2d, fp8_dtype=fp8_dtype, fp8_max=fp8_max)

    # Direct all-to-all on FP8 payload + FP16 scales (2 collectives).
    #
    # This matches the LightX2V style and avoids byte packing/cat overhead.
    # Note: Requires backend to support FP8 collectives; otherwise we fall back.
    y_q_flat = ft_c.all_to_all_single(
        x_q.reshape(-1),
        output_split_sizes=None,
        input_split_sizes=None,
        group=ulysses_pg,
    )
    y_s_flat = ft_c.all_to_all_single(
        x_s16.reshape(-1),
        output_split_sizes=None,
        input_split_sizes=None,
        group=ulysses_pg,
    )
    y_q_flat = _maybe_wait(y_q_flat)
    y_s_flat = _maybe_wait(y_s_flat)

    y_q = y_q_flat.reshape(m, d)
    y_s16 = y_s_flat.reshape(m, 1)

    # Dequantize back to original dtype/shape.
    y_2d = (y_q.to(torch.float16) * y_s16).to(orig_dtype)
    return y_2d.reshape(x.shape)


def _usp_all_to_all_single_fp8_async(x: torch.Tensor):
    """Async FP8-compressed all_to_all_single. Returns wait() -> Tensor."""
    ulysses_pg = get_sp_group().ulysses_group
    assert ulysses_pg is not None, "Ulysses process group is not initialized."

    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return lambda: x

    if x.ndim == 0 or x.shape[-1] == 0:
        return lambda: x

    d = int(x.shape[-1])
    if x.numel() % world_size != 0:
        return _usp_all_to_all_single_async(x)
    elem_per_rank = x.numel() // world_size
    if elem_per_rank % d != 0:
        return _usp_all_to_all_single_async(x)

    rows_per_rank = elem_per_rank // d
    m = x.numel() // d
    assert m == rows_per_rank * world_size

    fp8_dtype, fp8_max = _get_fp8_dtype_and_max()
    orig_dtype = x.dtype
    x_2d = x.contiguous().reshape(m, d)
    x_q, x_s16 = _per_token_quant_fp8(x_2d, fp8_dtype=fp8_dtype, fp8_max=fp8_max)

    y_q = ft_c.all_to_all_single(
        x_q.reshape(-1),
        output_split_sizes=None,
        input_split_sizes=None,
        group=ulysses_pg,
    )
    y_s = ft_c.all_to_all_single(
        x_s16.reshape(-1),
        output_split_sizes=None,
        input_split_sizes=None,
        group=ulysses_pg,
    )

    def wait():
        y_q_flat = _maybe_wait(y_q)
        y_s_flat = _maybe_wait(y_s)
        y_q_2d = y_q_flat.reshape(m, d)
        y_s16_2d = y_s_flat.reshape(m, 1)
        y_2d = (y_q_2d.to(torch.float16) * y_s16_2d).to(orig_dtype)
        return y_2d.reshape(x.shape)

    return wait


def _usp_all_to_all_single(x: torch.Tensor) -> torch.Tensor:
    ulysses_pg = get_sp_group().ulysses_group
    assert ulysses_pg is not None, "Ulysses process group is not initialized."
    if envs.SGLANG_USP_FP8_COMM:
        try:
            return _usp_all_to_all_single_fp8(x)
        except Exception as e:
            logger.warning("USP FP8 comm failed, falling back to fp16/bf16: %s", e)
    x_shape = x.shape
    x = x.flatten()
    x = ft_c.all_to_all_single(
        x, output_split_sizes=None, input_split_sizes=None, group=ulysses_pg
    )
    x = _maybe_wait(x)
    x = x.reshape(x_shape)
    return x


def _usp_all_to_all_single_async(x: torch.Tensor):
    """
    Async variant of all_to_all_single.
    Returns a callable `wait()` that produces the shaped tensor.
    """
    ulysses_pg = get_sp_group().ulysses_group
    assert ulysses_pg is not None, "Ulysses process group is not initialized."
    if envs.SGLANG_USP_FP8_COMM:
        try:
            return _usp_all_to_all_single_fp8_async(x)
        except Exception as e:
            logger.warning(
                "USP FP8 comm (async) failed, falling back to fp16/bf16: %s", e
            )
    x_shape = x.shape
    x_flat = x.flatten()
    y = ft_c.all_to_all_single(
        x_flat, output_split_sizes=None, input_split_sizes=None, group=ulysses_pg
    )

    def wait():
        y_waited = _maybe_wait(y)
        return y_waited.reshape(x_shape)

    return wait


def _usp_input_all_to_all(x: torch.Tensor, head_dim: int = 1) -> torch.Tensor:
    """
    Perform Ulysses-style input all-to-all over the head dimension.

    Default layout expects heads at dim=1 and sequence at dim=2:
        [b, h, s_local, d] -> [b, h_local, s_global, d]

    If heads are at dim=2 (input is [b, s_local, h, d]), set head_dim=2, and the
    function returns [b, s_global, h_local, d], preserving the original
    head/sequence dim ordering.

    Args:
        x: A 4D tensor with layout [b, *, *, d] where '*' are sequence and heads
        head_dim: Which dimension index corresponds to heads (1 or 2)

    Returns:
        Tensor with the same dim order as input, with heads sharded and sequence gathered.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"
    seq_dim = 1 if head_dim == 2 else 2

    # Bring to canonical [b, h, s, d]
    if head_dim == 1 and seq_dim == 2:
        x_c = x
    else:
        x_c = x.permute(0, head_dim, seq_dim, 3).contiguous()

    b, h, s, d = x_c.shape
    assert (
        h % world_size == 0
    ), f"h ({h}) must be divisible by world_size ({world_size})"

    # [b, h, s_local, d] -> [h, b, s_local, d]
    x_c = x_c.permute(1, 0, 2, 3).contiguous()
    # all-to-all along h
    x_c = _usp_all_to_all_single(x_c)
    # -> [b, h_local, s, d]
    x_c = (
        x_c.reshape(world_size, h // world_size, b, -1, d)
        .permute(2, 1, 0, 3, 4)
        .reshape(b, h // world_size, -1, d)
    )

    if head_dim == 1 and seq_dim == 2:
        return x_c

    # Map back to original ordering, preserving head/seq positions
    new_order = [0, None, None, 3]
    new_order[head_dim] = 1
    new_order[seq_dim] = 2
    return x_c.permute(tuple(new_order)).contiguous()


def _usp_input_all_to_all_async(x: torch.Tensor, head_dim: int = 1):
    """
    Async variant of _usp_input_all_to_all. Returns a callable `wait()` producing the output tensor.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return lambda: x

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"
    seq_dim = 1 if head_dim == 2 else 2

    # Bring to canonical [b, h, s, d]
    if head_dim == 1 and seq_dim == 2:
        x_c = x
    else:
        x_c = x.permute(0, head_dim, seq_dim, 3).contiguous()

    b, h, _s, d = x_c.shape
    assert (
        h % world_size == 0
    ), f"h ({h}) must be divisible by world_size ({world_size})"

    # [b, h, s_local, d] -> [h, b, s_local, d]
    x_c = x_c.permute(1, 0, 2, 3).contiguous()
    wait_a2a = _usp_all_to_all_single_async(x_c)

    def wait():
        x_w = wait_a2a()
        # -> [b, h_local, s, d]
        x_out = (
            x_w.reshape(world_size, h // world_size, b, -1, d)
            .permute(2, 1, 0, 3, 4)
            .reshape(b, h // world_size, -1, d)
        )
        if head_dim == 1 and seq_dim == 2:
            return x_out
        new_order = [0, None, None, 3]
        new_order[head_dim] = 1
        new_order[seq_dim] = 2
        return x_out.permute(tuple(new_order)).contiguous()

    return wait


def _usp_input_all_to_all_qkv_async(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_dim: int = 1
):
    """
    Fused async Ulysses input all-to-all for Q/K/V.

    Instead of launching 3 separate all-to-all collectives, we pack QKV on the
    batch dimension and do a single collective:
      q,k,v: [B, S_local, H, D]  -> pack -> [3*B, S_local, H, D]
      then run one _usp_input_all_to_all_async, and unpack back.

    Returns a callable wait() -> (q_out, k_out, v_out) with outputs shaped like:
      [B, S_global, H_local, D]
    """
    assert q.shape == k.shape == v.shape, "q/k/v must have identical shapes"
    b = q.shape[0]
    x = torch.cat([q, k, v], dim=0)  # [3B, S_local, H, D]
    wait_a2a = _usp_input_all_to_all_async(x, head_dim=head_dim)

    def wait():
        x_out = wait_a2a()  # [3B, S_global, H_local, D]
        q_out, k_out, v_out = x_out[:b], x_out[b : 2 * b], x_out[2 * b : 3 * b]
        return q_out, k_out, v_out

    return wait


def _usp_output_all_to_all(x: torch.Tensor, head_dim: int = 1) -> torch.Tensor:
    """
    Perform Ulysses-style output all-to-all over the head dimension (inverse of input).

    Default layout expects heads at dim=1 and sequence at dim=2:
        [b, h_local, s, d] -> [b, h, s_local, d]

    If heads are at dim=2 (input is [b, s_global, h // world_size, d]), set head_dim=2,
    and the function returns [b, s_local, h, d], preserving the original head/sequence
    dim ordering.

    Args:
        x: A 4D tensor with layout [b, *, *, d] where '*' are sequence and heads
        head_dim: Which dimension index corresponds to heads (1 or 2)

    Returns:
        Tensor with the same dim order as input, with heads gathered and sequence sharded.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"
    seq_dim = 1 if head_dim == 2 else 2

    # Bring to canonical [b, h, s, d]
    if head_dim == 1 and seq_dim == 2:
        x_c = x
    else:
        x_c = x.permute(0, head_dim, seq_dim, 3).contiguous()

    b, h, s, d = x_c.shape
    assert (
        s % world_size == 0
    ), f"s ({s}) must be divisible by world_size ({world_size})"

    # [b, h_local, s, d] -> [s, b, h_local, d]
    x_c = x_c.permute(2, 0, 1, 3).contiguous()
    x_c = _usp_all_to_all_single(x_c)
    # -> [b, h, s_local, d]
    x_c = (
        x_c.reshape(world_size, s // world_size, b, -1, d)
        .permute(2, 0, 3, 1, 4)
        .reshape(b, -1, s // world_size, d)
    )

    if head_dim == 1 and seq_dim == 2:
        return x_c

    # Map back to original ordering, preserving head/seq positions
    new_order = [0, None, None, 3]
    new_order[head_dim] = 1
    new_order[seq_dim] = 2
    return x_c.permute(tuple(new_order)).contiguous()


def _usp_output_all_to_all_async(x: torch.Tensor, head_dim: int = 1):
    """
    Async variant of _usp_output_all_to_all. Returns a callable `wait()` producing the output tensor.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return lambda: x

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"
    seq_dim = 1 if head_dim == 2 else 2

    # Bring to canonical [b, h, s, d]
    if head_dim == 1 and seq_dim == 2:
        x_c = x
    else:
        x_c = x.permute(0, head_dim, seq_dim, 3).contiguous()

    b, h, s, d = x_c.shape
    assert (
        s % world_size == 0
    ), f"s ({s}) must be divisible by world_size ({world_size})"

    # [b, h_local, s, d] -> [s, b, h_local, d]
    x_c = x_c.permute(2, 0, 1, 3).contiguous()
    wait_a2a = _usp_all_to_all_single_async(x_c)

    def wait():
        x_w = wait_a2a()
        # -> [b, h, s_local, d]
        x_out = (
            x_w.reshape(world_size, s // world_size, b, -1, d)
            .permute(2, 0, 3, 1, 4)
            .reshape(b, -1, s // world_size, d)
        )
        if head_dim == 1 and seq_dim == 2:
            return x_out
        new_order = [0, None, None, 3]
        new_order[head_dim] = 1
        new_order[seq_dim] = 2
        return x_out.permute(tuple(new_order)).contiguous()

    return wait


def ring_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_impl: "AttentionImpl",
    is_causal: bool = False,
    dropout_p: float = 0.0,
):
    """
    Ring Attention implementation.

    This function implements Ring Attention, a strategy for distributed attention
    computation that reduces peak memory usage. It accepts a generic attention
    implementation (`attn_impl`) which is called by the underlying PyTorch
    distributed attention primitive.

    Args:
        query, key, value: The input tensors for attention.
        attn_impl: An instance of an attention implementation backend
                   (e.g., FlashAttentionImpl) whose `forward` method will be
                   used as the computational kernel.
        is_causal: Whether to apply causal masking.
        dropout_p: Dropout probability.
    """
    # torch.distributed.tensor.experimental._attention is not a public API,
    from torch.distributed.tensor.experimental._attention import (
        _templated_ring_attention,
    )

    ring_pg = get_sp_group().ring_group
    assert ring_pg is not None, "Ring process group is not initialized."

    # Ring attention primitives expect tensors in [B, H, S, D] layout.
    # We permute the inputs here.
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    # Create an adapter function that matches the signature expected by
    # _templated_ring_attention. The `attn_impl` already has dropout and
    # causal settings configured during its initialization.

    # Note: Please be aware that Attention Backend and Ring Attention may require different QKV tensor shapes.
    # For example, FlashAttention expects the format to be BSHD.
    def attn_callable_adapter(q, k, v, *args, **kwargs):
        # We ignore the dropout_p and is_causal passed by _templated_ring_attention
        # and rely on the pre-configured attn_impl.
        # The `attn_metadata` is not available here, so we pass None.
        # This is a limitation we must accept when using this experimental API.
        q = torch.permute(q, [0, 2, 1, 3])
        k = torch.permute(k, [0, 2, 1, 3])
        v = torch.permute(v, [0, 2, 1, 3])
        # logger.warning(f"Warning: return_s·oftmax_lse is only supported for FlashAttentionImpl")
        output, softmax_lse, *rest = attn_impl.forward(
            q,
            k,
            v,
            attn_metadata=None,
            return_softmax_lse=True,
        )
        output = torch.permute(output, [0, 2, 1, 3])
        return output, softmax_lse, *rest

    # Starting from torch 2.6.0, _templated_ring_attention expects an integer
    # segment_id for the attention function.
    use_segment_id = parse(torch.__version__).release >= parse("2.6.0").release

    attn_kwargs = dict(
        op=attn_callable_adapter,
        dropout_p=dropout_p,
        is_causal=is_causal,
        query=query,
        key=key,
        value=value,
        group=ring_pg,  # https://github.com/pytorch/pytorch/blob/c907c778f42ba2fdaf25b733dd25baf9779c6a12/torch/distributed/tensor/experimental/_context_parallel/_attention.py#L309
    )

    if use_segment_id:
        # For torch >= 2.6, segment_id is required. The value '1' is a placeholder
        # as we are not using complex segmentation features.
        out, *_ = _templated_ring_attention(
            seq_dim=1,  # segment_id
            **attn_kwargs,
        )
    else:
        out, *_ = _templated_ring_attention(
            **attn_kwargs,
        )

    # Permute the output back to [B, S, H, D] layout.
    output = torch.permute(out, [0, 2, 1, 3])
    return output
