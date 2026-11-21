"""Automatic CLEF cache capability, stored on the engine's resolved ServerArgs."""

import logging

logger = logging.getLogger(__name__)
FIXED_WEIGHTS = "Restart the CLEF cache instance to change its model weights."


def enabled(server_args=None):
    """Read the same pickled resolution in the HTTP, scheduler and worker processes."""
    if server_args is None:
        from sglang.srt.runtime_context import get_server_args

        try:
            server_args = get_server_args()
        except ValueError:  # No engine has been published in this process.
            return False
    return getattr(server_args, "_clef_auto_cache_active", False)


def unsupported_reason(cfg, model, *, cuda, explicit_tree_backend):
    if cfg.disable_radix_cache:
        return "radix cache explicitly disabled"
    text = model.hf_text_config
    if (model.hf_config.architectures[0] != "Qwen3_5ForConditionalGeneration"
            or getattr(text, "model_type", None) != "qwen3_5_text"
            or (text.hidden_size, text.num_hidden_layers) != (2560, 32)
            or set(getattr(text, "layers_block_type", ())) != {"attention", "linear_attention"}
            or getattr(text, "is_causal", True) is False):
        return "outside the causal Qwen3.5-4B joint-head profile"
    if not cuda or str(cfg.device).split(":")[0] != "cuda":
        return "requires NVIDIA CUDA"
    if (cfg.tp_size, cfg.pp_size, cfg.dp_size, cfg.dcp_size) != (1, 1, 1, 1):
        return "requires a single local GPU"
    if (str(model.dtype) != "torch.bfloat16" or cfg.quantization or model.quantization
            or cfg.kv_cache_dtype not in ("auto", "bf16", "bfloat16")
            or (cfg.mamba_ssm_dtype or getattr(text, "mamba_ssm_dtype", None)) != "float32"):
        return "requires unquantized BF16 with FP32 recurrent state"
    for name in ("attention_backend", "prefill_attention_backend", "decode_attention_backend",
                 "mamba_backend", "linear_attn_backend", "linear_attn_prefill_backend",
                 "linear_attn_decode_backend"):
        if getattr(cfg, name) not in (None, "triton"):
            return f"{name} selects a backend outside the deterministic Triton profile"
    if cfg.mamba_radix_cache_strategy not in ("auto", "extra_buffer") or cfg.page_size not in (None, 1):
        return "requires extra_buffer hybrid checkpoints and 1-token KV pages"
    if explicit_tree_backend not in (None, "python") or cfg.radix_cache_backend is not None:
        return "requires the device-local unified Python radix tree"
    if cfg.disaggregation_mode != "null" or cfg.speculative_algorithm:
        return "requires ordinary single-engine execution"
    if cfg.enable_lora or cfg.lora_paths or cfg.uno_lora_path:
        return "requires fixed weights"
    for name in ("enable_hierarchical_cache", "enable_unified_memory", "enable_hisparse",
                 "enable_int8_mamba_checkpoint", "enable_memory_saver", "enable_lmcache",
                 "enable_flexkv", "enable_unified_cache_external_linker", "enable_kv_cache_sharding",
                 "enable_session_radix_cache", "prefill_only_disable_kv_cache"):
        if getattr(cfg, name):
            return f"{name} is outside the device-local cache profile"
    return None


def resolve(server_args, model_config):
    if model_config.joint_head_config is None:
        return

    from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_platform

    tree_backend = envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND
    reason = unsupported_reason(
        resolving_view(server_args), model_config, cuda=get_platform().is_cuda,
        explicit_tree_backend=tree_backend.get() if tree_backend.is_set() else None,
    )
    server_args._clef_auto_cache_active = reason is None
    if reason is not None:
        logger.info("CLEF automatic prefix cache disabled: %s", reason)
        return

    declare_resolution(server_args, "clef_auto_cache", attention_backend="triton",
                       enable_deterministic_inference=True, mamba_radix_cache_strategy="extra_buffer",
                       page_size=1, linear_attn_prefill_backend="triton")
