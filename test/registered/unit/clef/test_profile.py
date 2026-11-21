"""Check automatic CLEF resolution and the native opt-out using synthetic metadata."""

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

from sglang.srt import clef_policy_cache
from sglang.srt.arg_groups.model_override_base import model_config_of, resolving_view
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.runtime_context import get_context, override_platform, reset_context
from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class AutoProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.model = Path(temp.name)
        # Only metadata is loaded by the CLI resolver. The fixture needs no tokenizer or weights.
        config = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "dtype": "bfloat16",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 2560,
                "num_hidden_layers": 32,
                "full_attention_interval": 4,
                "mamba_ssm_dtype": "float32",
                "dtype": "bfloat16",
            },
        }
        (cls.model / "config.json").write_text(json.dumps(config))
        (cls.model / "joint_head_config.json").write_text("{}")

    def tearDown(self):
        reset_context()

    def server(self, **kwargs):
        args = ServerArgs(model_path=str(self.model), device="cuda", **kwargs)
        with override_platform(is_cuda=True):
            clef_policy_cache.resolve(args, model_config_of(args))
        return args

    @patch.dict(
        os.environ,
        {
            "SGLANG_CLEF_PREFIX_CACHE": "0",
            "SGLANG_CLEF_POLICY_PREFIX_IDS": "/nonexistent/unused",
        },
    )
    def test_regular_launch_uses_resolved_profile_without_any_policy(self):
        args = self.server()
        effective = resolving_view(args)
        self.assertTrue(clef_policy_cache.enabled(args))
        self.assertIsNone(args.attention_backend)
        self.assertEqual(effective.attention_backend, "triton")
        self.assertEqual(effective.mamba_radix_cache_strategy, "extra_buffer")
        self.assertEqual(effective.page_size, 1)
        self.assertTrue(effective.enable_deterministic_inference)
        self.assertEqual(effective.linear_attn_prefill_backend, "triton")

    def test_full_argument_pipeline_and_native_opt_out(self):
        for extras, expected in (([], True), (["--disable-radix-cache"], False)):
            args = prepare_server_args(
                [
                    "--model-path",
                    str(self.model),
                    "--device",
                    "cuda",
                    "--context-length",
                    "8192",
                    "--mem-fraction-static",
                    "0.5",
                    *extras,
                ]
            )
            with (
                override_platform(is_cuda=True),
                patch("torch.cuda.get_device_capability", return_value=(8, 0)),
                patch("torch.cuda.get_device_properties") as props,
            ):
                props.return_value.total_memory = 80 * 1024**3
                props.return_value.major = 8
                props.return_value.minor = 0
                props.return_value.name = "A800"
                args.resolve_once()
            self.assertEqual(clef_policy_cache.enabled(args), expected)
            cfg = args.resolved_dict()
            self.assertEqual(cfg["disable_radix_cache"], not expected)
            self.assertEqual(cfg["enable_deterministic_inference"], expected)
            self.assertTrue(cfg["is_embedding"])
            self.assertEqual(cfg["chunked_prefill_size"], -1)

    def test_original_config_kept_for_unsupported_options(self):
        for choices in (
            {"disable_radix_cache": True},
            {"attention_backend": "fa3"},
            {"mamba_radix_cache_strategy": "no_buffer"},
            {"page_size": 64},
            {"mamba_ssm_dtype": "bfloat16"},
            {"enable_lora": True},
            {"linear_attn_backend": "flashinfer"},
            {"dtype": "float16"},
        ):
            with self.subTest(choices=choices):
                args = self.server(**choices)
                self.assertFalse(clef_policy_cache.enabled(args))
                self.assertFalse(resolving_view(args).enable_deterministic_inference)

    @patch.dict(os.environ, {"SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "rust"})
    def test_explicit_alternative_tree(self):
        self.assertFalse(clef_policy_cache.enabled(self.server()))

    def test_engine_restart_and_pickled_worker_config(self):
        kernel = object.__new__(TritonGDNKernel)
        self.assertFalse(clef_policy_cache.enabled())
        active = pickle.loads(pickle.dumps(self.server()))
        with patch(
            "sglang.srt.layers.attention.linear.kernels.gdn_triton.is_cuda",
            return_value=True,
        ):
            with patch.object(
                type(get_context()), "server_args", new_callable=PropertyMock
            ) as published:
                published.return_value = active
                self.assertTrue(clef_policy_cache.enabled())
                self.assertTrue(kernel.supports_track_state_snapshot)
                published.return_value = self.server(disable_radix_cache=True)
                self.assertFalse(clef_policy_cache.enabled())
                self.assertFalse(kernel.supports_track_state_snapshot)


if __name__ == "__main__":
    unittest.main()
