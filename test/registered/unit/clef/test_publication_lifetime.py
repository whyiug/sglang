"""Check mixed outcomes and slot lifetimes with the upstream radix/allocator fixture."""

import os
import sys
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class PublicationLifetime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_dir = Path(__file__).resolve().parent.parent / "mem_cache"
        sys.path.insert(0, str(fixture_dir))
        try:
            # The upstream test module derives its test port from the first
            # visible device character on import. Reserve a nonexistent index.
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "99"}):
                import test_unified_radix_cache_unittest as upstream
        finally:
            sys.path.remove(str(fixture_dir))
        cls.sg = upstream

    def setUp(self):
        self.addCleanup(reset_context)
        sg = self.sg
        config = sg.CacheConfig(
            components=(sg.ComponentType.FULL, sg.ComponentType.MAMBA),
            enable_mamba_extra_buffer=True,
            num_layers=2,
            full_attention_layer_ids=(1,),
            kv_size=192,
            max_context_len=256,
        )
        with patch.object(sg, "get_device", return_value="cpu"):
            self.cache, self.alloc, self.pool = sg.build_fixture(
                config, mamba_cache_chunk_size=64
            )
        self.initial_mamba = self.pool.mamba_allocator.available_size()
        self.initial_kv = self.alloc.available_size()

    def request(self, tokens, salt):
        sg = self.sg
        sampling = sg.SamplingParams(temperature=0, max_new_tokens=1)
        sampling.normalize(None)
        req = sg.Req(
            rid=f"test-{salt}",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=sampling,
            cache_salt=salt,
        )
        self.pool.alloc([req])
        req.kv.kv_committed_len = req.kv.kv_allocated_len = len(tokens)
        req.last_node = self.cache.root_node_handle()
        req.kv.mamba_last_track_seqlen = len(tokens)
        req.kv.mamba_last_track_idx = 0
        req.kv.mamba_next_track_idx = self.pool.get_mamba_ping_pong_other_idx(0)
        req.clef_full_kv_match = req.clef_checkpoint_published = 0
        indices = self.alloc.alloc(len(tokens))
        self.pool.write((req.kv.req_pool_idx, slice(0, len(tokens))), indices)
        return req

    def match(self, tokens, salt):
        sg = self.sg
        return self.cache.match_prefix(
            sg.MatchPrefixParams(key=sg.RadixKey(array("q", tokens), cache_salt=salt))
        )

    def finish_batch(self, requests, scores):
        import torch
        from sglang.srt.managers.scheduler_components.batch_result_processor import (
            SchedulerBatchResultProcessor,
        )

        processor = SimpleNamespace(
            is_generation=False,
            tree_cache=self.cache,
            token_to_kv_pool_allocator=self.alloc,
            output_streamer=Mock(),
            _convert_embeddings=lambda *, result: [
                x.tolist() for x in result.embeddings
            ],
        )
        result = SimpleNamespace(
            copy_done=None,
            pooled_hidden_states=None,
            can_run_cuda_graph=False,
            embeddings=[torch.tensor(x) for x in scores],
        )
        batch = SimpleNamespace(reqs=requests, return_logprob=False, prefill_stats=None)
        SchedulerBatchResultProcessor.process_batch_result_prefill(
            processor, batch, result
        )

    def test_mixed_batch_only_successful_decision_publishes(self):
        """A rejected hidden row and a cancelled peer cannot seed an auto branch."""
        sg = self.sg
        tokens = list(range(64))
        good, invalid, cancelled = [
            self.request(tokens, salt) for salt in ("ok", "invalid", "cancel")
        ]
        cancelled.discard_output_reason = sg.FINISH_ABORT("test cancellation")
        self.finish_batch([good, invalid, cancelled], [[1], [], [2]])
        self.assertEqual(good.clef_checkpoint_published, 64)
        for request in (invalid, cancelled):
            self.assertEqual(request.clef_checkpoint_published, 0)
            self.assertEqual(
                self.match(tokens, request.cache_salt).full_kv_hit_length, 0
            )
        self.assertEqual(self.match(tokens, "ok").device_prefix_len, 64)
        self.assertEqual(self.alloc.available_size(), self.initial_kv - 64)
        self.assertEqual(
            self.pool.mamba_allocator.available_size(), self.initial_mamba - 1
        )
        self.cache.sanity_check()

    def test_pinned_branch_survives_eviction_and_freed_slots_can_be_reused(self):
        """Two namespaced uses of the identical tokens must have distinct owners."""
        sg = self.sg
        tokens = list(range(64))
        self.finish_batch(
            [self.request(tokens, "a"), self.request(tokens, "b")], [[1], [2]]
        )
        a, b = [self.match(tokens, salt) for salt in ("a", "b")]
        self.assertNotEqual(a.last_device_node, b.last_device_node)
        old_slots = {
            slot
            for match in (a, b)
            for slot in self.cache.path_device_indices(match.last_device_node).tolist()
        }
        self.assertEqual(self.match(tokens, "c").device_prefix_len, 0)
        lock = self.cache.lock(a.last_device_node)
        try:
            self.cache.evict(sg.EvictParams(num_tokens=512, mamba_num=20))
            self.assertEqual(self.match(tokens, "a").device_prefix_len, 64)
            self.assertEqual(self.match(tokens, "b").device_prefix_len, 0)
        finally:
            self.cache.unlock(lock)
        self.cache.evict(sg.EvictParams(num_tokens=512, mamba_num=20))
        self.assertEqual(self.alloc.available_size(), self.initial_kv)
        self.assertEqual(self.pool.mamba_allocator.available_size(), self.initial_mamba)
        new_slots = set()
        for i in range(3):
            new_tokens = [tokens[0] + i + 1, *tokens[1:]]
            salt = f"new-{i}"
            self.finish_batch([self.request(new_tokens, salt)], [[3]])
            new_branch = self.match(new_tokens, salt)
            self.assertEqual(new_branch.device_prefix_len, len(new_tokens))
            new_slots.update(
                self.cache.path_device_indices(new_branch.last_device_node).tolist()
            )
        self.assertTrue(old_slots.issubset(new_slots))
        self.assertEqual(self.match(tokens, "a").device_prefix_len, 0)
        self.assertEqual(self.match(tokens, "b").device_prefix_len, 0)
        self.cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
