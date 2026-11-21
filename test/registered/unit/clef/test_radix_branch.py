"""Exercise admission, checkpoint publication and eviction on the CPU radix tree."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class AutoBranchTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(reset_context)
        args = ServerArgs(model_path="dummy", device="cpu", page_size=1)
        args._mamba_cache_chunk_size = 64
        args._clef_auto_cache_active = True
        set_global_server_args_for_scheduler(args)
        self.pool = MagicMock(spec=HybridReqToTokenPool)
        self.pool.mamba_ckpt_pool = None
        self.pool.mamba_allocator = MagicMock()
        self.pool.get_mamba_ping_pong_keep_idx.side_effect = lambda req: (
            req.kv.mamba_last_track_idx
        )
        self.pool.get_mamba_ping_pong_other_idx.side_effect = lambda i: 1 - i
        self.pool.req_to_token = torch.zeros((1, 1024), dtype=torch.long)
        self.pool.write.side_effect = self.pool.req_to_token.__setitem__
        allocator = MagicMock()
        allocator.device = torch.device("cpu")
        allocator.page_size = 1
        self.tree = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                tree_components=(ComponentType.FULL, ComponentType.MAMBA),
                enable_mamba_extra_buffer=True,
                tree_core_backend="python",
            )
        )
        self.count = 0

    def admit(self, tokens, *, salt=None, rid=None, layout=(1,)):
        self.count += 1
        req = Req(
            rid=rid or str(self.count),
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(max_new_tokens=1),
            decision_layout=layout,
            cache_salt=salt,
        )
        req.kv.mamba_pool_idx = torch.tensor(3 * self.count)
        req.init_next_round_input(self.tree, cow_mamba=True)
        return req

    def finish(self, req):
        prefix, length = req.prefix_len, len(req.origin_input_ids)
        req.kv.req_pool_idx = 0
        allocation = req.kv.mamba_pool_idx.item()
        req.kv.mamba_ping_pong_track_buffer = torch.tensor(
            [allocation + 1, allocation + 2]
        )
        req.kv.mamba_next_track_idx = 0
        req.extend_end = length
        self.pool.req_to_token[0, :prefix] = self.tree.path_device_indices(
            req.last_node
        )
        self.pool.req_to_token[0, prefix:length] = 1024 * allocation + torch.arange(
            prefix, length
        )
        req.lock = self.tree.lock(req.last_node)
        batch = ScheduleBatch(reqs=[req])
        batch.tree_cache = self.tree
        batch.req_to_token_pool = self.pool
        batch.model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(mamba_chunk_size=64)
        )
        batch._mamba_radix_cache_v2_req_prepare_for_extend(req)
        req.finished_reason = FINISH_LENGTH(1)
        self.tree.checkpoint(req, up_to=length)
        self.tree.unlock(req.lock)
        return req.clef_checkpoint_published

    def test_cold_then_branch_then_hit_and_isolation(self):
        # The 213-token common path ends between grid positions 192 and 256.
        prefix = list(range(213))
        a = self.admit(prefix + [2001] * 109)
        self.assertEqual((a.clef_full_kv_match, a.prefix_len), (0, 0))
        self.assertEqual(self.finish(a), 320)
        b = self.admit(prefix + [2002] * 109)
        self.assertEqual(
            (b.clef_full_kv_match, b.prefix_len, b.mamba_branching_seqlen),
            (213, 0, 192),
        )
        self.assertEqual(self.finish(b), 192)
        c = self.admit(prefix + [2003] * 109)
        self.assertEqual((c.clef_full_kv_match, c.prefix_len), (213, 192))
        self.assertEqual(self.finish(c), 320)

        # A second, longer branch of a previous request can be discovered too.
        same_state = self.admit(prefix + [2001] * 64 + [2004] * 45)
        self.assertEqual(
            (same_state.clef_full_kv_match, same_state.prefix_len), (277, 192)
        )
        self.assertEqual(self.finish(same_state), 256)
        self.assertEqual(self.admit(prefix + [2001] * 64 + [2005] * 45).prefix_len, 256)

        isolated = self.admit(prefix + [2001] * 109, salt="another-tenant")
        self.assertEqual((isolated.clef_full_kv_match, isolated.prefix_len), (0, 0))
        for isolated in (
            self.admit(prefix, rid="clef-cache-retry-example"),
            self.admit(prefix, layout=None),
        ):
            self.assertEqual(isolated.prefix_len, 0)
            self.assertTrue(isolated.skip_radix_cache_insert)
        self.tree.reset()
        self.assertEqual(self.admit(prefix + [2003] * 109).clef_full_kv_match, 0)

    def test_concurrent_donation_and_eviction_respect_tree_lock(self):
        tokens = list(range(322))
        first, concurrent = self.admit(tokens), self.admit(tokens)
        self.assertEqual((first.prefix_len, concurrent.prefix_len), (0, 0))
        self.assertEqual(self.finish(first), 320)
        self.assertEqual(self.finish(concurrent), 0)  # Published earlier by first.
        reader = self.admit(tokens)
        self.assertEqual(reader.prefix_len, 320)
        pin = self.tree.lock(reader.last_node)
        self.assertEqual(
            self.tree.evict(EvictParams(num_tokens=500)).num_tokens_evicted, 0
        )
        self.tree.unlock(pin)
        self.assertEqual(
            self.tree.evict(EvictParams(num_tokens=500)).num_tokens_evicted, 320
        )
        self.assertEqual(self.admit(tokens).prefix_len, 0)


if __name__ == "__main__":
    unittest.main()
