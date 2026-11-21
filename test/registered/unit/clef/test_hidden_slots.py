"""Check the ordered rows delivered to the CLEF joint head after a cache hit."""

import unittest
from types import SimpleNamespace

import torch
from sglang.srt.layers.clef_hidden_slots import ClefHiddenSlots
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def rows(values):
    # Token 7 can have a different representation after different prefixes.
    return torch.tensor([[value, -value] for value in values], dtype=torch.bfloat16)


def batch(mapping, slots, tokens, expected):
    return SimpleNamespace(
        clef_req_to_token=torch.tensor(mapping, dtype=torch.int32),
        req_pool_indices=torch.arange(len(mapping)),
        out_cache_loc=torch.tensor(slots),
        input_ids=torch.tensor(tokens),
        clef_expected_prefixes=expected,
    )


class HiddenSlotTest(unittest.TestCase):
    def test_mixed_cold_and_hot_and_fully_reused_dynamic_rows(self):
        cache = ClefHiddenSlots(16, rows([0]))
        # A primer need not have a decision layout.
        cache.restore(
            rows([4, 6]), batch([[6, 2]], [6, 2], [1, 7], [[]]), [2], [0], [None]
        )
        layout = [10], [20]
        first = batch(
            [[6, 2, 9, 8], [1, 3, 0, 0]], [9, 8, 1, 3], [3, 4, 2, 7], [[1, 7], []]
        )
        hidden, ids, lengths, layouts = cache.restore(
            rows([7, 8, 1, 3]), first, [2, 2], [2, 0], layout
        )
        self.assertEqual(lengths, [4, 2])
        self.assertEqual(layouts, list(layout))
        self.assertTrue(torch.equal(hidden, rows([4, 6, 7, 8, 1, 3])))
        self.assertEqual(ids.tolist(), [1, 7, 3, 4, 2, 7])
        # Reuse newly written dynamic positions, from two requests in swapped order.
        second = batch(
            [[1, 3, 10, 0], [6, 2, 9, 11]], [10, 11], [8, 5], [[2, 7], [1, 7, 3]]
        )
        hidden, ids, lengths, _ = cache.restore(
            rows([9, 2]), second, [1, 1], [2, 3], layout
        )
        self.assertEqual(lengths, [3, 4])
        self.assertTrue(torch.equal(hidden, rows([1, 3, 9, 4, 6, 7, 2])))
        self.assertEqual(ids.tolist(), [2, 7, 8, 1, 7, 3, 5])

    def test_eviction_and_missing_identity_suppress_only_the_bad_request(self):
        cache = ClefHiddenSlots(12, rows([0]))
        cache.restore(
            rows([10, 20]), batch([[1, 4]], [1, 4], [2, 7], [[]]), [2], [0], [None]
        )
        cache.restore(
            rows([3, 4]), batch([[1, 4]], [1, 4], [9, 7], [[]]), [2], [0], [None]
        )
        expected = [[9, 7], [2, 7], [1]]
        mixed = batch(
            [[1, 4, 8], [1, 4, 9], [3, 11, 0]], [8, 9, 11], [5, 6, 2], expected
        )
        hidden, ids, lengths, layout = cache.restore(
            rows([7, 8, 9]), mixed, [1, 1, 1], [2, 2, 1], [1, 2, 3]
        )
        self.assertEqual(layout, [1, None, None])
        self.assertEqual(lengths, [3, 1, 1])
        self.assertTrue(torch.equal(hidden, rows([3, 4, 7, 8, 9])))
        self.assertEqual(ids.tolist(), [9, 7, 5, 6, 2])

    def test_padded_writes_are_ignored(self):
        cache = ClefHiddenSlots(4, rows([0]))
        pad = batch([[1, 2, 0]], [1, 2, 0], [4, 5, 0], [[]])
        hidden, ids, lengths, _ = cache.restore(rows([3, 4, 0]), pad, [2], [0], [[1]])
        self.assertEqual(hidden.tolist(), rows([3, 4]).tolist())
        self.assertEqual(ids.tolist(), [4, 5])
        self.assertEqual(cache.ids[0].item(), -1)
        self.assertEqual(lengths, [2])


if __name__ == "__main__":
    unittest.main()
