"""Final CLEF token representations indexed by SGLang's resident KV slots."""

import torch


class ClefHiddenSlots:
    def __init__(self, capacity, example):
        if example.ndim != 2 or example.element_size() != 2:
            raise ValueError("CLEF hidden slots require a 16-bit token-by-hidden tensor")
        self.hidden = example.new_empty((capacity, example.shape[1]))
        # The sentinel makes an unwritten slot distinguishable from every token ID.
        self.ids = torch.full((capacity,), -1, device=example.device, dtype=torch.long)

    def restore(self, hidden, batch, lengths, prefixes, layouts):
        """Write fresh rows, verify each hit, then rebuild the original head input.

        The scheduler owns slot lifetime, pins hits for the duration of the
        forward, and overwrites evicted slots with the next prefill. This is
        also called for a primer without a decision layout.
        """
        if not (len(prefixes) == len(layouts) == len(lengths)):
            raise ValueError("Inconsistent CLEF prefill batch lengths")
        count = sum(lengths)
        if min(hidden.shape[0], batch.input_ids.numel(), batch.out_cache_loc.numel()) < count:
            raise ValueError("CLEF hidden rows do not cover all extended tokens")
        fresh = hidden[:count]
        fresh_ids = batch.input_ids[:count]
        writes = batch.out_cache_loc[:count].long()
        self.hidden.index_copy_(0, writes, fresh)
        self.ids.index_copy_(0, writes, fresh_ids.long())
        if not any(prefixes):
            return fresh, fresh_ids, lengths, layouts

        prefix_slots = []
        checks = []
        hit_indexes = []
        for i, prefix in enumerate(prefixes):
            if not prefix:
                prefix_slots.append(None)
                continue
            # Index by a one-element device tensor to avoid synchronizing the
            # GPU to convert each request slot to a Python scalar.
            slots = batch.clef_req_to_token[batch.req_pool_indices[i:i+1], :prefix].reshape(-1).long()
            expected = torch.as_tensor(batch.clef_expected_prefixes[i], dtype=torch.long, device=fresh.device)
            if expected.numel() != prefix:
                raise ValueError("Missing CLEF request token IDs for a cache hit")
            prefix_slots.append(slots)
            hit_indexes.append(i)
            checks.append((self.ids.index_select(0, slots) == expected).all())
        valid = dict(zip(hit_indexes, torch.stack(checks).tolist()))
        rows, ids, full_lengths, valid_layouts = [], [], [], list(layouts)
        offset = 0
        for i, (length, prefix, slots) in enumerate(zip(lengths, prefixes, prefix_slots)):
            if prefix and valid[i]:
                rows.append(self.hidden.index_select(0, slots))
                ids.append(self.ids.index_select(0, slots))
                full_lengths.append(prefix + length)
            else:
                full_lengths.append(length)
                if prefix:
                    # Suppress this request's head. The endpoint retries it cold.
                    valid_layouts[i] = None
            rows.append(fresh[offset:offset + length])
            ids.append(fresh_ids[offset:offset + length])
            offset += length
        return torch.cat(rows), torch.cat(ids), full_lengths, valid_layouts
