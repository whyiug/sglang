"""System One response, recovery and per-request IPC metrics for the CLEF cache."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from sglang.srt.entrypoints.systemone.serving import (
    SystemOneRequest,
    SystemOneServing,
    _view,
)
from sglang.srt.managers import io_struct
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.multi_tokenizer_mixin import _handle_output_by_index
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from starlette.requests import Request

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def result(cached, logits):
    return {
        "embedding": logits,
        "meta_info": {
            "id": "main",
            "prompt_tokens": 10,
            "cached_tokens": cached,
            "clef_cache_trace": {
                "matched_tokens": 7,
                "branch_boundary": 6,
                "published_tokens": 0,
            },
        },
    }


class Replies:
    served_model_name = "fixture"

    def __init__(self, *replies, active=True):
        self.server_args = SimpleNamespace(_clef_auto_cache_active=active)
        self.replies = iter(replies)
        self.inputs = []

    async def generate_request(self, obj, raw_request):
        self.inputs.append(obj)
        yield next(self.replies)


class RecoveryTest(unittest.IsolatedAsyncioTestCase):
    async def score(self, replies, *, report=False):
        request = SystemOneRequest(
            model="fixture",
            state="synthetic",
            questions={
                "q": {"type": "choice", "criteria": {"A": "first", "B": "second"}}
            },
        )
        service = SimpleNamespace(
            tokenizer_manager=replies,
            clef_temperatures=[1, 1, 1],
            _encode_joint_schema_prompt=AsyncMock(),
        )
        prompt = EmbeddingReqInput(rid="business-rid", input_ids=list(range(10)))
        headers = [(b"x-sglang-clef-cache-report", b"1")] if report else []
        client_request = Request({"type": "http", "headers": headers})
        response = await SystemOneServing._answer_joint_schema(
            service,
            prompt,
            request,
            [_view(request.questions["q"])],
            client_request,
        )
        return json.loads(response.body)

    async def test_native_and_warm_response_schema_and_scores(self):
        logits = [0.3, 1.5]
        native = {"embedding": logits, "meta_info": {"prompt_tokens": 10}}
        reference = await self.score(Replies(native, active=False), report=True)
        self.assertEqual(set(reference), {"model", "usage", "answers"})
        self.assertEqual(reference["usage"], {"input_tokens": 10, "output_tokens": 0})
        self.assertEqual(await self.score(Replies(result(6, logits))), reference)
        self.assertEqual(
            await self.score(Replies(result(6, []), result(0, logits))), reference
        )
        diagnostic = await self.score(Replies(result(6, logits)), report=True)
        self.assertEqual(diagnostic["answers"], reference["answers"])
        self.assertEqual(
            diagnostic["metadata"]["clef_prefix_cache"]["matched_tokens"], 7
        )
        self.assertEqual(
            diagnostic["usage"]["prompt_tokens_details"]["cached_tokens"], 6
        )

    async def test_failed_hit_retries_once_and_accounts_for_both_attempts(self):
        cold = result(0, [0.5, 1.5])
        cold["meta_info"]["id"] = "retry"
        service = Replies(result(6, []), cold)
        reply = await self.score(service, report=True)
        self.assertEqual(len(service.inputs), 2)
        first, second = service.inputs
        self.assertEqual(first.input_ids, second.input_ids)
        self.assertEqual(first.rid, "business-rid")
        self.assertTrue(second.rid.startswith("clef-cache-retry-"))
        self.assertEqual(
            reply["metadata"]["clef_prefix_cache"],
            {
                "status": "fallback",
                "cached_tokens": 0,
                "engine_cached_tokens": 6,
                "computed_tokens": 14,
                "retry_tokens": 10,
                "rid": "main",
                "retry_rid": "retry",
                "matched_tokens": 7,
                "branch_boundary": 6,
                "published_tokens": 0,
            },
        )

    async def test_corrupt_recovery_never_commits_an_answer_or_retries_again(self):
        for attempts in [
            (result(0, []),),
            (result(6, [1.0]),),
            (result(6, []), result(2, [0.5, 1.5])),
            (result(6, []), result(0, [])),
        ]:
            service = Replies(*attempts)
            with self.subTest(attempts=len(attempts)), self.assertRaises(RuntimeError):
                await self.score(service)
            self.assertEqual(len(service.inputs), len(attempts))

    async def test_abort_is_terminal_on_both_initial_and_recovery_requests(self):
        aborted = {
            "meta_info": {
                "finish_reason": {"type": "abort", "message": "aborted by scheduler"}
            }
        }
        for attempts in [(aborted,), (result(6, []), aborted)]:
            service = Replies(*attempts)
            with (
                self.subTest(attempts=len(attempts)),
                self.assertRaises(HTTPException) as error,
            ):
                await self.score(service)
            self.assertEqual(error.exception.status_code, 409)
            self.assertEqual(len(service.inputs), len(attempts))


class IPCReceiptTest(unittest.TestCase):
    def test_batch_uses_only_finished_requests_and_preserves_individual_traces(self):
        def req(name, done, restored, tree_match, branch, published):
            return SimpleNamespace(
                finished=lambda: done,
                finished_reason=SimpleNamespace(to_json=lambda: {}),
                rid=name,
                http_worker_ipc="local-test",
                embedding=[1.0],
                origin_input_ids=list(range(10)),
                cached_tokens=restored,
                time_stats=None,
                retraction_count=0,
                pooled_hidden_state=None,
                clef_full_kv_match=tree_match,
                mamba_branching_seqlen=branch,
                clef_checkpoint_published=published,
            )

        sender = Mock()
        streamer = SimpleNamespace(
            server_args=SimpleNamespace(_clef_auto_cache_active=True),
            send_to_detokenizer=sender,
            get_cached_tokens_details=lambda _: None,
        )
        reqs = [
            req("a", True, 0, 0, None, 8),
            req("unfinished", False, 0, 0, 0, 0),
            req("b", True, 5, 8, 7, 7),
        ]
        with patch.object(io_struct, "_USE_PICKLE_IPC", False):
            SchedulerOutputStreamer._stream_output_embedding(streamer, reqs)
            batch = sender.send_output.call_args.args[0]
            self.assertEqual(batch.rids, ["a", "b"])
            decoded = msgpack_decode(msgpack_encode(batch))
            b = _handle_output_by_index(decoded, 1)
        self.assertEqual(b.cached_tokens, [5])
        self.assertEqual(b.retraction_counts, [0])
        self.assertEqual(
            b.clef_cache_traces,
            [{"matched_tokens": 8, "branch_boundary": 7, "published_tokens": 7}],
        )


if __name__ == "__main__":
    unittest.main()
