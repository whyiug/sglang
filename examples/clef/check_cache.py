"""Exercise CLEF auto-prefix caching through the normal System One HTTP API."""

import argparse
import json
import os
from pathlib import Path
import re
import secrets
import time
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


QUESTIONS = {
    "q01": {"type": "noul", "instructions": "Does the customer request a refund?"},
    "q03": {
        "type": "choice",
        "instructions": "Which team should respond?",
        "criteria": {"A": "Billing", "B": "Technical support", "C": "Sales"},
    },
    "q04": {
        "type": "score",
        "instructions": "How urgently is a response required?",
        "criteria": ["No urgency", "Soon", "Immediately"],
    },
}
TICKETS = [
    "Billing charged the invoice twice. Please refund the extra payment.",
    "Technical support is needed immediately. The dashboard is offline.",
    "Sales should send the new price list when it becomes available.",
]


def sample_requests(model, nonce):
    # Start with a run-specific value so repeated probes work on a running service
    # without flushing its cache or reusing this policy from a previous run.
    policy = f"Policy {nonce}\n" + "\n".join(
        f"Rule {n}: Use only the stated ticket facts to determine ownership, "
        "urgency and refund intent. Record uncertainty when evidence is missing."
        for n in range(1, 53)
    )
    return [
        {
            "model": model,
            "state": f"{policy}\n\nDYNAMIC STATE:\n{ticket}",
            "questions": QUESTIONS,
        }
        for ticket in TICKETS
    ]


def post(url, payload, timeout):
    request = Request(
        url.rstrip("/") + "/v1/systemone",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-SGLang-Clef-Cache-Report": "1"},
    )
    start = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as reply:
            body = json.load(reply)
    except HTTPError as error:
        raise ValueError(f"{request.full_url} returned HTTP {error.code}") from error
    elapsed = (time.perf_counter() - start) * 1000
    if body["model"] != payload["model"]:
        raise ValueError("unexpected served model name")
    answers = body["answers"]
    if set(answers) != set(QUESTIONS) or any(
        answers[q]["type"] != spec["type"] for q, spec in QUESTIONS.items()
    ):
        raise ValueError("expected answers for all three requested questions")
    return body, elapsed


def restored_work(body):
    receipt = (body.get("metadata") or {}).get("clef_prefix_cache")
    if not receipt:
        raise ValueError(
            "missing CLEF cache receipt; check model eligibility and server flags"
        )
    usage = body["usage"]
    total = usage["input_tokens"]
    cached, computed = receipt["cached_tokens"], receipt["computed_tokens"]
    if not (
        0 <= cached < total
        and cached
        == receipt["engine_cached_tokens"]
        == usage["prompt_tokens_details"]["cached_tokens"]
        and computed == total - cached
        and receipt["status"] == ("hit" if cached else "miss")
        and receipt["retry_tokens"] == 0
    ):
        raise ValueError(
            "inconsistent CLEF token accounting or fallback to full computation"
        )
    return total, cached, computed


def reference_result(url, payload, timeout):
    body, elapsed = post(url, payload, timeout)
    if (body.get("metadata") or {}).get("clef_prefix_cache") is not None:
        raise ValueError(
            "baseline has the automatic cache enabled; use --disable-radix-cache"
        )
    return {
        "input_tokens": body["usage"]["input_tokens"],
        "answers": body["answers"],
    }, elapsed


def record_baseline(args):
    path = args.save_baseline
    if path.exists():
        raise ValueError(f"baseline file exists: {path}; choose a new filename")

    nonce = secrets.token_hex(8)
    cases = []
    for number, payload in enumerate(sample_requests(args.model, nonce), start=1):
        expected, elapsed = reference_result(args.url, payload, args.timeout)
        print(
            f"{number}  input={expected['input_tokens']}  reference_e2e_ms={elapsed:.1f}"
        )
        cases.append({"request": payload, **expected})

    # Publish only a complete set of three replies, and never replace a concurrent export.
    with NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8") as tmp:
        json.dump({"version": 1, "nonce": nonce, "cases": cases}, tmp)
        tmp.write("\n")
        tmp.flush()
        os.link(tmp.name, path)
    print(f"Saved three synthetic requests and complete reference answers to {path}")


def load_baseline(path, model):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("unsupported baseline file format")
    nonce = data["nonce"]
    if not isinstance(nonce, str) or not re.fullmatch(r"[a-f0-9]{16}", nonce):
        raise ValueError("invalid baseline probe identifier")
    requests = sample_requests(model, nonce)
    cases = data["cases"]
    if not isinstance(cases, list) or len(cases) != len(requests):
        raise ValueError("baseline must contain all three synthetic requests")
    # Validate the model, policy, states and questions before sending any saved data.
    for expected, payload in zip(cases, requests):
        if (
            expected["request"] != payload
            or type(expected["input_tokens"]) is not int
            or expected["input_tokens"] < 1
            or set(expected["answers"]) != set(QUESTIONS)
        ):
            raise ValueError(
                "baseline differs from the synthetic fixture or requested model"
            )
    return cases


def run(args):
    if args.save_baseline:
        record_baseline(args)
        return
    if args.baseline_file:
        cases = load_baseline(args.baseline_file, args.model)
        requests = [case["request"] for case in cases]
    else:
        cases = None
        requests = sample_requests(args.model, secrets.token_hex(8))

    samples = []
    for index, payload in enumerate(requests):
        body, elapsed = post(args.url, payload, args.timeout)
        total, cached, computed = restored_work(body)
        samples.append((total, cached, computed))
        number = index + 1
        print(
            f"{number}  input={total}  restored={cached}  computed={computed}  e2e_ms={elapsed:.1f}"
        )
        expected = cases[index] if cases else None
        if args.baseline_url:
            expected, reference_ms = reference_result(
                args.baseline_url, payload, args.timeout
            )
            print(f"   reference_e2e_ms={reference_ms:.1f}")
        if expected is not None:
            if (
                expected["input_tokens"] != total
                or expected["answers"] != body["answers"]
            ):
                raise ValueError(
                    f"request {number}: input token count or complete answers differ"
                )
            print("   complete_answers=equal")

    first_total, first_cached, first_computed = samples[0]
    last_total, last_cached, last_computed = samples[-1]
    # The synthetic policy dominates this specific workload. A small shared API
    # template can be resident even on the first request and is not a policy hit.
    if 2 * first_cached >= first_total:
        raise ValueError(
            "first request already had a material hit; cold discovery is inconclusive"
        )
    if 2 * last_cached < last_total or last_computed >= first_computed:
        raise ValueError(
            "third request did not materially restore the policy and reduce computation"
        )
    print("PASS: material cross-request prefix restoration and fewer computed tokens.")
    if args.baseline_url or args.baseline_file:
        print("PASS: all three complete answers match the baseline on each request.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:38471",
        help="server base URL (full-compute baseline when recording)",
    )
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument(
        "--baseline-url", help="compare against a second, live full-compute server"
    )
    reference.add_argument(
        "--save-baseline",
        type=Path,
        metavar="FILE",
        help="record a synthetic fixture from the uncached --url in a new file",
    )
    reference.add_argument(
        "--baseline-file",
        type=Path,
        metavar="FILE",
        help="replay and compare the saved fixture against the auto-cache --url",
    )
    parser.add_argument("--model", default="guard_base")
    parser.add_argument(
        "--timeout", type=int, default=120, help="socket timeout in seconds per request"
    )
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    try:
        run(args)
    except (ValueError, KeyError, TypeError, OSError, URLError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
