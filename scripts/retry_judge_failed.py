"""Retry judge rows that failed (ERROR / empty content) in a judge_*.json.

Judge API failures were previously recorded as score=None rows and silently
dropped alongside genuine REFUSAL verdicts by the metric script. This script
finds those rows (an error flag, raw starting with ERROR, or empty raw) in an
existing judge output, re-judges ONLY them with the same rubric via the
patched sec3/judge_responses.judge_one (which retries empty content and 429s),
and updates the file in place (a .bak copy is written first).

Genuine REFUSAL / NOT_APPLICABLE verdicts are left untouched.

Usage (same env as the judge run — source em-not-magical/.env, override
JUDGE_MODEL for the judge that produced the file):
  python scripts/retry_judge_failed.py \
      --judge data/chatbot_verified/qwen2.5-7b-em-medical-advice-vllm-hybrid/judge_glm-5p3_chatbot_verified.json \
      --dataset chatbot_verified
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "em-not-magical" / "sec3"))
from judge_responses import JUDGE_PROMPTS, JUDGE_MODEL, judge_one  # noqa: E402


def is_failed(r):
    """True for judge-API failures (retryable); False for refusals/NA/scores."""
    if r.get("score") is not None:
        return False
    if r.get("error"):
        return True
    raw = (r.get("raw_judgment") or r.get("raw") or "")
    if not raw.strip():
        return True
    return raw.startswith("ERROR")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, help="judge_*.json to repair")
    ap.add_argument("--dataset", required=True,
                    help="dataset key for the rubric (e.g. chatbot_verified)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    recs = json.load(open(args.judge))
    failed = [r for r in recs if is_failed(r)]
    print(f"[retry] {args.judge}: {len(recs)} rows, {len(failed)} failed "
          f"(idx {[r['idx'] for r in failed]}); judge model {JUDGE_MODEL}")
    if not failed:
        return

    api_key = os.environ["OPENAI_API_KEY"]
    template = JUDGE_PROMPTS[args.dataset]
    client = httpx.Client(timeout=180)
    still = {}

    def run(r):
        return r, judge_one(client, api_key, template, r["prompt"], r["response"], "evil")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run, r) for r in failed]
        for f in as_completed(futs):
            r, j = f.result()
            raw = (j["raw"] or "").strip()
            if not raw:
                still[r["idx"]] = "empty judge content after retries"
                continue
            r["raw_judgment"] = raw
            r["score"] = j["score"]
            r["judge_model"] = JUDGE_MODEL
            r.pop("error", None)
            print(f"[retry] idx {r['idx']}: score={r['score']} raw={raw[:60]!r}")

    bak = args.judge + ".bak"
    if os.path.exists(bak):
        os.remove(bak)
    os.replace(args.judge, bak)
    with open(args.judge, "w") as f:
        json.dump(recs, f, indent=2)
    n_err = sum(1 for r in recs if is_failed(r))
    print(f"[retry] updated {args.judge} (backup at {bak}); "
          f"{len(failed) - len(still)}/{len(failed)} recovered, {n_err} still failing")
    if still:
        print(f"[retry] STILL FAILING: {still}")
        sys.exit(1)


if __name__ == "__main__":
    main()
