"""Training-data centroids for ALL THREE activation reductions (sec3 fix).

The authors' sec5/compute_training_centroid.py computes only the response-mean
centroid. Their per-extraction-method analyses (Table 4, appendix B.1) each use
a centroid built with the SAME reduction as the eval activations:
last-prompt-token distances are measured against a last-prompt-token centroid,
prompt-mean against a prompt-mean centroid. Comparing e.g. last_prompt_tok
activations to a response_mean centroid is a reference mismatch — that bug
invalidated the non-headline rows of our run-1 results.

Same forward pass as the authors' script (base model over prompt+completion of
each training example); we just accumulate three running sums instead of one.

Usage (GPU):
  python scripts/compute_training_centroids_all.py \
      --base_model Qwen/Qwen2.5-7B-Instruct \
      --train_jsonl bad_medical_advice.jsonl \
      --out_dir centroids --tag bad_medical_Qwen2.5-7B-Instruct \
      --max_examples 5000 --device_map cuda:0

Writes <out_dir>/<tag>_{response_mean,prompt_mean,last_prompt_tok}.npy,
each [n_layers+1, hidden].
"""
import argparse
import json
import os

import numpy as np
import torch


def _split_prompt_completion(messages):
    last_asst = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    return messages[:last_asst], messages[last_asst]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--max_seq_len", type=int, default=1280)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device_map", default="auto")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in open(args.train_jsonl) if l.strip()]
    if args.max_examples:
        rows = rows[:args.max_examples]
    print(f"[centroids] {len(rows)} training examples from {args.train_jsonl}")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map=args.device_map, trust_remote_code=True)
    model.eval()
    dev = next(model.parameters()).device

    items = []
    for r in rows:
        prompt_msgs, completion = _split_prompt_completion(r["messages"])
        ps = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        pids = tok(ps, add_special_tokens=False)["input_ids"]
        fids = tok(ps + completion, add_special_tokens=False)["input_ids"]
        if fids[:len(pids)] != pids:
            fids = pids + tok(completion, add_special_tokens=False)["input_ids"]
        items.append((fids[:args.max_seq_len], len(pids)))
    items.sort(key=lambda x: len(x[0]))

    sums = {"response_mean": None, "prompt_mean": None, "last_prompt_tok": None}
    n_used = 0
    for j in range(0, len(items), args.batch_size):
        batch = items[j:j + args.batch_size]
        maxlen = max(len(fid) for fid, _ in batch)
        ids = torch.full((len(batch), maxlen), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for k, (fid, _n) in enumerate(batch):
            ids[k, :len(fid)] = torch.tensor(fid)
            mask[k, :len(fid)] = 1
        with torch.no_grad():
            out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                        use_cache=False, output_hidden_states=True)
        hs = torch.stack([h.detach().float().cpu() for h in out.hidden_states], dim=0)  # [L+1,B,S,H]
        del out
        for k, (fid, n_p) in enumerate(batch):
            end = len(fid)
            if end <= n_p:
                continue
            pm = hs[:, k, :n_p, :].mean(dim=1).numpy()          # [L+1, H]
            rm = hs[:, k, n_p:end, :].mean(dim=1).numpy()
            lt = hs[:, k, n_p - 1, :].numpy()
            for key, arr in (("prompt_mean", pm), ("response_mean", rm), ("last_prompt_tok", lt)):
                sums[key] = arr if sums[key] is None else sums[key] + arr
            n_used += 1
        if (j // args.batch_size) % 20 == 0:
            print(f"[centroids] {j + len(batch)}/{len(items)}  used={n_used}")
        del hs
        torch.cuda.empty_cache()

    os.makedirs(args.out_dir, exist_ok=True)
    for key, s in sums.items():
        c = (s / n_used).astype(np.float32)
        path = os.path.join(args.out_dir, f"{args.tag}_{key}.npy")
        np.save(path, c)
        print(f"[centroids] saved {path} shape={c.shape} (averaged {n_used} examples)")


if __name__ == "__main__":
    main()
