"""ASR accuracy + speed on a fixed-seed LibriSpeech test-clean sample -> results/asr_wer.json.

Downloads one parquet shard (openslr/librispeech_asr, clean/test/0000.parquet, ~350 MB,
no auth) into data/cache/. Each model decodes each utterance as one whole segment on CPU
(int8, greedy, 4 threads) - the same settings the live pipeline uses per utterance.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from sales_copilot.asr import FasterWhisperTranscriber
from sales_copilot.audio import resample_linear
from sales_copilot.metrics import bootstrap_wer, corpus_wer

ROOT = Path(__file__).resolve().parents[1]
REPO, SHARD = "openslr/librispeech_asr", "clean/test/0000.parquet"


def load_sample(n: int, seed: int) -> list[dict[str, object]]:
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, SHARD, repo_type="dataset", cache_dir=str(ROOT / "data/cache"))
    table = pq.read_table(path, columns=["id", "text", "audio"])
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(table.num_rows, size=n, replace=False).tolist())
    out = []
    for i in idx:
        row = table.slice(i, 1).to_pylist()[0]
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        out.append({"id": row["id"], "text": row["text"], "audio": resample_linear(audio, sr)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tiny", "base"])
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results/asr_wer.json"))
    args = ap.parse_args()

    sample = load_sample(args.n, args.seed)
    audio_s = float(sum(len(s["audio"]) for s in sample) / 16_000)  # type: ignore[arg-type,misc]
    refs = [str(s["text"]) for s in sample]
    runs = []
    for size in args.models:
        load0 = os.getloadavg()[0]
        t0 = time.perf_counter()
        asr = FasterWhisperTranscriber(size, cpu_threads=args.threads)
        load_s = time.perf_counter() - t0
        asr.transcribe(sample[0]["audio"])  # type: ignore[arg-type]  # warm-up, not timed
        hyps, times = [], []
        for s in sample:
            t = time.perf_counter()
            hyps.append(asr.transcribe(s["audio"]))  # type: ignore[arg-type]
            times.append(time.perf_counter() - t)
        wer, errs, words = corpus_wer(refs, hyps)
        durs = np.array([len(s["audio"]) / 16_000 for s in sample])  # type: ignore[arg-type]
        per_rtf = np.array(times) / durs
        run = {
            "model": size,
            "backend": asr.name,
            "beam_size": 1,
            "cpu_threads": args.threads,
            "wer": round(wer, 4),
            "wer_ci95": bootstrap_wer(refs, hyps, n_boot=2000, seed=0),
            "errors": errs,
            "ref_words": words,
            "rtf": round(sum(times) / audio_s, 4),
            "rtf_p50": round(float(np.median(per_rtf)), 4),
            "rtf_p95": round(float(np.quantile(per_rtf, 0.95)), 4),
            "mean_decode_s": round(float(np.mean(times)), 4),
            "model_load_s": round(load_s, 2),
            "loadavg_1m_start": round(load0, 2),
            "loadavg_1m_end": round(os.getloadavg()[0], 2),
        }
        runs.append(run)
        print(json.dumps(run))
        hyp_path = ROOT / f"results/asr_hyps_{size}.jsonl"
        with hyp_path.open("w") as fh:
            for s, h in zip(sample, hyps, strict=True):
                fh.write(json.dumps({"id": s["id"], "ref": s["text"], "hyp": h}) + "\n")
        del asr
    out = {
        "dataset": f"{REPO} ({SHARD})",
        "n_utterances": len(sample),
        "audio_s": round(audio_s, 1),
        "sample_seed": args.seed,
        "normalizer": "lower-case, digits spelled out, punctuation stripped (see metrics.py)",
        "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} cores, shared",
        "runs": runs,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
