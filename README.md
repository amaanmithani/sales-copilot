# sales-copilot

> **Credits.** Built by Amaan Mithani with Claude (Anthropic) as the AI coding assistant.

A live sales-call copilot that runs on a laptop CPU. It transcribes the call as it streams in
and shows cue cards during the call: price, timing, authority and status-quo objections,
competitor mentions, pricing questions and agreed next steps. Each card carries a suggested
response from a YAML playbook. The pipeline measures the time from the end of speech to the
cue appearing, and the README reports it.

```
 audio (WAV replay paced to real time, or mic)
   │ 100 ms chunks
   ▼
 Endpointer (energy VAD, 450 ms trailing silence, 12 s max)   ── sales_copilot/vad.py
   │ finished utterance
   ▼
 faster-whisper tiny|base, int8, greedy, CPU                   ── sales_copilot/asr.py
   │ text
   ▼
 CueEngine: classifier → threshold → cooldown / de-dup         ── sales_copilot/engine.py
   │   classifiers: rules (regex) | lr+tfidf | lr+all-MiniLM-L6-v2   ── sales_copilot/cues/
   │   responses + competitor battlecards from playbook.yaml
   │   optional LLM rewrite (OFF by default, env-gated)        ── sales_copilot/llm.py
   ▼
 Rich terminal UI (transcript | cue cards) + JSONL event log   ── sales_copilot/tui.py
```

`sales_copilot/pipeline.py` does the latency accounting. In real-time replay, each chunk is
released once its last sample would have been captured. If ASR falls behind, chunks queue up
the way a capture buffer would, and that backlog counts toward latency.

## Quick start

```bash
uv sync --all-extras                      # core + faster-whisper + sentence-transformers + bench deps
uv run sales-copilot replay data/audio/call_a.wav --asr tiny          # live TUI, real-time replay
uv run sales-copilot replay data/audio/call_b.wav --asr base --classifier rules --no-tui
uv run sales-copilot classify "that's way over our budget" "are you cheaper than gong"
uv run --extra mic sales-copilot mic --asr tiny                       # microphone (untested in CI)
```

`data/audio/call_*.wav` contain **synthetic** audio. `scripts/make_demo_call.py` produced them
with macOS `say` (two TTS voices per call, a fixed script, and -60 dBFS of added hiss). The
scripts and their gold labels are in `data/demo/`.

Optional LLM rewrite of the suggested response (off by default; tests and evaluations never
use it):

```bash
SALES_COPILOT_LLM_REWRITE=1 SALES_COPILOT_LLM_BASE_URL=http://localhost:8080/v1 \
SALES_COPILOT_LLM_MODEL=<model> uv run sales-copilot replay data/audio/call_a.wav
```

Any OpenAI-compatible `/chat/completions` endpoint works, for example a local ModelMux gateway.
If the call fails or takes longer than 3 s, the card shows the playbook text instead.

## Reproduce the results

```bash
uv run python scripts/eval_cues.py        # -> results/cue_eval.json   (uses cached all-MiniLM-L6-v2)
uv run python scripts/bench_asr.py        # -> results/asr_wer.json    (downloads ~350 MB LibriSpeech shard + models)
uv run python scripts/make_demo_call.py   # regenerate synthetic calls (macOS only)
uv run python scripts/bench_latency.py    # -> results/latency.json    (real-time replay, ~5 min)
uv run python scripts/render_readme.py    # rewrite the tables below from results/*.json
```

Quality gates: `uv run ruff check src scripts tests`, `uv run ruff format --check src scripts tests`,
`uv run mypy` (strict), `uv run pytest --cov` (coverage floor 75%). CI (`.github/workflows/ci.yml`)
installs only the core and dev dependencies. It downloads no models, because the tests stub the ASR
and the embedder. CI also checks that the README tables match the committed JSON.

## Results

Every number below comes from `scripts/render_readme.py`, which reads the committed
`results/*.json`. Nothing in the tables was typed by hand.

<!-- RESULTS:START -->
**Cue classification** (held-out test set, n = 172 utterances; trained on n = 252; exact-text overlap train/test = 0; percentile, 2000 resamples of test utterances, seed 0, 95% CI).

| System | Macro-F1 [95% CI] | Micro-F1 [95% CI] | ms / utterance (batched) |
|---|---|---|---|
| rules | 0.539 [0.454, 0.608] | 0.584 [0.500, 0.660] | 0.394 |
| lr+tfidf | 0.693 [0.619, 0.751] | 0.705 [0.638, 0.768] | 1.027 |
| lr+all-MiniLM-L6-v2 | 0.739 [0.676, 0.789] | 0.740 [0.685, 0.794] | 4.025 |

Per cue type, F1 [95% CI] (support in brackets):

| Cue (support) | rules | lr+tfidf | lr+all-MiniLM-L6-v2 |
|---|---|---|---|
| objection_price (20) | 0.462 [0.182, 0.688]<br>P 1.00 / R 0.30 | 0.471 [0.240, 0.667]<br>P 0.57 / R 0.40 | 0.649 [0.444, 0.811]<br>P 0.71 / R 0.60 |
| objection_timing (18) | 0.467 [0.222, 0.667]<br>P 0.58 / R 0.39 | 0.581 [0.343, 0.765]<br>P 0.69 / R 0.50 | 0.629 [0.412, 0.789]<br>P 0.65 / R 0.61 |
| objection_authority (18) | 0.812 [0.632, 0.944]<br>P 0.93 / R 0.72 | 0.750 [0.545, 0.898]<br>P 0.86 / R 0.67 | 0.706 [0.485, 0.857]<br>P 0.75 / R 0.67 |
| objection_status_quo (19) | 0.273 [0.000, 0.516]<br>P 1.00 / R 0.16 | 0.452 [0.214, 0.649]<br>P 0.58 / R 0.37 | 0.634 [0.435, 0.784]<br>P 0.59 / R 0.68 |
| competitor_mention (20) | 1.000 [1.000, 1.000]<br>P 1.00 / R 1.00 | 0.947 [0.857, 1.000]<br>P 1.00 / R 0.90 | 0.950 [0.870, 1.000]<br>P 0.95 / R 0.95 |
| pricing_question (20) | 0.333 [0.087, 0.560]<br>P 1.00 / R 0.20 | 0.833 [0.667, 0.947]<br>P 0.94 / R 0.75 | 0.927 [0.815, 1.000]<br>P 0.90 / R 0.95 |
| next_step (20) | 0.429 [0.174, 0.645]<br>P 0.75 / R 0.30 | 0.821 [0.667, 0.938]<br>P 0.84 / R 0.80 | 0.679 [0.511, 0.809]<br>P 0.55 / R 0.90 |

Paired bootstrap, macro-F1 difference vs the rules baseline:

| Comparison | Δ macro-F1 [95% CI] | share of resamples where rules >= |
|---|---|---|
| lr+tfidf minus rules | +0.154 [+0.061, +0.249] | 0.001 |
| lr+all-MiniLM-L6-v2 minus rules | +0.200 [+0.109, +0.291] | 0.0 |

**ASR accuracy and speed** (openslr/librispeech_asr (clean/test/0000.parquet), n = 200 utterances = 24.0 min, sample seed 0; Darwin arm64, 8 cores, shared).

| Model | WER [95% CI] | Errors / ref words | RTF (total) | RTF p50 / p95 per utt | load avg (1 min) start→end |
|---|---|---|---|---|---|
| faster-whisper-tiny-int8 (beam 1, 4 threads) | 8.3% [6.8, 9.8] | 320 / 3872 | 0.082 | 0.088 / 0.222 | 14.88 → 11.8 |
| faster-whisper-base-int8 (beam 1, 4 threads) | 5.8% [4.6, 7.1] | 224 / 3872 | 0.147 | 0.164 / 0.362 | 11.8 → 12.54 |

**Streaming end-to-end** (SYNTHETIC: macOS `say` TTS, two voices per call, -60 dBFS hiss (see scripts); chunks 100 ms; VAD energy, -42 dBFS, 450 ms trailing silence). Latency = wall time from capture of the last voiced sample to cue ready (endpoint wait + queueing + ASR + classify); excludes UI paint.

| ASR | Audio | Streaming RTF | Speech-end→cue p50 / p90 / max (cue segments) | All segments p50 / p90 | Mean breakdown: endpoint / ASR / classify | Call WER | load avg start→end |
|---|---|---|---|---|---|---|---|
| faster-whisper-tiny-int8 | 158.4 s, 35 segments | 0.091 | 753 / 884 / 2460 ms (n=18) | 768 / 1330 ms | 450 / 414 / 35 ms | 6.3% | 3.07 → 6.01 |
| faster-whisper-base-int8 | 158.4 s, 35 segments | 0.076 | 873 / 921 / 943 ms (n=16) | 873 / 930 ms | 450 / 343 / 22 ms | 4.8% | 6.01 → 2.68 |

Turn-level cue detection on the synthetic calls (same classifier: lr+all-MiniLM-L6-v2), ASR transcript vs the gold script text:

| ASR | On ASR text: P / R / F1 (tp, fp, fn) | On gold text: P / R / F1 (tp, fp, fn) |
|---|---|---|
| faster-whisper-tiny-int8 | 0.55 / 0.94 / 0.70 (16, 13, 1) | 0.54 / 0.88 / 0.67 (15, 13, 2) |
| faster-whisper-base-int8 | 0.54 / 0.82 / 0.65 (14, 12, 3) | 0.54 / 0.88 / 0.67 (15, 13, 2) |
<!-- RESULTS:END -->

### Reading the results

- **Learned beats rules.** Both learned classifiers beat the regex baseline on held-out macro-F1,
  and the paired-bootstrap CIs exclude zero. Most of the gap comes from rules breaking on
  unpunctuated, informal test phrasing: pricing-question recall is 0.20 and status-quo recall is
  0.16. Rules still win on closed-list competitor mentions and do well on authority objections.
  A hybrid (rules for competitors, MiniLM for everything else) is the obvious next step. I did
  not evaluate it here because I would have picked it after seeing the test scores.
- **MiniLM vs TF-IDF is not settled.** MiniLM leads TF-IDF by about 0.05 macro-F1, but their
  CIs overlap. MiniLM over-fires `next_step` (P 0.55).
- **Latency is about 0.75–0.9 s from the end of speech to the cue.** About 450 ms of that is the
  deliberate silence endpoint. ASR is about 0.35–0.4 s per utterance, because Whisper always
  encodes a 30 s window, so short utterances do not decode proportionally faster. Classification
  is 20–35 ms.
- **The streaming run does not rank tiny against base.** It is one run over 35 segments on a
  shared machine. The tiny pass ran first and hit one warm-up-like 2.5–3 s outlier, and its mean
  ASR time came out above base's. For speed, use the offline RTF (n = 200): tiny is about
  1.8× faster than base.
- **ASR errors barely hurt cues on the synthetic calls.** Turn-level cue F1 is about the same on
  ASR text and on gold text.
- **Most false positives come from the rep's own turns.** Gold-text false positives: 7 of 11 turns
  are rep turns. The rep's questions ("what happens after a call ends?") read as
  `next_step`/`pricing_question`. Diarization, or cueing only on the prospect channel, would
  remove most of them.

## Method

**Cue labels.** There are seven cue types and the task is multi-label: `objection_price`,
`objection_timing`, `objection_authority`, `objection_status_quo`, `competitor_mention`,
`pricing_question` and `next_step`. An utterance can carry none of them or several.

**Data.** I wrote all the labelled utterances myself; none were sourced. Train and test are
separate files written separately, and they share no text exactly.
- `data/cues/train.tsv` holds 252 utterances. They are written like tidy transcripts, with
  punctuation and capitals.
- `data/cues/test.tsv` holds 172 utterances. On purpose they use a different register, closer to
  raw ASR output: lower case, no punctuation, fillers ("um", "like"), indirect phrasing and hard
  negatives such as "coffee prices ... are insane" or "my boss just got back from leave".
- The rules were written against the train file only. The test file was read once, to score.

**Classifiers.**
- `rules` uses regexes plus the playbook's competitor aliases.
- `lr+tfidf` is word 1–2-gram plus char 3–5-gram TF-IDF with one-vs-rest logistic regression.
- `lr+all-MiniLM-L6-v2` feeds 384-d normalised sentence embeddings (CPU) into the same logistic
  regression.
- For the learned models, C is picked per label by stratified 5-fold CV on train (F1), with
  `class_weight=balanced` and a 0.5 threshold. Nothing is tuned on test.
- Confidence intervals are percentile bootstraps over test utterances (2000 resamples, seed 0).
  The comparison against the rules baseline is a paired bootstrap that uses the same resampled
  rows for both systems.

**ASR.** I took a fixed-seed sample (seed 0) of 200 utterances from LibriSpeech test-clean (one
parquet shard of `openslr/librispeech_asr`). Each utterance is decoded as a single segment with
the same settings the live path uses: int8, greedy, 4 CPU threads, no internal VAD. WER is the
corpus WER after light normalisation (lower-case, digits spelled out, punctuation stripped). Its
95% CI comes from bootstrapping over utterances. RTF is ASR compute time divided by audio
duration, measured after one untimed warm-up.

**Latency.** Latency runs from the capture of the last voiced sample to the moment the cue is
ready to render. It includes:
- endpoint wait: 450 ms of trailing silence, plus up to 100 ms of chunk quantisation
- any queueing
- ASR on the whole utterance
- classification

It excludes terminal paint time. The synthetic calls are replayed in real time, one heavy
process at a time.

## Limits

- **Single author, synthetic phrasing.** The labelled set is small and one person wrote it.
  Train and test differ in register but not in author. There are no real call transcripts and
  no second annotator, so there is no inter-annotator agreement either. The F1 numbers show how
  far the classifiers generalise across my own phrasing styles. Real calls will be messier, and
  performance on them is probably lower.
- **Synthetic demo audio.** The demo calls are clean TTS on one mono channel, with no crosstalk,
  no diarization and no telephone codec. The call WER and latency figures there are best-case
  for the acoustics. LibriSpeech is read audiobook speech, not conversational sales speech.
- **Shared, loaded machine.** All benchmarks ran on a shared Apple-silicon laptop (8 cores,
  16 GB) while other jobs were running. Every result JSON records the 1-minute load average,
  which ran well above the core count. Treat the RTF and latency figures as pessimistic and
  noisy. Each is a single run, not a distribution over repeated runs.
- **Endpoint-then-decode.** The design never shows partial hypotheses, so a cue cannot appear
  before the speaker pauses. A 450 ms silence endpoint is the floor. Long monologues are cut at
  12 s.
- **Energy VAD.** It is fine for clean audio but will misfire on noisy lines. faster-whisper's
  Silero VAD or WebRTC VAD would be the next step.
- **Competitor detection is a closed list.** Common words such as "outreach" ("our outreach
  team") produce false positives in the rules path.
- **Rules depend on punctuation.** Several rules key on a `?`, which ASR output and the test
  set often lack. This is the main reason the rules' pricing-question recall is low.
