# sales-copilot

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
<!-- RESULTS:END -->

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
