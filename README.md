# C13: SpectrumLens

Signals, Wireless and Control — signal-window spectrum occupancy detection, built on the [Cordis](https://github.com/cordiverse/cordis) plugin framework (TypeScript, Node ESM).

```bash
npm install
npx tsx src/index.ts   # or: npm start — runs the Gen-0 baseline and the Phase-2 evolution loop
npm run typecheck
```

## Phase 1 — Generation 0 (First Demo)

### Architecture

| Plugin | File | Registers as | Purpose |
|---|---|---|---|
| `DspFixtureService` | `src/plugins/dsp-fixtures.ts` | `ctx.dsp` | Radix-2 complex FFT, tapers (rectangular / Hann / Hamming / Blackman), Welch PSD (average or max-hold), sub-band aggregation (mean + peak bin); seeded synthetic I/Q fixtures (tones + pulsed bursts) at −10/−5/0/+5/+10 dB SNR, split into `dev` and held-out `test`; deterministic Pd/Pfa evaluator. |
| `SignalWindowDetector` | `src/plugins/window-detector.ts` | `ctx.detector` (`inject = ['dsp']`) | The signal window as an editable unit (length, sub-window slicing, taper, noise-floor method, threshold rule). Gen-0 = Baseline Energy Detector: rectangular taper, single periodogram, fixed absolute threshold, no noise-floor adaptation. Emits `occupiedBands` plus per-band evidence (center Hz, PSD dB/Hz, noise floor, SNR, margin, confidence). |

### Fixture conventions

- Complex AWGN, nominal noise PSD −60 dB/Hz at 1 MS/s, with per-window noise-floor drift (uniform ±3 dB) to model receiver uncertainty.
- SNR is **in-band**: emitter power relative to the true noise inside one sub-band (16 sub-bands of 62.5 kHz), measured while the emitter is keyed on. Bursts have 20–50 % duty cycle, so their average SNR is lower.
- Every window draws from a sub-seed hashed from `(seed, split, snr, index)`, so runs are bit-for-bit reproducible.
- Scoring is per (window, sub-band) cell: Pd = TP / occupied cells, Pfa = FP / vacant cells.

## Phase 2 — Self-Evolution

`SelfEvolver` (`src/plugins/self-evolver.ts`, registers as `ctx.evolver`, `inject = ['dsp']`) evolves the editable signal window over **Gen 0 → Gen 3**:

1. **Evaluate & trace.** The active `ctx.detector` is run on the `dev` SNR slices (−10 … +10 dB). Each error becomes a structured `FailureTrace`: a missed detection (emitter kind, duty cycle) or a false alarm (empty window or next to an occupied band). Each trace carries its residual margin, its noise-floor error and the band's peak-to-mean ratio.
2. **Diagnose & propose** an upgraded `SignalWindowConfig`:
   - **Gemini mode** (`GEMINI_API_KEY` set): the traces go to `gemini-2.5-flash` over the REST API (native `fetch`, JSON response mode). The returned config is sanitized and clamped. Gemini may also ask for an exact threshold calibration. Any API or parse failure falls back to the heuristic optimizer for that generation. Set `GEMINI_MODEL` to use another model.
   - **Heuristic mode** (no key, zero cost, deterministic): each failure mode is scored by the number of dev errors it accounts for:
     - noise-floor mismatch → median/percentile floor with a relative (CFAR) threshold
     - narrowband dilution → peak-bin statistic
     - statistic variance → multi-window integration
     - spectral leakage → Hann taper

     The highest-scoring mode is fixed. Then the detection margin is set by an exact sweep over the miss and false-alarm residuals, aiming at a 5 % design Pfa that leaves headroom below the gate.
3. **Hot-swap & regression gate.** `await fiber.dispose()` removes the incumbent detector. The candidate `SignalWindowDetector` is mounted onto the context and re-verified on `dev`. It is kept only if `Pd − 1.5·Pfa` improves **and** `Pfa ≤ 0.10`. Otherwise the candidate is disposed and the previous generation's plugin is re-mounted (rollback).
4. Generation snapshots, diagnoses, rationales, gate decisions, hot-swap logs and the final dev/test metrics are written to `artifacts/evolution_history.json`.

Implementation note: Cordis service proxies bind `this.ctx` to the *caller's* context. The evolver therefore mounts detectors onto its own pinned plugin context. That ties the evolved detector's lifetime to the evolver plugin, not to the `inject` scope that called `evolve()`.

### Result (heuristic mode, seed `0xc13`)

| Gen | Signal window | U (dev) | Pd −10 dB | Pd −5 dB | Pfa | Gate on next candidate |
|---|---|---|---|---|---|---|
| 0 | Baseline energy detector (rect, band mean, assumed floor, absolute threshold) | 0.452 | 3.8 % | 6.0 % | 1.4 % | accept peak-bin statistic |
| 1 | + peak-bin statistic | 0.784 | 39.2 % | 62.7 % | 1.4 % | accept median noise floor + CFAR margin |
| 2 | + median noise floor, relative margin 9.83 dB | 0.848 | 50.6 % | 80.6 % | 1.5 % | accept 4× Welch integration |
| 3 | + 4 sub-windows averaged, margin 5.88 dB (survivor) | 0.858 | 48.1 % | 80.6 % | 0.7 % | — |

Held-out test, Gen 0 → Gen 3: Pd at −10 dB 7.6 % → 46.8 %, at −5 dB 13.6 % → 74.1 %; overall Pd 43.5 % → 85.7 %, Pfa 2.0 % → 0.8 %.

The heuristic optimizer kept the rectangular taper on purpose. Only about 19 % of the remaining false alarms sit next to occupied bands, so leakage is not a failure mode on this data. A Hann taper would cost about 1.8 dB of equivalent-noise-bandwidth on tone peaks.
