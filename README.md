# C13: SpectrumLens

**Signals, Wireless and Control: explainable spectrum-occupancy detection on recorded I/Q, with a self-evolving signal window and a learned-detector comparison.**

SpectrumLens takes a window of complex baseband samples. It decides which of its 16 sub-bands are occupied and explains each decision with quantitative spectral evidence: the measured PSD statistic, the noise floor it was compared with, the SNR, the margin to the threshold and a confidence. The detector is built around one explicit, editable unit, the **signal window** (`SignalWindowConfig`). That unit is first evolved by a failure-driven loop and then benchmarked against a learned multi-feature detector on a held-out split. Everything is written as plugins on the [Cordis](https://github.com/cordiverse/cordis) v4 framework (TypeScript, Node ESM).

### Mapping to Wireless EMSO signal explanation

Electromagnetic Spectrum Operations (EMSO) depend on *spectrum awareness*: knowing which channels are in use, by what kind of emitter, and how sure you are. SpectrumLens covers that sensing-and-explanation step:

| EMSO need | SpectrumLens mechanism |
|---|---|
| Channel occupancy sensing | Per-sub-band binary decision over a 1 MHz span (16 × 62.5 kHz) |
| Robustness to an uncalibrated receiver | Median noise-floor estimate per window with a CFAR-style relative threshold, evolved from a fixed-floor baseline that fails under ±3 dB drift |
| Weak and intermittent emitters | Peak-bin narrowband statistic, Welch integration, and temporal-persistence features for pulsed bursts |
| Operator-auditable decisions | Evidence per band (PSD in dB/Hz, floor, SNR, margin, confidence), feature contributions to the learned logit, and a narrative per window ([`artifacts/sample_evidence.md`](artifacts/sample_evidence.md)) |
| Controlled adaptation | Self-evolution behind a regression gate (utility must improve, Pfa ≤ 10 %) with automatic rollback, all logged in [`artifacts/evolution_history.json`](artifacts/evolution_history.json) |

## Reproduce

Requires Node ≥ 20.

```bash
npm install
```

```bash
npm run typecheck
```

```bash
npm start
```

`npm start` (equivalent to `npx tsx src/index.ts`) runs all three phases in about 4 s. With no API key the run is **deterministic**: the self-evolution uses the built-in failure-driven optimizer, and every number in this README is reproduced exactly. It writes three files to `artifacts/` (only the `generatedAt` timestamps change between runs):
- `evolution_history.json`
- `benchmark_report.json`
- `sample_evidence.md`

**Optional: Gemini-driven evolution.** The self-evolution proposals then come from `gemini-2.5-flash` over the REST API (native `fetch`, no SDK):

```bash
GEMINI_API_KEY=your-key npm start
```

Set `GEMINI_MODEL` to use another model. Every Gemini proposal is sanitized and passes the same regression gate. If the API call fails or returns unparsable output, that generation falls back to the deterministic optimizer. LLM proposals are not deterministic, so the Self-Evolved numbers may differ from the table below in this mode. The baseline and learned detectors do not depend on the mode.

## Architecture

```
┌──────────────────────────────────── Cordis Context (root) ─────────────────────────────────────┐
│                                                                                                │
│  ┌───────────────────────────┐   provides ctx.dsp    ┌──────────────────────────────────────┐  │
│  │ DspFixtureService         │──────────────────────▶│ (injected via static inject=['dsp']) │  │
│  │  src/plugins/dsp-fixtures │                       └──────────────────────────────────────┘  │
│  │  • radix-2 FFT, tapers    │          │                     │                     │          │
│  │  • Welch PSD (avg / max)  │          ▼                     ▼                     ▼          │
│  │  • sub-band mean / peak   │  ┌──────────────────┐ ┌────────────────────┐ ┌─────────────────┐ │
│  │  • seeded I/Q fixtures    │  │SignalWindow-     │ │ SelfEvolver        │ │LearnedSpectral- │ │
│  │    dev | held-out test    │  │Detector          │ │  ctx.evolver       │ │Detector         │ │
│  │  • Pd / Pfa evaluator     │  │ ctx.detector     │ │  inject ['dsp']    │ │ ctx.learned     │ │
│  └───────────────────────────┘  │                  │ │  (survives every   │ │ 9 features/band │ │
│                                 │ ┌──────────────┐ │ │   detector swap)   │ │ L2 logistic reg.│ │
│                                 │ │SIGNAL WINDOW │ │ └─────────┬──────────┘ │ fit on dev only │ │
│                                 │ │(editable unit│ │           │            └─────────────────┘ │
│                                 │ │ taper        │ │           │ owns the detector fiber          │
│                                 │ │ subWindows   │ │◀──────────┘                                │
│                                 │ │ integration  │ │                                            │
│                                 │ │ bandStatistic│ │                                            │
│                                 │ │ noiseFloor   │ │                                            │
│                                 │ │ threshold    │ │                                            │
│                                 │ └──────────────┘ │                                            │
│                                 └──────────────────┘                                            │
└────────────────────────────────────────────────────────────────────────────────────────────────┘

 Fiber hot-swap self-evolution loop (SelfEvolver.evolve, Gen 0 → Gen 3, dev split only)

   ┌──────────────────────────┐   ┌───────────────────────┐   ┌────────────────────────────────┐
   │ 1. evaluate ctx.detector │──▶│ 2. diagnose failures  │──▶│ 3. propose SignalWindowConfig  │
   │    on dev SNR slices     │   │  misses @ low SNR,    │   │  Gemini (REST)  if API key     │
   │    → FailureTrace[]      │   │  false alarms on empty│   │  else deterministic optimizer  │
   └────────────▲─────────────┘   │  bands, floor error…  │   │  + exact residual threshold    │
                │                 └───────────────────────┘   └───────────────┬────────────────┘
                │                                                             ▼
   ┌────────────┴─────────────┐   ┌───────────────────────┐   ┌────────────────────────────────┐
   │ 6. next generation       │◀──│ 5. regression gate    │◀──│ 4. await fiber.dispose()       │
   │    (accepted candidate,  │   │  U = Pd − 1.5·Pfa ↑   │   │    ctx.plugin(SignalWindow-    │
   │     or rollback: dispose │   │  and Pfa ≤ 0.10 ?     │   │      Detector, candidate)      │
   │     candidate, re-mount  │   │  verify on dev        │   │    → new ctx.detector live     │
   │     previous generation) │   └───────────────────────┘   └────────────────────────────────┘
   └──────────────────────────┘
```

| Plugin | File | Registers as | Injects |
|---|---|---|---|
| `DspFixtureService` | `src/plugins/dsp-fixtures.ts` | `ctx.dsp` | none |
| `SignalWindowDetector` | `src/plugins/window-detector.ts` | `ctx.detector` | `dsp` |
| `SelfEvolver` | `src/plugins/self-evolver.ts` | `ctx.evolver` | `dsp` |
| `LearnedSpectralDetector` | `src/plugins/learned-detector.ts` | `ctx.learned` | `dsp` |

`src/benchmark.ts` builds the three-way comparison, `benchmark_report.json` and `sample_evidence.md`. `src/index.ts` is the entry point.

**Cordis details that matter:**
- The evolver injects only `dsp`, so disposing `ctx.detector` never unloads the evolver itself.
- Cordis service proxies bind `this.ctx` to the *caller's* context. The evolver and the learned detector therefore pin their own plugin context at construction and mount and inject through it.
- The entry point disposes each scoped `ctx.inject` right after use. Otherwise Cordis would re-run it on every detector hot-swap.

## Data and scoring

- **Fixtures:** seeded synthetic complex baseband (seed `0xc13`), 1 MS/s, 4096-sample windows, 16 sub-bands. AWGN at a nominal −60 dB/Hz with a per-window noise-floor drift drawn uniformly from ±3 dB. Each window has 0–3 emitters (25 % of windows are empty), each a narrowband tone or a pulsed burst (1–4 pulses, 20–50 % duty).
- **SNR slices:** −10, −5, 0, +5 and +10 dB of *in-band* SNR (emitter power relative to the noise inside one sub-band, while keyed on). There are 60 windows per slice per split.
- **Splits:** `dev` and a held-out `test` drawn from disjoint sub-seeds. Evolution, threshold calibration, feature standardisation and model fitting all use **`dev` only**. `LearnedSpectralDetector.train()` throws if given any non-`dev` fixture.
- **Metrics:** scoring is per (window, sub-band) cell. Pd = TP / occupied cells and Pfa = FP / vacant cells. Utility = Pd − 1.5·Pfa.

## The three detectors

**1. Baseline Fixed-Threshold Energy Detector (Gen 0)**
- Rectangular taper, single 4096-point periodogram, band-mean PSD.
- Assumed noise floor of −60 dB/Hz with a fixed absolute threshold of −57 dB/Hz.

**2. Self-Evolved Signal-Window Detector (Gen 3 survivor)**
- Produced by `SelfEvolver` from Gen 0 in three gated mutations, all accepted on dev in deterministic mode:

| Gen | Mutation (dominant failure mode it fixed) | U (dev) | Pd −10 dB | Pd −5 dB | Pfa |
|---|---|---|---|---|---|
| 0 | Baseline | 0.452 | 3.8 % | 6.0 % | 1.4 % |
| 1 | Peak-bin statistic: missed bands had a peak-to-mean ratio 4.8 dB above vacant bands (narrowband energy diluted by band averaging) | 0.784 | 39.2 % | 62.7 % | 1.4 % |
| 2 | Median noise floor + relative margin 9.83 dB: the assumed floor was 1.72 dB RMS off | 0.848 | 50.6 % | 80.6 % | 1.5 % |
| 3 | 4 averaged Welch sub-windows, margin 5.88 dB: χ²₂ variance of single-bin peaks | 0.858 | 48.1 % | 80.6 % | 0.7 % |

- **Final config:** `{bandStatistic: peak, subWindows: 4, integration: average, taper: rectangular, noiseFloor: median, threshold: relative +5.879 dB}`.
- **Why the taper stayed rectangular:** the optimizer kept it on purpose. Only about 19 % of false alarms sit next to occupied bands, so leakage is not a failure mode on this data. A Hann taper would cost about 1.8 dB of equivalent-noise-bandwidth on tone peaks.

**3. Learned Multi-Feature Spectral Detector**
- **Features:** 9 per sub-band, all relative to the window's own median floor (so drift-invariant by construction):

| Feature | What it measures |
|---|---|
| `peakToMedianDb` | Peak-to-median ratio of the 4096-point periodogram |
| `meanToMedianDb` | Band-mean energy over the floor |
| `spectralFlatnessDb` | Spectral flatness |
| `spectralEntropy` | Normalized spectral entropy |
| `logKurtosis` | Sub-band kurtosis (crest/impulsiveness) |
| `welchPeakDb` | Welch-averaged (4× Hann) peak |
| `persistence` | Fraction of 8 sub-windows above +9 dB (temporal persistence) |
| `maxSubPeakDb` | Strongest sub-window peak |
| `subPeakSpreadDb` | Spread of sub-window peaks |

- **Classifier:** L2-regularized (λ = 10⁻³) logistic regression on standardized features. It is fitted by 3000 iterations of deterministic full-batch gradient descent from zero, on the 4800 dev cells (410 occupied).
- **Threshold:** posterior > 0.178, the utility-optimal cut on dev under a 5 % design Pfa.
- **Largest standardized weights:** `welchPeakDb` +1.82, `logKurtosis` +1.33, `meanToMedianDb` +0.99.

## Held-out `test` benchmark

This is the exact output of `npm start` (deterministic mode), also in [`artifacts/benchmark_report.json`](artifacts/benchmark_report.json):

| SNR slice | Baseline Energy Pd | Baseline Energy Pfa | Self-Evolved Window Pd | Self-Evolved Window Pfa | Learned Feature Pd | Learned Feature Pfa |
|---|---:|---:|---:|---:|---:|---:|
| -10 dB | 7.6% | 3.2% | 46.8% | 0.9% | 50.6% | 0.6% |
| -5 dB | 13.6% | 1.7% | 74.1% | 0.7% | 80.2% | 0.2% |
| +0 dB | 22.0% | 2.3% | 100.0% | 0.9% | 100.0% | 0.6% |
| +5 dB | 75.8% | 2.4% | 100.0% | 0.2% | 100.0% | 0.2% |
| +10 dB | 94.2% | 0.6% | 100.0% | 1.1% | 100.0% | 0.5% |
| **Overall** | 43.5% | 2.0% | 85.7% | 0.8% | 87.5% | 0.4% |
| **Utility** | 0.405 |  | 0.845 |  | 0.869 |  |

Counts behind the overall row (test has 441 occupied and 4359 vacant cells):

| Detector | TP | FP |
|---|---:|---:|
| Baseline Energy | 192 | 89 |
| Self-Evolved Window | 378 | 34 |
| Learned Feature | 386 | 18 |

**Findings:**
- **Most of the gain over energy detection comes from the signal window, not from learning.** A drift-robust floor and a narrowband-matched statistic take overall Pd from 43.5 % to 85.7 % and cut Pfa from 2.0 % to 0.8 %. At −10 dB and −5 dB, Pd rises roughly 6× and 5×.
- **The learned detector adds a smaller, consistent increment:**
  - Pd +3.8 pp at −10 dB and +6.1 pp at −5 dB.
  - About half the false alarms (18 vs 34 cells).
  - Utility 0.869 vs 0.845.
- **Above 0 dB both reach 100 % Pd.** The gain comes from combining complementary statistics (peak, Welch peak, kurtosis, persistence) rather than from a better single statistic.
- **Remaining misses are mostly short −10 dB bursts.** Their energy is spread over only 20–50 % of the window and stays below every detector's noise-limited threshold (see example 2 in the evidence file).

## Signal evidence

[`artifacts/sample_evidence.md`](artifacts/sample_evidence.md) explains five held-out windows: a −10 dB tone, a −10 dB burst (a miss for all three detectors), a −5 dB multi-emitter window, an empty window on a hot receiver, and a +10 dB mixed window. An excerpt for `test/-10dB/012`, a −10 dB tone in band 1 at −393.2 kHz:

| Band | Truth | Baseline: band-mean PSD → margin | Self-Evolved: peak-bin PSD − floor → margin (conf.) | Learned: p (top feature contributions) |
|---:|:---:|---|---|---|
| 1 | occupied | ❌ MISS: −60.32 dB/Hz → −3.32 dB | ✅ HIT: −53.91 dB/Hz, SNR +6.68 dB → +0.80 dB (64 %) | ✅ HIT: p = 0.727 (logKurtosis +2.27, welchPeakDb +1.48, peakToMedianDb +0.53) |

In the hot-receiver example (`test/-10dB/013`, drift +2.89 dB, no emitters), the baseline raises 6 false alarms because its assumed floor is 2.9 dB too low. The evolved and learned detectors estimate the floor within 0.05 dB and stay silent.

## Repository layout

```
src/index.ts                     entry point: baseline → self-evolution → three-way benchmark
src/benchmark.ts                 comparison tables, benchmark_report.json, sample_evidence.md
src/plugins/dsp-fixtures.ts      ctx.dsp: FFT/PSD/sub-bands, seeded fixtures, evaluator
src/plugins/window-detector.ts   ctx.detector: the editable signal window + evidence
src/plugins/self-evolver.ts      ctx.evolver: Gen 0→3 evolution, hot-swap, regression gate
src/plugins/learned-detector.ts  ctx.learned: spectral features + logistic regression
artifacts/                       evolution_history.json, benchmark_report.json, sample_evidence.md
```

## Limitations

- **Synthetic data:** fixtures are synthetic AWGN with ideal tones and gated bursts. No multipath, phase noise, ADC effects or wideband or modulated emitters, and there is a single seed. The absolute numbers describe this fixture family, not over-the-air performance.
- **Same selection data:** the evolution gate and both threshold calibrations use the same `dev` data. Held-out `test` numbers are the unbiased estimate.
- **Gemini mode:** it is exercised with a mocked REST response (parsing, sanitizing, the guardrail rollback and the fallback). Results with the live model are not part of the reported benchmark.
