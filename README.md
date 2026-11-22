# C13: SpectrumLens

Signals, Wireless and Control — signal-window spectrum occupancy detection, built on the [Cordis](https://github.com/cordiverse/cordis) plugin framework (TypeScript, Node ESM).

## Phase 1 — Generation 0 (First Demo)

```bash
npm install
npx tsx src/index.ts   # or: npm start
npm run typecheck
```

### Architecture

| Plugin | File | Registers as | Purpose |
|---|---|---|---|
| `DspFixtureService` | `src/plugins/dsp-fixtures.ts` | `ctx.dsp` | Radix-2 complex FFT, tapers (rectangular / Hann), Welch PSD, sub-band aggregation; seeded synthetic I/Q fixtures (tones + pulsed bursts) at −10/−5/0/+5/+10 dB SNR, split into `dev` and held-out `test`; deterministic Pd/Pfa evaluator. |
| `SignalWindowDetector` | `src/plugins/window-detector.ts` | `ctx.detector` (`inject = ['dsp']`) | The signal window as an editable unit (length, sub-window slicing, taper, noise-floor method, threshold rule). Gen-0 = Baseline Energy Detector: rectangular taper, single periodogram, fixed absolute threshold, no noise-floor adaptation. Emits `occupiedBands` plus per-band evidence (center Hz, PSD dB/Hz, noise floor, SNR, margin, confidence). |

### Fixture conventions

- Complex AWGN, nominal noise PSD −60 dB/Hz at 1 MS/s, with per-window noise-floor drift (uniform ±3 dB) to model receiver uncertainty.
- SNR is **in-band**: emitter power relative to the true noise inside one sub-band (16 sub-bands of 62.5 kHz), measured while the emitter is keyed on. Bursts have 20–50 % duty cycle, so their average SNR is lower.
- Every window draws from a sub-seed hashed from `(seed, split, snr, index)`, so runs are bit-for-bit reproducible.
- Scoring is per (window, sub-band) cell: Pd = TP / occupied cells, Pfa = FP / vacant cells.
