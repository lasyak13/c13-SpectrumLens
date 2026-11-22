/**
 * LearnedSpectralDetector — Cordis plugin registered as `ctx.learned` (injects `['dsp']`).
 *
 * Graduate extension: a lightweight learned detector to compare against energy detection.
 * Each (window, sub-band) cell is described by a multi-dimensional spectral feature vector
 * computed from the raw I/Q with `ctx.dsp`, and an L2-regularised logistic regression is
 * fitted by deterministic full-batch gradient descent **on the `dev` split only**. Fitting
 * refuses any fixture whose split is not `dev`, so `test` never touches training,
 * standardisation or threshold selection.
 *
 * All features are expressed relative to the window's own median sub-band noise floor, so
 * the model is invariant to the ±3 dB receiver drift by construction.
 */
import { Context, Service } from 'cordis'
import type { SignalWindowFixture, WindowPrediction } from './dsp-fixtures.ts'
import { toDb } from './dsp-fixtures.ts'

declare module 'cordis' {
  interface Context {
    learned: LearnedSpectralDetector
  }
}

export const FEATURE_NAMES = [
  'peakToMedianDb', //        4096-pt periodogram: peak bin vs. window median floor
  'meanToMedianDb', //        band-mean PSD vs. window median floor (radiometer energy)
  'spectralFlatnessDb', //    geometric / arithmetic mean of band bins (≈ −2.5 dB for noise, ≪ for tones)
  'spectralEntropy', //       normalised Shannon entropy of the band's bin distribution
  'logKurtosis', //           log10 kurtosis of band bin powers (crest / impulsiveness)
  'welchPeakDb', //           4×1024 Welch-averaged peak bin vs. floor (variance-reduced peak)
  'persistence', //           fraction of 8 sub-windows whose peak exceeds +9 dB over floor
  'maxSubPeakDb', //          strongest sub-window peak vs. floor (burst-sensitive)
  'subPeakSpreadDb', //       std-dev of sub-window peaks (temporal on/off keying)
] as const
export type FeatureName = (typeof FEATURE_NAMES)[number]

export interface LearnedConfig {
  /** L2 penalty on standardised weights. */
  l2: number
  learningRate: number
  iterations: number
  /** Design Pfa cap used when picking the decision threshold on dev. */
  designPfa: number
  /** Utility weight on Pfa (utility = Pd − w·Pfa). */
  pfaWeight: number
  subWindows: number
  persistenceThresholdDb: number
}

export interface LearnedModel {
  kind: 'l2-logistic-regression'
  trainedOn: { split: 'dev'; windows: number; cells: number; positives: number }
  features: readonly FeatureName[]
  /** Standardisation statistics (dev only). */
  mean: number[]
  std: number[]
  weights: number[]
  bias: number
  /** Posterior threshold chosen on dev. */
  threshold: number
  training: { finalLoss: number; iterations: number; devPd: number; devPfa: number }
  config: LearnedConfig
}

export interface LearnedBandEvidence {
  band: number
  centerFrequencyHz: number
  features: Record<FeatureName, number>
  /** Standardised feature × weight, i.e. each feature's additive contribution to the logit. */
  contributions: Record<FeatureName, number>
  logit: number
  probability: number
  occupied: boolean
}

export interface LearnedDetection extends WindowPrediction {
  noiseFloorDb: number
  bands: LearnedBandEvidence[]
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))

export class LearnedSpectralDetector extends Service {
  static inject = ['dsp']

  static readonly defaultConfig: LearnedConfig = {
    l2: 1e-3,
    learningRate: 0.5,
    iterations: 3000,
    designPfa: 0.05,
    pfaWeight: 1.5,
    subWindows: 8,
    persistenceThresholdDb: 9,
  }

  readonly home: Context
  config: LearnedConfig
  model?: LearnedModel

  constructor(ctx: Context, config: Partial<LearnedConfig> = {}) {
    super(ctx, 'learned')
    this.home = ctx
    this.config = { ...LearnedSpectralDetector.defaultConfig, ...config }
  }

  // ------------------------------ Features ------------------------------

  /** Per-band feature vectors for one window, plus its median noise floor (linear). */
  extractFeatures(w: Pick<SignalWindowFixture, 'samplesI' | 'samplesQ' | 'sampleRateHz' | 'numSubBands'>) {
    const dsp = this.home.dsp
    const nb = w.numSubBands
    const n = w.samplesI.length
    const med = (xs: number[]) => {
      const s = [...xs].sort((a, b) => a - b)
      const m = s.length >> 1
      return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
    }

    // (1) Full-resolution periodogram.
    const full = dsp.computePsd(w.samplesI, w.samplesQ, w.sampleRateHz)
    const fullBands = dsp.subBands(full, nb)
    const floor = med(fullBands.map((b) => b.meanPsd))
    const bpb = full.fftSize / nb

    // (2) Welch: 4 × (n/4), Hann taper.
    const welch = dsp.subBands(
      dsp.computePsd(w.samplesI, w.samplesQ, w.sampleRateHz, { segments: 4, taper: 'hann' }),
      nb,
    )
    const welchFloor = med(welch.map((b) => b.meanPsd))

    // (3) Short sub-windows for temporal persistence.
    const K = this.config.subWindows
    const seg = n / K
    const subPeaksDb: number[][] = Array.from({ length: nb }, () => [])
    for (let k = 0; k < K; k++) {
      const i = w.samplesI.subarray(k * seg, (k + 1) * seg) as Float64Array
      const q = w.samplesQ.subarray(k * seg, (k + 1) * seg) as Float64Array
      const bands = dsp.subBands(dsp.computePsd(i, q, w.sampleRateHz), nb)
      const f = med(bands.map((b) => b.meanPsd))
      bands.forEach((b, j) => subPeaksDb[j].push(toDb(b.peakPsd / f)))
    }

    const rows: number[][] = []
    for (let b = 0; b < nb; b++) {
      const bins = full.psd.subarray(b * bpb, (b + 1) * bpb)
      let sum = 0
      let logSum = 0
      for (const x of bins) (sum += x), (logSum += Math.log(Math.max(x, 1e-300)))
      const mean = sum / bpb
      let entropy = 0
      let m2 = 0
      let m4 = 0
      for (const x of bins) {
        const p = x / sum
        if (p > 0) entropy -= p * Math.log(p)
        m2 += (x - mean) ** 2
        m4 += (x - mean) ** 4
      }
      m2 /= bpb
      m4 /= bpb
      const sp = subPeaksDb[b]
      const spMean = sp.reduce((a, c) => a + c, 0) / K
      rows.push([
        toDb(fullBands[b].peakPsd / floor),
        toDb(mean / floor),
        toDb(Math.exp(logSum / bpb) / mean),
        entropy / Math.log(bpb),
        Math.log10(m4 / (m2 * m2)),
        toDb(welch[b].peakPsd / welchFloor),
        sp.filter((x) => x > this.config.persistenceThresholdDb).length / K,
        Math.max(...sp),
        Math.sqrt(sp.reduce((a, c) => a + (c - spMean) ** 2, 0) / K),
      ])
    }
    return { rows, floor, centers: fullBands.map((b) => b.centerHz) }
  }

  // ------------------------------ Training ------------------------------

  train(dev: SignalWindowFixture[]): LearnedModel {
    const leaked = dev.filter((f) => f.split !== 'dev')
    if (leaked.length) throw new Error(`train: refusing ${leaked.length} non-dev fixtures (e.g. ${leaked[0].id})`)
    const cfg = this.config
    const X: number[][] = []
    const y: number[] = []
    for (const fx of dev) {
      const truth = new Set(fx.occupiedBands)
      this.extractFeatures(fx).rows.forEach((r, b) => {
        X.push(r)
        y.push(truth.has(b) ? 1 : 0)
      })
    }
    const d = FEATURE_NAMES.length
    const N = X.length
    const mean = Array.from({ length: d }, (_, j) => X.reduce((a, r) => a + r[j], 0) / N)
    const std = Array.from({ length: d }, (_, j) =>
      Math.sqrt(X.reduce((a, r) => a + (r[j] - mean[j]) ** 2, 0) / N) || 1,
    )
    const Z = X.map((r) => r.map((v, j) => (v - mean[j]) / std[j]))

    // Deterministic full-batch gradient descent from a zero initialisation.
    const wts = new Array(d).fill(0)
    let bias = 0
    let loss = 0
    for (let it = 0; it < cfg.iterations; it++) {
      const gw = new Array(d).fill(0)
      let gb = 0
      loss = 0
      for (let i = 0; i < N; i++) {
        let z = bias
        for (let j = 0; j < d; j++) z += wts[j] * Z[i][j]
        const p = sigmoid(z)
        const e = p - y[i]
        gb += e
        for (let j = 0; j < d; j++) gw[j] += e * Z[i][j]
        loss -= y[i] ? Math.log(Math.max(p, 1e-12)) : Math.log(Math.max(1 - p, 1e-12))
      }
      loss = loss / N + (cfg.l2 / 2) * wts.reduce((a, w) => a + w * w, 0)
      for (let j = 0; j < d; j++) wts[j] -= cfg.learningRate * (gw[j] / N + cfg.l2 * wts[j])
      bias -= cfg.learningRate * (gb / N)
    }

    // Decision threshold: utility-optimal posterior cut on dev under the design Pfa cap.
    const scored = Z.map((z, i) => ({ p: sigmoid(bias + z.reduce((a, v, j) => a + v * wts[j], 0)), y: y[i] }))
    scored.sort((a, b) => b.p - a.p)
    const nPos = y.filter(Boolean).length
    const nNeg = N - nPos
    let tp = 0
    let fp = 0
    let best = { threshold: 0.5, pd: 0, pfa: 0, u: -Infinity }
    for (let i = 0; i < scored.length; i++) {
      scored[i].y ? tp++ : fp++
      if (i + 1 < scored.length && scored[i + 1].p === scored[i].p) continue
      const pd = tp / nPos
      const pfa = fp / nNeg
      if (pfa > cfg.designPfa) break
      const u = pd - cfg.pfaWeight * pfa
      if (u > best.u) {
        const next = i + 1 < scored.length ? scored[i + 1].p : 0
        best = { threshold: (scored[i].p + next) / 2, pd, pfa, u }
      }
    }

    this.model = {
      kind: 'l2-logistic-regression',
      trainedOn: { split: 'dev', windows: dev.length, cells: N, positives: nPos },
      features: FEATURE_NAMES,
      mean,
      std,
      weights: wts,
      bias,
      threshold: best.threshold,
      training: { finalLoss: loss, iterations: cfg.iterations, devPd: best.pd, devPfa: best.pfa },
      config: cfg,
    }
    return this.model
  }

  // ------------------------------ Inference ------------------------------

  detect(w: SignalWindowFixture): LearnedDetection {
    const m = this.model
    if (!m) throw new Error('detect: model not trained')
    const { rows, floor, centers } = this.extractFeatures(w)
    const bands = rows.map((r, b): LearnedBandEvidence => {
      const contrib = r.map((v, j) => ((v - m.mean[j]) / m.std[j]) * m.weights[j])
      const logit = m.bias + contrib.reduce((a, c) => a + c, 0)
      const probability = sigmoid(logit)
      return {
        band: b,
        centerFrequencyHz: centers[b],
        features: Object.fromEntries(FEATURE_NAMES.map((f, j) => [f, r[j]])) as Record<FeatureName, number>,
        contributions: Object.fromEntries(FEATURE_NAMES.map((f, j) => [f, contrib[j]])) as Record<FeatureName, number>,
        logit,
        probability,
        occupied: probability > m.threshold,
      }
    })
    return {
      windowId: w.id,
      occupiedBands: bands.filter((b) => b.occupied).map((b) => b.band),
      noiseFloorDb: toDb(floor),
      bands,
    }
  }

  detectAll(ws: SignalWindowFixture[]): LearnedDetection[] {
    return ws.map((w) => this.detect(w))
  }
}

export default LearnedSpectralDetector
