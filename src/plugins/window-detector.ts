/**
 * SignalWindowDetector — Cordis plugin registered as `ctx.detector`.
 *
 * The *signal window* is the explicit, editable unit of the detector: its length,
 * sub-window slicing (Welch segments), taper, noise-floor estimator and threshold
 * rule are all fields of `SignalWindowConfig`. Generation 0 ("First Demo") is the
 * classic Baseline Energy Detector: rectangular taper, a single periodogram, a raw
 * fixed energy threshold and no noise-floor adaptation.
 */
import { Context, Service } from 'cordis'
import type {
  PsdOptions,
  SignalWindowFixture,
  SubBand,
  TaperName,
  WindowPrediction,
} from './dsp-fixtures.ts'
import { DspFixtureService, toDb } from './dsp-fixtures.ts'

declare module 'cordis' {
  interface Context {
    detector: SignalWindowDetector
  }
}

// ---------------------------------------------------------------------------
// Signal-window model
// ---------------------------------------------------------------------------

export type NoiseFloorMethod =
  /** Trust a fixed, assumed noise floor (no adaptation). */
  | { kind: 'assumed'; floorDb: number }
  /** Median of sub-band PSDs (robust to a minority of occupied bands). */
  | { kind: 'median' }
  /** Given percentile of sub-band PSDs (e.g. 25). */
  | { kind: 'percentile'; percentile: number }

export type ThresholdRule =
  /** Raw energy threshold in dB/Hz, independent of the estimated floor. */
  | { kind: 'absolute'; thresholdDb: number }
  /** Threshold expressed as a margin above the estimated noise floor, dB. */
  | { kind: 'relative'; marginDb: number }

export interface SignalWindowConfig {
  /** Generation label, carried into evidence for traceability. */
  generation: string
  /** Number of samples consumed from each window (from the start). 0 = all. */
  windowLength: number
  /** Number of non-overlapping sub-windows averaged into the PSD. */
  subWindows: number
  taper: TaperName
  numSubBands: number
  noiseFloor: NoiseFloorMethod
  threshold: ThresholdRule
  /**
   * Assumed one-sigma uncertainty of the noise-floor estimate in dB, combined with
   * the statistical PSD variance to turn SNR margins into confidences.
   */
  floorUncertaintyDb: number
}

export interface BandEvidence {
  band: number
  centerFrequencyHz: number
  bandwidthHz: number
  measuredPsdDb: number
  noiseFloorDb: number
  thresholdDb: number
  /** Measured PSD minus estimated noise floor. */
  snrDb: number
  /** Measured PSD minus detection threshold (positive ⇒ declared occupied). */
  snrMarginDb: number
  occupied: boolean
  /** Confidence (0.5–1) in the decision that was made for this band. */
  confidence: number
  summary: string
}

export interface WindowEvidence {
  windowId: string
  generation: string
  config: SignalWindowConfig
  sampleRateHz: number
  samplesUsed: number
  fftSize: number
  binWidthHz: number
  noiseFloorDb: number
  thresholdDb: number
  occupiedBands: number[]
  bands: BandEvidence[]
  narrative: string
}

export interface DetectionResult extends WindowPrediction {
  evidence: WindowEvidence
}

/** Generation 0: Baseline Energy Detector. */
export function baselineEnergyDetectorConfig(nominalNoiseFloorDb: number): SignalWindowConfig {
  return {
    generation: 'Gen-0 Baseline Energy Detector',
    windowLength: 0,
    subWindows: 1,
    taper: 'rectangular',
    numSubBands: 16,
    noiseFloor: { kind: 'assumed', floorDb: nominalNoiseFloorDb },
    threshold: { kind: 'absolute', thresholdDb: nominalNoiseFloorDb + 3 },
    floorUncertaintyDb: 1,
  }
}

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

export class SignalWindowDetector extends Service {
  static inject = ['dsp']

  config: SignalWindowConfig

  constructor(ctx: Context, config?: Partial<SignalWindowConfig>) {
    super(ctx, 'detector')
    const nominal = ctx.dsp.nominalNoiseFloorDb(DspFixtureService.defaultFixtureOptions.sampleRateHz)
    this.config = { ...baselineEnergyDetectorConfig(nominal), ...config }
  }

  /** Edit the signal window in place (returns the new config). */
  configure(patch: Partial<SignalWindowConfig>): SignalWindowConfig {
    this.config = { ...this.config, ...patch }
    return this.config
  }

  estimateNoiseFloorDb(bands: SubBand[], method: NoiseFloorMethod): number {
    if (method.kind === 'assumed') return method.floorDb
    const sorted = bands.map((b) => b.meanPsd).sort((a, b) => a - b)
    const p = method.kind === 'median' ? 50 : method.percentile
    const pos = (Math.min(100, Math.max(0, p)) / 100) * (sorted.length - 1)
    const lo = Math.floor(pos)
    const hi = Math.ceil(pos)
    return toDb(sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo))
  }

  detect(
    window: Pick<SignalWindowFixture, 'id' | 'samplesI' | 'samplesQ' | 'sampleRateHz'>,
    override: Partial<SignalWindowConfig> = {},
  ): DetectionResult {
    const cfg: SignalWindowConfig = { ...this.config, ...override }
    const total = window.samplesI.length
    const used = cfg.windowLength > 0 ? Math.min(cfg.windowLength, total) : total
    const segLen = 2 ** Math.floor(Math.log2(used / cfg.subWindows))
    const psdOpts: PsdOptions = { fftSize: segLen, segments: cfg.subWindows, taper: cfg.taper }
    const { psd, bands } = this.ctx.dsp.subBandPsd(
      window.samplesI.subarray(0, used) as Float64Array,
      window.samplesQ.subarray(0, used) as Float64Array,
      window.sampleRateHz,
      cfg.numSubBands,
      psdOpts,
    )

    const noiseFloorDb = this.estimateNoiseFloorDb(bands, cfg.noiseFloor)
    const thresholdDb =
      cfg.threshold.kind === 'absolute' ? cfg.threshold.thresholdDb : noiseFloorDb + cfg.threshold.marginDb

    // Std-dev (dB) of a band-mean PSD: averaged chi-square with (bins × segments) dof.
    const binsPerBand = psd.fftSize / cfg.numSubBands
    const measSigmaDb = 10 / Math.LN10 / Math.sqrt(binsPerBand * cfg.subWindows)
    const sigmaDb = Math.hypot(measSigmaDb, cfg.floorUncertaintyDb)

    const evidence: BandEvidence[] = bands.map((b) => {
      const snrDb = b.meanPsdDb - noiseFloorDb
      const snrMarginDb = b.meanPsdDb - thresholdDb
      const occupied = snrMarginDb > 0
      // Logistic approximation of Φ(|margin|/σ).
      const confidence = 1 / (1 + Math.exp((-1.702 * Math.abs(snrMarginDb)) / sigmaDb))
      const f = (hz: number) => `${(hz / 1e3).toFixed(1)} kHz`
      return {
        band: b.index,
        centerFrequencyHz: b.centerHz,
        bandwidthHz: b.bandwidthHz,
        measuredPsdDb: b.meanPsdDb,
        noiseFloorDb,
        thresholdDb,
        snrDb,
        snrMarginDb,
        occupied,
        confidence,
        summary:
          `Band ${String(b.index).padStart(2)} @ ${f(b.centerHz)} (±${f(b.bandwidthHz / 2)}): ` +
          `PSD ${b.meanPsdDb.toFixed(2)} dB/Hz vs floor ${noiseFloorDb.toFixed(2)} dB/Hz ` +
          `→ SNR ${snrDb >= 0 ? '+' : ''}${snrDb.toFixed(2)} dB, margin ${snrMarginDb >= 0 ? '+' : ''}` +
          `${snrMarginDb.toFixed(2)} dB vs threshold ⇒ ${occupied ? 'OCCUPIED' : 'vacant'} ` +
          `(confidence ${(confidence * 100).toFixed(1)}%)`,
      }
    })

    const occupiedBands = evidence.filter((e) => e.occupied).map((e) => e.band)
    const narrative =
      occupiedBands.length === 0
        ? `No sub-band exceeded the ${thresholdDb.toFixed(2)} dB/Hz threshold; window declared vacant.`
        : `${occupiedBands.length} of ${cfg.numSubBands} sub-bands exceeded the ${thresholdDb.toFixed(2)} dB/Hz ` +
          `threshold: ${evidence
            .filter((e) => e.occupied)
            .map((e) => `${(e.centerFrequencyHz / 1e3).toFixed(1)} kHz (+${e.snrMarginDb.toFixed(1)} dB)`)
            .join(', ')}.`

    return {
      windowId: window.id,
      occupiedBands,
      evidence: {
        windowId: window.id,
        generation: cfg.generation,
        config: cfg,
        sampleRateHz: window.sampleRateHz,
        samplesUsed: segLen * cfg.subWindows,
        fftSize: psd.fftSize,
        binWidthHz: psd.binWidthHz,
        noiseFloorDb,
        thresholdDb,
        occupiedBands,
        bands: evidence,
        narrative,
      },
    }
  }

  detectAll(windows: SignalWindowFixture[], override?: Partial<SignalWindowConfig>): DetectionResult[] {
    return windows.map((w) => this.detect(w, override))
  }
}

export default SignalWindowDetector
