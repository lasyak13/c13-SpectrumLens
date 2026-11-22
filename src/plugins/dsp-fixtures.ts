/**
 * DspFixtureService — Cordis plugin registered as `ctx.dsp`.
 *
 * Provides:
 *   a) Complex radix-2 FFT, tapers, Welch-style PSD and sub-band power aggregation.
 *   b) A deterministic, seeded synthetic I/Q fixture generator (narrowband tones and
 *      pulsed/bursty emitters) across SNR slices, split into `dev` and held-out `test`.
 *   c) A deterministic ground-truth evaluator computing Pd / Pfa per SNR slice and overall.
 */
import { Context, Service } from 'cordis'

declare module 'cordis' {
  interface Context {
    dsp: DspFixtureService
  }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type TaperName = 'rectangular' | 'hann' | 'hamming' | 'blackman'

export interface PsdOptions {
  /** FFT length per segment (power of two). Defaults to the largest power of two ≤ sample count. */
  fftSize?: number
  /** Number of non-overlapping segments averaged (Welch/Bartlett). 1 = single periodogram. */
  segments?: number
  /** Taper applied to each segment. */
  taper?: TaperName
  /**
   * How per-segment periodograms are combined: `mean` (Welch/Bartlett averaging, lowers
   * variance) or `max` (max-hold, keeps short bursts that are on in only some segments).
   */
  combine?: 'mean' | 'max'
}

export interface PsdResult {
  sampleRateHz: number
  fftSize: number
  segments: number
  taper: TaperName
  /** Bin resolution in Hz. */
  binWidthHz: number
  /** Bin center frequencies (fft-shifted, ascending, -fs/2 .. +fs/2). */
  freqsHz: Float64Array
  /** One-sided-in-time, two-sided-in-frequency PSD in linear units of power/Hz. */
  psd: Float64Array
}

export interface SubBand {
  index: number
  centerHz: number
  lowHz: number
  highHz: number
  bandwidthHz: number
  /** Mean PSD across the band's bins, linear power/Hz. */
  meanPsd: number
  /** Mean PSD across the band's bins, dB/Hz. */
  meanPsdDb: number
  /** Integrated band power, dB. */
  powerDb: number
  /** Largest single-bin PSD inside the band, linear power/Hz (narrowband matched statistic). */
  peakPsd: number
  /** Largest single-bin PSD inside the band, dB/Hz. */
  peakPsdDb: number
  /** Frequency of the peak bin, Hz. */
  peakHz: number
}

export type EmitterKind = 'tone' | 'burst'

export interface EmitterLabel {
  kind: EmitterKind
  band: number
  /** Baseband frequency of the carrier in Hz. */
  frequencyHz: number
  /** In-band SNR in dB while the emitter is keyed on (relative to noise within one sub-band). */
  snrDb: number
  /** Fraction of the window the emitter is on (1 for continuous tones). */
  dutyCycle: number
  /** Number of pulses within the window (bursts only). */
  pulses: number
}

export interface SignalWindowFixture {
  id: string
  split: Split
  snrDb: SnrSlice
  sampleRateHz: number
  numSubBands: number
  samplesI: Float64Array
  samplesQ: Float64Array
  /** True per-sample noise power deviation from nominal, dB (models receiver/noise-floor drift). */
  noiseOffsetDb: number
  /** True noise PSD of this window, dB/Hz. */
  trueNoiseFloorDb: number
  emitters: EmitterLabel[]
  /** Ground truth: sorted list of occupied sub-band indices. */
  occupiedBands: number[]
}

export const SNR_SLICES = [-10, -5, 0, 5, 10] as const
export type SnrSlice = (typeof SNR_SLICES)[number]
export type Split = 'dev' | 'test'

export interface FixtureOptions {
  seed?: number
  sampleRateHz?: number
  windowLength?: number
  numSubBands?: number
  windowsPerSlice?: number
  snrSlices?: readonly SnrSlice[]
  /** Peak-to-peak range of the per-window noise-floor drift, dB (uniform ±range/2). */
  noiseDriftDb?: number
  /** Probability that a window contains no emitters at all. */
  emptyWindowProbability?: number
  /** Maximum emitters per window. */
  maxEmitters?: number
}

export interface FixtureSet {
  options: Required<FixtureOptions>
  /** Nominal noise PSD (no drift), dB/Hz. */
  nominalNoiseFloorDb: number
  dev: SignalWindowFixture[]
  test: SignalWindowFixture[]
}

export interface WindowPrediction {
  windowId: string
  occupiedBands: number[]
}

export interface SliceMetrics {
  snrDb: SnrSlice | 'overall'
  windows: number
  occupiedTotal: number
  vacantTotal: number
  truePositives: number
  falsePositives: number
  missed: number
  /** Probability of detection = TP / occupied. */
  pd: number
  /** Probability of false alarm = FP / vacant. */
  pfa: number
}

export interface EvaluationReport {
  split: Split | 'mixed'
  slices: SliceMetrics[]
  overall: SliceMetrics
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (mulberry32) + Gaussian (Box–Muller)
// ---------------------------------------------------------------------------

export class SeededRandom {
  private state: number
  private spare: number | null = null

  constructor(seed: number) {
    this.state = seed >>> 0
  }

  next(): number {
    let t = (this.state = (this.state + 0x6d2b79f5) >>> 0)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }

  uniform(lo: number, hi: number): number {
    return lo + (hi - lo) * this.next()
  }

  int(lo: number, hiInclusive: number): number {
    return lo + Math.floor(this.next() * (hiInclusive - lo + 1))
  }

  gaussian(): number {
    if (this.spare !== null) {
      const s = this.spare
      this.spare = null
      return s
    }
    let u = 0
    while (u === 0) u = this.next()
    const v = this.next()
    const r = Math.sqrt(-2 * Math.log(u))
    this.spare = r * Math.sin(2 * Math.PI * v)
    return r * Math.cos(2 * Math.PI * v)
  }
}

/** Stable 32-bit hash for deriving sub-seeds from strings (FNV-1a). */
function hashSeed(...parts: (string | number)[]): number {
  let h = 0x811c9dc5
  for (const ch of parts.join('|')) {
    h ^= ch.charCodeAt(0)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

export const toDb = (x: number) => 10 * Math.log10(Math.max(x, 1e-30))
export const fromDb = (db: number) => 10 ** (db / 10)

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

export class DspFixtureService extends Service {
  constructor(ctx: Context) {
    super(ctx, 'dsp')
  }

  // ------------------------------- (a) DSP --------------------------------

  /** In-place iterative radix-2 complex FFT. */
  fft(re: Float64Array, im: Float64Array): void {
    const n = re.length
    if (n !== im.length) throw new RangeError('fft: re/im length mismatch')
    if (n & (n - 1)) throw new RangeError(`fft: length ${n} is not a power of two`)
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1
      for (; j & bit; bit >>= 1) j ^= bit
      j ^= bit
      if (i < j) {
        ;[re[i], re[j]] = [re[j], re[i]]
        ;[im[i], im[j]] = [im[j], im[i]]
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len
      const wr = Math.cos(ang)
      const wi = Math.sin(ang)
      const half = len >> 1
      for (let i = 0; i < n; i += len) {
        let cr = 1
        let ci = 0
        for (let k = 0; k < half; k++) {
          const a = i + k
          const b = a + half
          const tr = re[b] * cr - im[b] * ci
          const ti = re[b] * ci + im[b] * cr
          re[b] = re[a] - tr
          im[b] = im[a] - ti
          re[a] += tr
          im[a] += ti
          const ncr = cr * wr - ci * wi
          ci = cr * wi + ci * wr
          cr = ncr
        }
      }
    }
  }

  taper(name: TaperName, n: number): Float64Array {
    const w = new Float64Array(n)
    if (name === 'rectangular') return w.fill(1)
    // Periodic (DFT-even) cosine-sum tapers.
    const [a0, a1, a2] =
      name === 'hann' ? [0.5, 0.5, 0] : name === 'hamming' ? [0.54, 0.46, 0] : [0.42, 0.5, 0.08]
    for (let i = 0; i < n; i++) {
      const x = (2 * Math.PI * i) / n
      w[i] = a0 - a1 * Math.cos(x) + a2 * Math.cos(2 * x)
    }
    return w
  }

  /**
   * Power spectral density of complex baseband samples (I + jQ).
   * Normalised so that Σ psd[k]·binWidthHz equals the mean sample power.
   */
  computePsd(
    samplesI: ArrayLike<number>,
    samplesQ: ArrayLike<number>,
    sampleRateHz: number,
    options: PsdOptions = {},
  ): PsdResult {
    const total = samplesI.length
    if (total !== samplesQ.length) throw new RangeError('computePsd: I/Q length mismatch')
    const segments = Math.max(1, Math.floor(options.segments ?? 1))
    const taperName = options.taper ?? 'rectangular'
    let fftSize = options.fftSize ?? 2 ** Math.floor(Math.log2(Math.floor(total / segments)))
    if (fftSize * segments > total) {
      throw new RangeError(`computePsd: ${segments}×${fftSize} exceeds ${total} samples`)
    }
    const w = this.taper(taperName, fftSize)
    let wPower = 0
    for (let i = 0; i < fftSize; i++) wPower += w[i] * w[i]

    const acc = new Float64Array(fftSize)
    const re = new Float64Array(fftSize)
    const im = new Float64Array(fftSize)
    for (let s = 0; s < segments; s++) {
      const off = s * fftSize
      for (let i = 0; i < fftSize; i++) {
        re[i] = samplesI[off + i] * w[i]
        im[i] = samplesQ[off + i] * w[i]
      }
      this.fft(re, im)
      if (options.combine === 'max') {
        for (let k = 0; k < fftSize; k++) acc[k] = Math.max(acc[k], segments * (re[k] * re[k] + im[k] * im[k]))
      } else {
        for (let k = 0; k < fftSize; k++) acc[k] += re[k] * re[k] + im[k] * im[k]
      }
    }

    const binWidthHz = sampleRateHz / fftSize
    const psd = new Float64Array(fftSize)
    const freqsHz = new Float64Array(fftSize)
    const half = fftSize >> 1
    const scale = 1 / (segments * wPower * sampleRateHz)
    for (let k = 0; k < fftSize; k++) {
      const src = (k + half) % fftSize // fft-shift
      psd[k] = acc[src] * scale
      freqsHz[k] = (k - half) * binWidthHz
    }
    return { sampleRateHz, fftSize, segments, taper: taperName, binWidthHz, freqsHz, psd }
  }

  /** Aggregate a PSD into `numSubBands` equal-width contiguous sub-bands spanning -fs/2..+fs/2. */
  subBands(psd: PsdResult, numSubBands: number): SubBand[] {
    const binsPerBand = psd.fftSize / numSubBands
    if (!Number.isInteger(binsPerBand)) {
      throw new RangeError(`subBands: fftSize ${psd.fftSize} not divisible by ${numSubBands}`)
    }
    const bandwidthHz = psd.sampleRateHz / numSubBands
    const bands: SubBand[] = []
    for (let b = 0; b < numSubBands; b++) {
      let sum = 0
      let peak = 0
      let peakK = b * binsPerBand
      for (let k = b * binsPerBand; k < (b + 1) * binsPerBand; k++) {
        sum += psd.psd[k]
        if (psd.psd[k] > peak) (peak = psd.psd[k]), (peakK = k)
      }
      const meanPsd = sum / binsPerBand
      const lowHz = -psd.sampleRateHz / 2 + b * bandwidthHz
      bands.push({
        index: b,
        lowHz,
        highHz: lowHz + bandwidthHz,
        centerHz: lowHz + bandwidthHz / 2,
        bandwidthHz,
        meanPsd,
        meanPsdDb: toDb(meanPsd),
        powerDb: toDb(meanPsd * bandwidthHz),
        peakPsd: peak,
        peakPsdDb: toDb(peak),
        peakHz: psd.freqsHz[peakK],
      })
    }
    return bands
  }

  /** Convenience: PSD + sub-band aggregation in one call. */
  subBandPsd(
    samplesI: ArrayLike<number>,
    samplesQ: ArrayLike<number>,
    sampleRateHz: number,
    numSubBands: number,
    options: PsdOptions = {},
  ) {
    const psd = this.computePsd(samplesI, samplesQ, sampleRateHz, options)
    return { psd, bands: this.subBands(psd, numSubBands) }
  }

  // ------------------------- (b) Fixture generator -------------------------

  static readonly defaultFixtureOptions: Required<FixtureOptions> = {
    seed: 0xc13,
    sampleRateHz: 1_000_000,
    windowLength: 4096,
    numSubBands: 16,
    windowsPerSlice: 60,
    snrSlices: SNR_SLICES,
    noiseDriftDb: 6,
    emptyWindowProbability: 0.25,
    maxEmitters: 3,
  }

  /** Nominal noise PSD in dB/Hz for unit-variance complex noise. */
  nominalNoiseFloorDb(sampleRateHz: number): number {
    return toDb(1 / sampleRateHz)
  }

  /**
   * Generate a reproducible labeled fixture set. Each (split, slice, index) triple draws
   * from its own sub-seed, so dev and test never share random streams and adding slices
   * never perturbs existing windows.
   */
  generateFixtures(options: FixtureOptions = {}): FixtureSet {
    const opts: Required<FixtureOptions> = { ...DspFixtureService.defaultFixtureOptions, ...options }
    const build = (split: Split) =>
      opts.snrSlices.flatMap((snr) =>
        Array.from({ length: opts.windowsPerSlice }, (_, i) => this.generateWindow(opts, split, snr, i)),
      )
    return {
      options: opts,
      nominalNoiseFloorDb: this.nominalNoiseFloorDb(opts.sampleRateHz),
      dev: build('dev'),
      test: build('test'),
    }
  }

  generateWindow(
    opts: Required<FixtureOptions>,
    split: Split,
    snrDb: SnrSlice,
    index: number,
  ): SignalWindowFixture {
    const rng = new SeededRandom(hashSeed(opts.seed, split, snrDb, index))
    const { sampleRateHz: fs, windowLength: n, numSubBands } = opts
    const bandwidthHz = fs / numSubBands

    // Complex AWGN with per-window noise-floor drift.
    const noiseOffsetDb = rng.uniform(-opts.noiseDriftDb / 2, opts.noiseDriftDb / 2)
    const noisePower = fromDb(noiseOffsetDb)
    const sigma = Math.sqrt(noisePower / 2)
    const samplesI = new Float64Array(n)
    const samplesQ = new Float64Array(n)
    for (let t = 0; t < n; t++) {
      samplesI[t] = sigma * rng.gaussian()
      samplesQ[t] = sigma * rng.gaussian()
    }

    // Emitters on distinct sub-bands.
    const emitters: EmitterLabel[] = []
    const count = rng.next() < opts.emptyWindowProbability ? 0 : rng.int(1, opts.maxEmitters)
    const free = Array.from({ length: numSubBands }, (_, b) => b)
    for (let e = 0; e < count; e++) {
      const band = free.splice(rng.int(0, free.length - 1), 1)[0]
      const kind: EmitterKind = rng.next() < 0.5 ? 'tone' : 'burst'
      const lowHz = -fs / 2 + band * bandwidthHz
      const frequencyHz = lowHz + bandwidthHz * rng.uniform(0.15, 0.85)
      // In-band SNR is defined against the noise falling inside one sub-band (true noise).
      const amp = Math.sqrt((fromDb(snrDb) * noisePower) / numSubBands)
      const phase = rng.uniform(0, 2 * Math.PI)

      const gate = new Uint8Array(n).fill(1)
      let dutyCycle = 1
      let pulses = 0
      if (kind === 'burst') {
        pulses = rng.int(1, 4)
        dutyCycle = rng.uniform(0.2, 0.5)
        const pulseLen = Math.max(1, Math.floor((dutyCycle * n) / pulses))
        const period = Math.floor(n / pulses)
        gate.fill(0)
        for (let p = 0; p < pulses; p++) {
          const start = p * period + rng.int(0, Math.max(0, period - pulseLen))
          gate.fill(1, start, Math.min(n, start + pulseLen))
        }
        let on = 0
        for (let t = 0; t < n; t++) on += gate[t]
        dutyCycle = on / n
      }
      const omega = (2 * Math.PI * frequencyHz) / fs
      for (let t = 0; t < n; t++) {
        if (!gate[t]) continue
        samplesI[t] += amp * Math.cos(omega * t + phase)
        samplesQ[t] += amp * Math.sin(omega * t + phase)
      }
      emitters.push({ kind, band, frequencyHz, snrDb, dutyCycle, pulses })
    }

    return {
      id: `${split}/${snrDb >= 0 ? '+' : ''}${snrDb}dB/${String(index).padStart(3, '0')}`,
      split,
      snrDb,
      sampleRateHz: fs,
      numSubBands,
      samplesI,
      samplesQ,
      noiseOffsetDb,
      trueNoiseFloorDb: this.nominalNoiseFloorDb(fs) + noiseOffsetDb,
      emitters,
      occupiedBands: emitters.map((e) => e.band).sort((a, b) => a - b),
    }
  }

  // -------------------------- (c) Evaluator -------------------------------

  /**
   * Deterministic sub-band-level scoring. Every (window, sub-band) cell is a binary
   * decision: Pd = TP / occupied cells, Pfa = FP / vacant cells.
   */
  evaluate(fixtures: SignalWindowFixture[], predictions: WindowPrediction[]): EvaluationReport {
    const byId = new Map(predictions.map((p) => [p.windowId, p]))
    const blank = (snrDb: SliceMetrics['snrDb']): SliceMetrics => ({
      snrDb, windows: 0, occupiedTotal: 0, vacantTotal: 0,
      truePositives: 0, falsePositives: 0, missed: 0, pd: 0, pfa: 0,
    })
    const slices = new Map<SnrSlice, SliceMetrics>()
    const overall = blank('overall')

    for (const fx of fixtures) {
      const pred = byId.get(fx.id)
      if (!pred) throw new Error(`evaluate: missing prediction for window ${fx.id}`)
      const truth = new Set(fx.occupiedBands)
      const said = new Set(pred.occupiedBands)
      let slice = slices.get(fx.snrDb)
      if (!slice) slices.set(fx.snrDb, (slice = blank(fx.snrDb)))
      for (const m of [slice, overall]) {
        m.windows++
        for (let b = 0; b < fx.numSubBands; b++) {
          if (truth.has(b)) {
            m.occupiedTotal++
            if (said.has(b)) m.truePositives++
            else m.missed++
          } else {
            m.vacantTotal++
            if (said.has(b)) m.falsePositives++
          }
        }
      }
    }
    const finish = (m: SliceMetrics) => {
      m.pd = m.occupiedTotal ? m.truePositives / m.occupiedTotal : 0
      m.pfa = m.vacantTotal ? m.falsePositives / m.vacantTotal : 0
      return m
    }
    const splits = new Set(fixtures.map((f) => f.split))
    return {
      split: splits.size === 1 ? [...splits][0] : 'mixed',
      slices: [...slices.values()].sort((a, b) => (a.snrDb as number) - (b.snrDb as number)).map(finish),
      overall: finish(overall),
    }
  }
}

export default DspFixtureService
