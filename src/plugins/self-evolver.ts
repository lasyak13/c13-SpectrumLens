/**
 * SelfEvolver — Cordis plugin registered as `ctx.evolver` (injects `['dsp']` only).
 *
 * Phase 2: failure-driven self-evolution of the editable *signal window*.
 *
 * Generations Gen 0 … Gen 3. In every generation the active `ctx.detector` is
 * evaluated on the `dev` SNR slices (−10 … +10 dB) and structured failure traces are
 * collected (missed detections, false alarms on empty bands). In Gen 0 … Gen 2 those
 * traces are diagnosed and an upgraded `SignalWindowConfig` is proposed, either by
 *
 *   a) Gemini (`gemini-2.5-flash` over the REST API, native `fetch`) when
 *      `GEMINI_API_KEY` is set, or
 *   b) a deterministic failure-driven optimizer that reads the exact false-alarm and
 *      missed-detection residuals (no API, no cost, fully reproducible).
 *
 * Each candidate is hot-swapped through the Cordis plugin lifecycle: the incumbent
 * detector fiber is disposed (`await fiber.dispose()`), the candidate
 * `SignalWindowDetector` is mounted onto `ctx`, verified on `dev`, and kept only if
 * the composite utility `Pd − 1.5·Pfa` improves and `Pfa ≤ 0.10`; otherwise the
 * candidate is disposed and the previous generation's plugin is re-mounted (rollback).
 * Gen 3 is the survivor: it is evaluated and traced but not mutated further.
 *
 * Because the evolver injects only `dsp`, disposing the detector never unloads the
 * evolver itself — it can tear down and re-mount `ctx.detector` freely.
 */
import { mkdir, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import { Context, Service, type Fiber } from 'cordis'
import type {
  EvaluationReport,
  FixtureSet,
  SignalWindowFixture,
  SliceMetrics,
  SnrSlice,
  SubBand,
  TaperName,
} from './dsp-fixtures.ts'
import { fromDb } from './dsp-fixtures.ts'
import SignalWindowDetector, {
  type BandStatistic,
  type DetectionResult,
  type NoiseFloorMethod,
  type SignalWindowConfig,
} from './window-detector.ts'

declare module 'cordis' {
  interface Context {
    evolver: SelfEvolver
  }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface FailureTrace {
  windowId: string
  snrDb: SnrSlice
  band: number
  type: 'missed_detection' | 'false_alarm'
  /** Emitter that was missed (missed detections only). */
  emitter?: { kind: 'tone' | 'burst'; dutyCycle: number; pulses: number }
  /** True when the false alarm fired in a window that has no emitters at all. */
  emptyWindow: boolean
  /** True when the false alarm sits next to an occupied band (spectral leakage suspect). */
  adjacentToOccupied: boolean
  /** Statistic minus estimated floor, dB. */
  statisticSnrDb: number
  /** Statistic minus threshold, dB (residual: <0 for misses, >0 for false alarms). */
  marginDb: number
  /** Estimated minus true noise floor, dB. */
  floorErrorDb: number
  peakToMeanDb: number
}

export interface FailureSummary {
  missedDetections: number
  falseAlarms: number
  missedLowSnr: number
  falseAlarmsEmptyWindows: number
  falseAlarmsAdjacent: number
  missedBySlice: Record<string, number>
  falseAlarmsBySlice: Record<string, number>
  missedByKind: { tone: number; burst: number }
  /** RMS error of the estimated noise floor vs. truth across dev windows, dB. */
  floorRmsErrorDb: number
  /** Errors in windows whose floor estimate is off by more than 1 dB. */
  errorsInBiasedWindows: number
  /** Median peak-to-mean ratio of missed occupied bands vs. correctly vacant bands, dB. */
  missedPeakToMeanDb: number
  vacantPeakToMeanDb: number
  /** Std-dev of (statistic − floor) over vacant cells, dB. */
  vacantStatisticSpreadDb: number
  /** Median residual (margin) of missed detections, dB. */
  missedMedianMarginDb: number
}

export interface Metrics {
  pd: number
  pfa: number
  utility: number
  slices: { snrDb: SnrSlice; pd: number; pfa: number; tp: number; occupied: number; fp: number; vacant: number }[]
}

export interface Proposal {
  source: 'gemini' | 'heuristic'
  model?: string
  operator: string
  diagnosis: string[]
  rationale: string
  config: SignalWindowConfig
  /** Set when Gemini was requested but failed and the heuristic optimizer stepped in. */
  fallbackReason?: string
}

export interface GenerationRecord {
  generation: number
  label: string
  config: SignalWindowConfig
  dev: Metrics
  failures: FailureSummary
  /** Worst failure traces (largest |residual|), capped for the log. */
  sampleTraces: FailureTrace[]
  proposal?: Proposal
  candidate?: {
    label: string
    dev: Metrics
    decision: 'accepted' | 'rejected'
    reason: string
    hotSwap: string[]
  }
}

export interface EvolutionHistory {
  project: string
  phase: string
  generatedAt: string
  mode: 'gemini' | 'heuristic'
  objective: string
  guardrail: { maxPfa: number }
  fixtures: Omit<FixtureSet['options'], 'snrSlices'> & { snrSlices: number[] }
  generations: GenerationRecord[]
  final: { label: string; config: SignalWindowConfig; dev: Metrics; test: Metrics }
}

export interface EvolveOptions {
  fixtures: FixtureSet
  /** Fiber of the currently mounted Gen-0 `SignalWindowDetector`. Ownership passes to the evolver. */
  detectorFiber: Fiber
  generations?: number
  historyPath?: string
  log?: (line: string) => void
}

export const UTILITY_PFA_WEIGHT = 1.5
export const MAX_PFA = 0.1
/** Design target used when calibrating thresholds on dev, leaving headroom below the gate. */
export const DESIGN_PFA = 0.05
const LOW_SNR = -5
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash'
const TAPERS: TaperName[] = ['rectangular', 'hann', 'hamming', 'blackman']

const utilityOf = (pd: number, pfa: number) => pd - UTILITY_PFA_WEIGHT * pfa
const median = (xs: number[]) => {
  if (!xs.length) return NaN
  const s = [...xs].sort((a, b) => a - b)
  const m = s.length >> 1
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}
const std = (xs: number[]) => {
  const mu = xs.reduce((a, b) => a + b, 0) / xs.length
  return Math.sqrt(xs.reduce((a, b) => a + (b - mu) ** 2, 0) / xs.length)
}
const r3 = (x: number) => Math.round(x * 1000) / 1000
const pct = (x: number) => `${(x * 100).toFixed(1)}%`

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

export class SelfEvolver extends Service {
  static inject = ['dsp']

  /**
   * The evolver's own plugin context. Cordis service proxies rebind `this.ctx` to the
   * *caller's* context, so detectors are mounted onto this pinned context instead: their
   * lifetime then follows the evolver plugin, not whichever short-lived `inject` scope
   * happened to call `evolve()`.
   */
  readonly home: Context

  constructor(ctx: Context) {
    super(ctx, 'evolver')
    this.home = ctx
  }

  get mode(): 'gemini' | 'heuristic' {
    return process.env.GEMINI_API_KEY ? 'gemini' : 'heuristic'
  }

  // ------------------------------ Cordis lifecycle ------------------------------

  /** Mount a `SignalWindowDetector` onto ctx and wait until `ctx.detector` is live. */
  async mountDetector(config: SignalWindowConfig): Promise<Fiber> {
    const fiber = this.home.plugin(SignalWindowDetector, config)
    await fiber
    return fiber
  }

  /** Run `fn` against the live `ctx.detector` via a scoped injection. */
  async withDetector<T>(fn: (detector: SignalWindowDetector) => T): Promise<T> {
    let out!: T
    const probe = this.home.inject(['detector'], (ctx) => {
      out = fn(ctx.detector)
    })
    await probe
    await probe.dispose()
    return out
  }

  // ------------------------------ Evaluation & traces ------------------------------

  toMetrics(report: EvaluationReport): Metrics {
    const s = (m: SliceMetrics) => ({
      snrDb: m.snrDb as SnrSlice,
      pd: r3(m.pd),
      pfa: r3(m.pfa),
      tp: m.truePositives,
      occupied: m.occupiedTotal,
      fp: m.falsePositives,
      vacant: m.vacantTotal,
    })
    return {
      pd: r3(report.overall.pd),
      pfa: r3(report.overall.pfa),
      utility: r3(utilityOf(report.overall.pd, report.overall.pfa)),
      slices: report.slices.map(s),
    }
  }

  collectFailures(fixtures: SignalWindowFixture[], dets: DetectionResult[]) {
    const traces: FailureTrace[] = []
    const floorErrors: number[] = []
    const missedP2M: number[] = []
    const vacantP2M: number[] = []
    const vacantStat: number[] = []
    let errorsInBiasedWindows = 0
    fixtures.forEach((fx, i) => {
      const ev = dets[i].evidence
      const truth = new Set(fx.occupiedBands)
      const floorErrorDb = ev.noiseFloorDb - fx.trueNoiseFloorDb
      floorErrors.push(floorErrorDb)
      for (const b of ev.bands) {
        const occ = truth.has(b.band)
        if (!occ) vacantStat.push(b.snrDb)
        if (!occ && !b.occupied) vacantP2M.push(b.peakToMeanDb)
        if (occ === b.occupied) continue
        if (Math.abs(floorErrorDb) > 1) errorsInBiasedWindows++
        const em = fx.emitters.find((e) => e.band === b.band)
        if (occ) missedP2M.push(b.peakToMeanDb)
        traces.push({
          windowId: fx.id,
          snrDb: fx.snrDb,
          band: b.band,
          type: occ ? 'missed_detection' : 'false_alarm',
          ...(em ? { emitter: { kind: em.kind, dutyCycle: r3(em.dutyCycle), pulses: em.pulses } } : {}),
          emptyWindow: fx.emitters.length === 0,
          adjacentToOccupied: !occ && (truth.has(b.band - 1) || truth.has(b.band + 1)),
          statisticSnrDb: r3(b.snrDb),
          marginDb: r3(b.snrMarginDb),
          floorErrorDb: r3(floorErrorDb),
          peakToMeanDb: r3(b.peakToMeanDb),
        })
      }
    })
    const missed = traces.filter((t) => t.type === 'missed_detection')
    const fas = traces.filter((t) => t.type === 'false_alarm')
    const bySlice = (ts: FailureTrace[]) =>
      Object.fromEntries(
        [...new Set(fixtures.map((f) => f.snrDb))].map((s) => [`${s}dB`, ts.filter((t) => t.snrDb === s).length]),
      )
    const summary: FailureSummary = {
      missedDetections: missed.length,
      falseAlarms: fas.length,
      missedLowSnr: missed.filter((t) => t.snrDb <= LOW_SNR).length,
      falseAlarmsEmptyWindows: fas.filter((t) => t.emptyWindow).length,
      falseAlarmsAdjacent: fas.filter((t) => t.adjacentToOccupied).length,
      missedBySlice: bySlice(missed),
      falseAlarmsBySlice: bySlice(fas),
      missedByKind: {
        tone: missed.filter((t) => t.emitter?.kind === 'tone').length,
        burst: missed.filter((t) => t.emitter?.kind === 'burst').length,
      },
      floorRmsErrorDb: r3(Math.sqrt(floorErrors.reduce((a, b) => a + b * b, 0) / floorErrors.length)),
      errorsInBiasedWindows,
      missedPeakToMeanDb: r3(median(missedP2M)),
      vacantPeakToMeanDb: r3(median(vacantP2M)),
      vacantStatisticSpreadDb: r3(std(vacantStat)),
      missedMedianMarginDb: r3(median(missed.map((t) => t.marginDb))),
    }
    const sampleTraces = [...traces].sort((a, b) => Math.abs(b.marginDb) - Math.abs(a.marginDb)).slice(0, 24)
    return { traces, summary, sampleTraces }
  }

  // ------------------------------ Threshold calibration ------------------------------

  /**
   * Exact residual-driven threshold optimisation. For a fixed signal-window structure the
   * per-cell residual r = statistic − floor determines every decision (occupied ⇔ r > margin),
   * so sweeping the margin across the sorted residuals of occupied (miss-side) and vacant
   * (false-alarm-side) cells finds the utility-optimal margin subject to Pfa ≤ DESIGN_PFA.
   */
  calibrateMargin(fixtures: SignalWindowFixture[], dets: DetectionResult[]) {
    const cells: { r: number; occ: boolean }[] = []
    fixtures.forEach((fx, i) => {
      const truth = new Set(fx.occupiedBands)
      for (const b of dets[i].evidence.bands) cells.push({ r: b.snrDb, occ: truth.has(b.band) })
    })
    cells.sort((a, b) => b.r - a.r)
    const nOcc = cells.filter((c) => c.occ).length
    const nVac = cells.length - nOcc
    let tp = 0
    let fp = 0
    let best = { marginDb: cells[0].r + 0.01, pd: 0, pfa: 0, utility: 0 }
    for (let i = 0; i < cells.length; i++) {
      if (cells[i].occ) tp++
      else fp++
      if (i + 1 < cells.length && cells[i + 1].r === cells[i].r) continue
      const pd = tp / nOcc
      const pfa = fp / nVac
      if (pfa > DESIGN_PFA) break
      const u = utilityOf(pd, pfa)
      if (u > best.utility) {
        const next = i + 1 < cells.length ? cells[i + 1].r : cells[i].r - 0.02
        best = { marginDb: r3((cells[i].r + next) / 2), pd, pfa, utility: u }
      }
    }
    return best
  }

  // ------------------------------ Heuristic optimizer ------------------------------

  async proposeHeuristic(
    fixtures: SignalWindowFixture[],
    current: SignalWindowConfig,
    f: FailureSummary,
    nextGen: number,
  ): Promise<Proposal> {
    const cfg: SignalWindowConfig = structuredClone(current)
    const binsPerBand = 2 ** Math.floor(Math.log2(fixtures[0].samplesI.length / cfg.subWindows)) / cfg.numSubBands
    const noiseP2M = 10 * Math.log10(Math.log(binsPerBand) + 0.5772) // E[max of n Exp(1)] ≈ ln n + γ
    const concentrationDb = f.missedPeakToMeanDb - f.vacantPeakToMeanDb
    const leakageShare = f.falseAlarms ? f.falseAlarmsAdjacent / f.falseAlarms : 0

    // Failure-mode catalogue: each candidate operator is scored by how many dev errors it is
    // directly attributable to; the dominant one is fixed this generation.
    const ops: { name: string; score: number; diagnosis: string; apply: () => string }[] = []

    if (cfg.noiseFloor.kind === 'assumed' && f.floorRmsErrorDb > 1) {
      ops.push({
        name: 'adaptive-noise-floor',
        score: f.errorsInBiasedWindows,
        diagnosis:
          `Noise-floor mismatch: assumed floor is off by ${f.floorRmsErrorDb.toFixed(2)} dB RMS (receiver drift); ` +
          `${f.errorsInBiasedWindows} of ${f.missedDetections + f.falseAlarms} errors occur in windows biased by >1 dB ` +
          `(false alarms cluster on hot windows, misses on cold windows).`,
        apply: () => {
          // Pick the robust estimator with the lowest floor error on dev.
          const methods: NoiseFloorMethod[] = [
            { kind: 'median' },
            { kind: 'percentile', percentile: 25 },
            { kind: 'percentile', percentile: 37.5 },
          ]
          const scored = methods.map((m) => ({ m, rms: this.floorRmsFor(fixtures, m) }))
          scored.sort((a, b) => a.rms - b.rms)
          cfg.noiseFloor = scored[0].m
          cfg.threshold = { kind: 'relative', marginDb: 1 }
          return (
            `Replace the assumed floor with a per-window ${scored[0].m.kind === 'median' ? 'median' : `P${(scored[0].m as any).percentile}`} ` +
            `sub-band noise-floor estimate (floor RMS error ${scored.map((s) => `${s.m.kind === 'median' ? 'median' : 'P' + (s.m as any).percentile}=${s.rms.toFixed(2)} dB`).join(', ')}) ` +
            `and express the threshold as a margin above that floor (CFAR-style), so drift cancels out.`
          )
        },
      })
    }
    if ((cfg.bandStatistic ?? 'mean') === 'mean' && concentrationDb > 3) {
      ops.push({
        name: 'narrowband-peak-statistic',
        score: f.missedDetections,
        diagnosis:
          `Narrowband energy dilution: missed occupied bands show a peak-to-mean ratio of ${f.missedPeakToMeanDb.toFixed(1)} dB ` +
          `vs ${f.vacantPeakToMeanDb.toFixed(1)} dB on vacant bands (+${concentrationDb.toFixed(1)} dB), i.e. the carrier ` +
          `is concentrated in a few of the ${binsPerBand} bins and averaging the whole band throws that gain away. ` +
          `${f.missedLowSnr} misses are at ≤${LOW_SNR} dB.`,
        apply: () => {
          cfg.bandStatistic = 'peak'
          cfg.threshold = { kind: 'relative', marginDb: r3(noiseP2M + 3) }
          return (
            `Switch the band statistic from band-mean to peak-bin PSD (a matched statistic for narrowband carriers, ` +
            `≈${(10 * Math.log10(binsPerBand)).toFixed(0)} dB of processing gain at −10 dB in-band SNR) and recalibrate the margin ` +
            `above the extreme-value noise level (E[max] ≈ ${noiseP2M.toFixed(1)} dB over ${binsPerBand} bins).`
          )
        },
      })
    }
    if (cfg.bandStatistic === 'peak' && cfg.subWindows === 1 && f.missedLowSnr > 0) {
      ops.push({
        name: 'multi-window-integration',
        score: f.missedLowSnr,
        diagnosis:
          `Statistic variance: a single-bin periodogram is χ²₂-distributed (≈5.6 dB std), giving a vacant-cell spread of ` +
          `${f.vacantStatisticSpreadDb.toFixed(2)} dB that forces a high margin; ${f.missedLowSnr} low-SNR misses sit a median ` +
          `${Math.abs(f.missedMedianMarginDb).toFixed(2)} dB below threshold.`,
        apply: () => {
          cfg.subWindows = 4
          cfg.integration = 'average'
          if (leakageShare > 0.3) cfg.taper = 'hann'
          return (
            `Integrate the signal window as 4 Welch sub-windows (variance ↓ by √4) while keeping enough per-bin ` +
            `coherent gain for tones; ` +
            (leakageShare > 0.3
              ? `apply a Hann taper because ${pct(leakageShare)} of false alarms sit next to occupied bands (leakage).`
              : `keep the rectangular taper: only ${pct(leakageShare)} of false alarms are adjacent to occupied bands, so leakage ` +
                `is not a failure mode here and Hann/Hamming would cost ~1.8/1.3 dB of ENBW on tone peaks.`)
          )
        },
      })
    }
    if (cfg.taper === 'rectangular' && leakageShare > 0.3 && f.falseAlarmsAdjacent >= 5) {
      ops.push({
        name: 'leakage-taper',
        score: f.falseAlarmsAdjacent,
        diagnosis: `Spectral leakage: ${f.falseAlarmsAdjacent} of ${f.falseAlarms} false alarms are adjacent to occupied bands.`,
        apply: () => {
          cfg.taper = 'hann'
          return `Switch the rectangular taper to Hann (−31 dB first sidelobe, −18 dB/oct roll-off) to stop leakage into neighbours.`
        },
      })
    }

    ops.sort((a, b) => b.score - a.score)
    const op = ops[0]
    const diagnosis = ops.map((o) => `[${o.name}, ${o.score} errors] ${o.diagnosis}`)
    let action = op
      ? op.apply()
      : 'No structural failure mode left above trigger level; only re-tune the detection threshold.'
    cfg.generation = `Gen-${nextGen} ${op ? op.name : 'threshold-retune'}`

    // Calibrate the detection margin on the exact residuals of the candidate structure.
    if (cfg.threshold.kind === 'relative') {
      const dets = await this.withDetector((d) => d.detectAll(fixtures, cfg))
      const cal = this.calibrateMargin(fixtures, dets)
      cfg.threshold = { kind: 'relative', marginDb: cal.marginDb }
      action +=
        ` Threshold tuned on the miss/false-alarm residuals to margin ${cal.marginDb.toFixed(2)} dB ` +
        `(calibration Pd ${pct(cal.pd)}, Pfa ${pct(cal.pfa)} under a ${pct(DESIGN_PFA)} design cap).`
    }
    if (!diagnosis.length) diagnosis.push('No dominant failure mode; residual threshold retune only.')
    return { source: 'heuristic', operator: op?.name ?? 'threshold-retune', diagnosis, rationale: action, config: cfg }
  }

  private floorRmsFor(fixtures: SignalWindowFixture[], method: NoiseFloorMethod): number {
    // Re-estimate each window's floor from its band means under the candidate estimator.
    const errs: number[] = []
    const dets = this.lastDetections ?? []
    fixtures.forEach((fx, i) => {
      const ev = dets[i]?.evidence
      if (!ev) return
      const bands = ev.bands.map((b) => ({ meanPsd: fromDb(b.bandMeanPsdDb) }) as SubBand)
      errs.push(this.estimateFloor(bands, method) - fx.trueNoiseFloorDb)
    })
    return Math.sqrt(errs.reduce((a, b) => a + b * b, 0) / Math.max(1, errs.length))
  }

  private estimateFloor(bands: SubBand[], method: NoiseFloorMethod): number {
    // Same estimator as SignalWindowDetector.estimateNoiseFloorDb (kept stateless here so
    // it can be scored while no detector is mounted).
    if (method.kind === 'assumed') return method.floorDb
    const sorted = bands.map((b) => b.meanPsd).sort((a, b) => a - b)
    const p = method.kind === 'median' ? 50 : method.percentile
    const pos = (Math.min(100, Math.max(0, p)) / 100) * (sorted.length - 1)
    const lo = Math.floor(pos)
    const hi = Math.ceil(pos)
    return 10 * Math.log10(sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo))
  }

  private lastDetections?: DetectionResult[]

  // ------------------------------ Gemini optimizer ------------------------------

  async proposeGemini(
    fixtures: SignalWindowFixture[],
    current: SignalWindowConfig,
    record: GenerationRecord,
    history: GenerationRecord[],
    nextGen: number,
  ): Promise<Proposal> {
    const prompt = [
      'You are an RF/DSP engineer evolving a spectrum-occupancy energy detector.',
      'The editable unit is the "signal window": a 4096-sample complex baseband window at 1 MS/s, split into',
      `${current.numSubBands} equal sub-bands; each sub-band is declared occupied when its statistic exceeds a threshold.`,
      'Emitters are narrowband tones and pulsed bursts (duty 20-50%) at in-band SNR -10..+10 dB; the noise floor drifts ±3 dB per window.',
      `Objective: maximise utility = Pd - ${UTILITY_PFA_WEIGHT}*Pfa on dev, with a hard guardrail Pfa <= ${MAX_PFA}.`,
      '',
      'SignalWindowConfig fields you may set:',
      '  taper: "rectangular" | "hann" | "hamming" | "blackman"',
      '  subWindows: 1 | 2 | 4 | 8   (non-overlapping sub-windows integrated into the PSD)',
      '  integration: "average" | "max-hold"   (how sub-window periodograms are combined)',
      '  bandStatistic: "mean" | "peak"   (band-averaged PSD vs. largest single bin in the band)',
      '  noiseFloor: {"kind":"assumed","floorDb":number} | {"kind":"median"} | {"kind":"percentile","percentile":number}',
      '  threshold: {"kind":"absolute","thresholdDb":number} | {"kind":"relative","marginDb":number}   (margin above the estimated floor)',
      '  floorUncertaintyDb: number',
      'Set "calibrateThreshold": true to have the tool pick the utility-optimal relative margin exactly from the dev residuals',
      `(subject to Pfa <= ${DESIGN_PFA}); recommended unless you have a specific reason to fix the margin yourself.`,
      '',
      `Current config (generation ${record.generation}):`,
      JSON.stringify(current),
      'Current dev metrics:',
      JSON.stringify(record.dev),
      'Failure summary:',
      JSON.stringify(record.failures),
      'Worst failure traces (marginDb = statistic - threshold):',
      JSON.stringify(record.sampleTraces.slice(0, 16)),
      'Previous generations (config, utility, decision):',
      JSON.stringify(
        history.map((h) => ({
          generation: h.generation,
          config: h.config,
          utility: h.dev.utility,
          proposal: h.proposal && { operator: h.proposal.operator, config: h.proposal.config },
          decision: h.candidate && { decision: h.candidate.decision, utility: h.candidate.dev.utility },
        })),
      ),
      '',
      'Diagnose the dominant failure mode and propose ONE upgraded config. Do not re-propose a rejected config.',
      'Reply with JSON only: {"operator": string, "diagnosis": string[], "rationale": string, "calibrateThreshold": boolean, "config": SignalWindowConfig}',
    ].join('\n')

    const url = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-goog-api-key': process.env.GEMINI_API_KEY! },
      body: JSON.stringify({
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: { responseMimeType: 'application/json', temperature: 0.2 },
      }),
      signal: AbortSignal.timeout(90_000),
    })
    if (!res.ok) throw new Error(`Gemini HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`)
    const body: any = await res.json()
    const text: string = (body?.candidates?.[0]?.content?.parts ?? [])
      .map((p: any) => p.text ?? '')
      .join('')
      .trim()
    if (!text) throw new Error(`Gemini returned no text (finishReason ${body?.candidates?.[0]?.finishReason ?? 'n/a'})`)
    const json = JSON.parse(text.replace(/^```(?:json)?\s*|\s*```$/g, ''))
    const cfg = this.sanitize(json.config ?? {}, current)
    const operator = String(json.operator ?? 'gemini-update').slice(0, 64)
    cfg.generation = `Gen-${nextGen} ${operator}`
    let rationale = String(json.rationale ?? '')
    if (json.calibrateThreshold && cfg.threshold.kind === 'relative') {
      const dets = await this.withDetector((d) => d.detectAll(fixtures, cfg))
      const cal = this.calibrateMargin(fixtures, dets)
      cfg.threshold = { kind: 'relative', marginDb: cal.marginDb }
      rationale += ` [tool] Margin calibrated on dev residuals to ${cal.marginDb.toFixed(2)} dB.`
    }
    return {
      source: 'gemini',
      model: GEMINI_MODEL,
      operator,
      diagnosis: Array.isArray(json.diagnosis) ? json.diagnosis.map(String) : [String(json.diagnosis ?? '')],
      rationale,
      config: cfg,
    }
  }

  /** Validate and clamp an untrusted (LLM-produced) config onto the current one. */
  sanitize(raw: any, current: SignalWindowConfig): SignalWindowConfig {
    const cfg = structuredClone(current)
    const num = (x: any, lo: number, hi: number, d: number) =>
      typeof x === 'number' && Number.isFinite(x) ? Math.min(hi, Math.max(lo, x)) : d
    if (TAPERS.includes(raw.taper)) cfg.taper = raw.taper
    if ([1, 2, 4, 8].includes(raw.subWindows)) cfg.subWindows = raw.subWindows
    if (raw.integration === 'average' || raw.integration === 'max-hold') cfg.integration = raw.integration
    if (raw.bandStatistic === 'mean' || raw.bandStatistic === 'peak') cfg.bandStatistic = raw.bandStatistic as BandStatistic
    const nf = raw.noiseFloor
    if (nf?.kind === 'median') cfg.noiseFloor = { kind: 'median' }
    else if (nf?.kind === 'percentile') cfg.noiseFloor = { kind: 'percentile', percentile: num(nf.percentile, 1, 99, 25) }
    else if (nf?.kind === 'assumed') cfg.noiseFloor = { kind: 'assumed', floorDb: num(nf.floorDb, -120, 0, -60) }
    const th = raw.threshold
    if (th?.kind === 'relative') cfg.threshold = { kind: 'relative', marginDb: num(th.marginDb, 0, 30, 3) }
    else if (th?.kind === 'absolute') cfg.threshold = { kind: 'absolute', thresholdDb: num(th.thresholdDb, -120, 0, -57) }
    cfg.floorUncertaintyDb = num(raw.floorUncertaintyDb, 0, 6, cfg.floorUncertaintyDb)
    cfg.windowLength = 0
    return cfg
  }

  // ------------------------------ Evolution loop ------------------------------

  async evolve(options: EvolveOptions): Promise<EvolutionHistory> {
    const { fixtures } = options
    const total = options.generations ?? 4
    const log = options.log ?? ((l: string) => this.ctx.logger('evolver').info(l))
    const historyPath = options.historyPath ?? 'artifacts/evolution_history.json'
    const dev = fixtures.dev
    let fiber = options.detectorFiber
    let incumbent = await this.withDetector((d) => structuredClone(d.config))
    const generations: GenerationRecord[] = []
    const mode = this.mode
    log(`Self-evolution mode: ${mode === 'gemini' ? `Gemini (${GEMINI_MODEL}, REST)` : 'deterministic failure-driven optimizer (no API key)'}`)

    const evaluate = async () => {
      const dets = await this.withDetector((d) => d.detectAll(dev))
      return { dets, metrics: this.toMetrics(this.home.dsp.evaluate(dev, dets)) }
    }

    for (let g = 0; g < total; g++) {
      // 1. Evaluate the active ctx.detector and collect structured failure traces.
      const { dets, metrics } = await evaluate()
      this.lastDetections = dets
      const { summary, sampleTraces } = this.collectFailures(dev, dets)
      const record: GenerationRecord = {
        generation: g,
        label: incumbent.generation,
        config: structuredClone(incumbent),
        dev: metrics,
        failures: summary,
        sampleTraces,
      }
      generations.push(record)
      log(
        `Gen ${g} [${incumbent.generation}] dev Pd ${pct(metrics.pd)} Pfa ${pct(metrics.pfa)} U ${metrics.utility.toFixed(3)} — ` +
          `${summary.missedDetections} misses (${summary.missedLowSnr} at ≤${LOW_SNR} dB), ` +
          `${summary.falseAlarms} false alarms (${summary.falseAlarmsEmptyWindows} in empty windows)`,
      )
      if (g === total - 1) break

      // 2. Diagnose and propose an upgraded signal window.
      let proposal: Proposal
      if (mode === 'gemini') {
        try {
          proposal = await this.proposeGemini(dev, incumbent, record, generations.slice(0, -1), g + 1)
        } catch (err) {
          const reason = (err as Error).message
          log(`  Gemini proposal failed (${reason}); falling back to the deterministic optimizer.`)
          proposal = await this.proposeHeuristic(dev, incumbent, summary, g + 1)
          proposal.fallbackReason = reason
        }
      } else {
        proposal = await this.proposeHeuristic(dev, incumbent, summary, g + 1)
      }
      record.proposal = proposal
      for (const d of proposal.diagnosis) log(`  diagnose: ${d}`)
      log(`  propose (${proposal.source}/${proposal.operator}): ${proposal.rationale}`)

      // 3. Cordis hot-swap: dispose the incumbent fiber, mount the candidate onto ctx.
      const hotSwap: string[] = []
      await fiber.dispose()
      hotSwap.push(`disposed fiber "${incumbent.generation}"`)
      const candidateFiber = await this.mountDetector(proposal.config)
      hotSwap.push(`mounted candidate "${proposal.config.generation}" onto ctx.detector`)

      // 4. Verify on dev and apply the regression gate.
      const cand = (await evaluate()).metrics
      const improves = cand.utility > metrics.utility
      const safe = cand.pfa <= MAX_PFA
      const accepted = improves && safe
      const reason = accepted
        ? `utility ${metrics.utility.toFixed(3)} → ${cand.utility.toFixed(3)} (+${(cand.utility - metrics.utility).toFixed(3)}), Pfa ${pct(cand.pfa)} ≤ ${pct(MAX_PFA)}`
        : !safe
          ? `Pfa guardrail violated: ${pct(cand.pfa)} > ${pct(MAX_PFA)}`
          : `no utility gain: ${cand.utility.toFixed(3)} ≤ ${metrics.utility.toFixed(3)}`
      if (accepted) {
        fiber = candidateFiber
        incumbent = structuredClone(proposal.config)
        hotSwap.push('kept candidate')
      } else {
        await candidateFiber.dispose()
        hotSwap.push(`disposed candidate "${proposal.config.generation}"`)
        fiber = await this.mountDetector(incumbent)
        hotSwap.push(`rolled back: re-mounted "${incumbent.generation}"`)
      }
      record.candidate = {
        label: proposal.config.generation,
        dev: cand,
        decision: accepted ? 'accepted' : 'rejected',
        reason,
        hotSwap,
      }
      log(`  gate: ${accepted ? 'ACCEPT' : 'REJECT + ROLLBACK'} — ${reason}`)
    }

    const testMetrics = this.toMetrics(
      this.home.dsp.evaluate(fixtures.test, await this.withDetector((d) => d.detectAll(fixtures.test))),
    )
    const last = generations[generations.length - 1]
    const { snrSlices, ...fxOpts } = fixtures.options
    const history: EvolutionHistory = {
      project: 'C13 SpectrumLens',
      phase: 'Phase 2: Self-Evolution',
      generatedAt: new Date().toISOString(),
      mode,
      objective: `maximise Pd - ${UTILITY_PFA_WEIGHT}*Pfa on dev (sub-band level)`,
      guardrail: { maxPfa: MAX_PFA },
      fixtures: { ...fxOpts, snrSlices: [...snrSlices] },
      generations,
      final: { label: last.label, config: last.config, dev: last.dev, test: testMetrics },
    }
    await mkdir(dirname(historyPath), { recursive: true })
    await writeFile(historyPath, JSON.stringify(history, null, 2) + '\n')
    log(`Evolution history written to ${historyPath}`)
    this.detectorFiber = fiber
    return history
  }

  /** Fiber of the surviving detector after `evolve()` (owned by the evolver's context). */
  detectorFiber?: Fiber
}

export default SelfEvolver
