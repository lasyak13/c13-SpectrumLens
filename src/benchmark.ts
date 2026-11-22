/**
 * Phase 3 benchmark: Gen-0 energy detector vs. Gen-3 self-evolved signal window vs.
 * learned multi-feature detector on the held-out `test` split, plus the sample
 * signal-evidence report.
 */
import { mkdir, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import type { EvaluationReport, FixtureSet, SignalWindowFixture, SliceMetrics } from './plugins/dsp-fixtures.ts'
import type { DetectionResult, SignalWindowConfig } from './plugins/window-detector.ts'
import { FEATURE_NAMES, type LearnedDetection, type LearnedModel } from './plugins/learned-detector.ts'

export const PFA_WEIGHT = 1.5
const r4 = (x: number) => Math.round(x * 1e4) / 1e4
const pct = (x: number) => `${(x * 100).toFixed(1)}%`
const sdb = (x: number) => `${x >= 0 ? '+' : ''}${x}`
const kHz = (hz: number) => `${(hz / 1e3).toFixed(1)} kHz`
const sgn = (x: number, d = 2) => `${x >= 0 ? '+' : ''}${x.toFixed(d)}`

export interface DetectorEntry {
  key: 'baseline' | 'evolved' | 'learned'
  name: string
  description: string
  config: unknown
  report: EvaluationReport
}

function sliceJson(m: SliceMetrics) {
  return {
    snrDb: m.snrDb,
    windows: m.windows,
    truePositives: m.truePositives,
    occupied: m.occupiedTotal,
    missed: m.missed,
    falsePositives: m.falsePositives,
    vacant: m.vacantTotal,
    pd: r4(m.pd),
    pfa: r4(m.pfa),
    utility: r4(m.pd - PFA_WEIGHT * m.pfa),
  }
}

export function buildReport(fixtures: FixtureSet, entries: DetectorEntry[], evolutionMode: string) {
  const { snrSlices, ...opts } = fixtures.options
  return {
    project: 'C13 SpectrumLens',
    phase: 'Phase 3: Graduate Extension: learned detector vs. energy detection',
    generatedAt: new Date().toISOString(),
    evaluationSplit: 'test',
    trainingAndTuningSplit: 'dev',
    leakage: 'none: evolution, threshold calibration, feature standardisation and model fitting use dev only',
    scoring: 'per (window, sub-band) cell; Pd = TP/occupied, Pfa = FP/vacant',
    utility: `Pd - ${PFA_WEIGHT} * Pfa`,
    evolutionMode,
    fixtures: { ...opts, snrSlices: [...snrSlices] },
    detectors: entries.map((e) => ({
      key: e.key,
      name: e.name,
      description: e.description,
      config: e.config,
      slices: e.report.slices.map(sliceJson),
      overall: sliceJson(e.report.overall),
    })),
  }
}

export function printComparison(entries: DetectorEntry[]) {
  const head = ['SNR slice', ...entries.flatMap((e) => [`${e.name} Pd`, 'Pfa'])]
  const rows = [...entries[0].report.slices.map((s) => s.snrDb), 'overall' as const].map((snr) => {
    const get = (e: DetectorEntry) =>
      snr === 'overall' ? e.report.overall : e.report.slices.find((s) => s.snrDb === snr)!
    return [snr === 'overall' ? 'Overall' : `${sdb(snr as number)} dB`, ...entries.flatMap((e) => [pct(get(e).pd), pct(get(e).pfa)])]
  })
  const util = ['Utility', ...entries.flatMap((e) => [(e.report.overall.pd - PFA_WEIGHT * e.report.overall.pfa).toFixed(3), ''])]
  const widths = head.map((h, i) => Math.max(h.length, ...[...rows, util].map((r) => r[i].length)))
  const line = (c: string[]) => '│ ' + c.map((x, i) => x.padStart(widths[i])).join(' │ ') + ' │'
  const rule = (l: string, m: string, r: string) => l + widths.map((w) => '─'.repeat(w + 2)).join(m) + r
  console.log('\nHeld-out TEST split: Pd / Pfa per SNR slice (sub-band level; utility = Pd − 1.5·Pfa)')
  console.log(rule('┌', '┬', '┐'))
  console.log(line(head))
  console.log(rule('├', '┼', '┤'))
  rows.forEach((r, i) => {
    if (i === rows.length - 1) console.log(rule('├', '┼', '┤'))
    console.log(line(r))
  })
  console.log(line(util))
  console.log(rule('└', '┴', '┘'))
}

/** README-ready Markdown version of the comparison table. */
export function markdownComparison(entries: DetectorEntry[]): string {
  const head = `| SNR slice | ${entries.map((e) => `${e.name} Pd | ${e.name} Pfa`).join(' | ')} |`
  const sep = `|---|${entries.map(() => '---:|---:').join('|')}|`
  const row = (label: string, pick: (e: DetectorEntry) => SliceMetrics) =>
    `| ${label} | ${entries.map((e) => `${pct(pick(e).pd)} | ${pct(pick(e).pfa)}`).join(' | ')} |`
  const lines = entries[0].report.slices.map((s) =>
    row(`${sdb(s.snrDb as number)} dB`, (e) => e.report.slices.find((x) => x.snrDb === s.snrDb)!),
  )
  lines.push(row('**Overall**', (e) => e.report.overall))
  lines.push(
    `| **Utility** | ${entries.map((e) => `${(e.report.overall.pd - PFA_WEIGHT * e.report.overall.pfa).toFixed(3)} | `).join(' | ')} |`,
  )
  return [head, sep, ...lines].join('\n')
}

// ---------------------------------------------------------------------------
// Sample evidence (Markdown)
// ---------------------------------------------------------------------------

export interface EvidenceInputs {
  test: SignalWindowFixture[]
  baseline: DetectionResult[]
  evolved: DetectionResult[]
  learned: LearnedDetection[]
  baselineConfig: SignalWindowConfig
  evolvedConfig: SignalWindowConfig
  model: LearnedModel
  comparisonMarkdown: string
}

function pickExamples(x: EvidenceInputs) {
  const { test } = x
  const exact = (dets: { occupiedBands: number[] }[], i: number) => dets[i].occupiedBands.join() === test[i].occupiedBands.join()
  const find = (title: string, pred: (f: SignalWindowFixture, i: number) => boolean) => {
    const i = test.findIndex(pred)
    return i >= 0 ? { title, i } : null
  }
  return [
    find('Low SNR (−10 dB): continuous narrowband tone, missed by the energy detector', (f, i) =>
      f.snrDb === -10 && f.emitters.length === 1 && f.emitters[0].kind === 'tone' && !exact(x.baseline, i) && exact(x.evolved, i)),
    find('Low SNR (−10 dB): pulsed burst, a hard case', (f) =>
      f.snrDb === -10 && f.emitters.some((e) => e.kind === 'burst')),
    find('Low SNR (−5 dB): multiple emitters', (f, i) => f.snrDb === -5 && f.emitters.length >= 2 && exact(x.learned, i)),
    find('Empty band, hot receiver: false alarm from noise-floor drift', (f, i) =>
      f.emitters.length === 0 && f.noiseOffsetDb > 1.5 && x.baseline[i].occupiedBands.length > 0),
    find('High SNR (+10 dB): multiple emitters, tone and burst', (f) =>
      f.snrDb === 10 && f.emitters.length >= 2 && new Set(f.emitters.map((e) => e.kind)).size === 2),
  ].filter((e): e is { title: string; i: number } => e !== null)
}

export function renderSampleEvidence(x: EvidenceInputs): string {
  const out: string[] = []
  const m = x.model
  out.push('# SpectrumLens: sample signal evidence (held-out `test` windows)')
  out.push('')
  out.push(
    'Each example is one recorded 4096-sample complex baseband window (1 MS/s, 16 sub-bands of 62.5 kHz) from the ' +
      'held-out `test` split. For every relevant sub-band, SpectrumLens reports *why* each detector declared it ' +
      'occupied or vacant: the measured PSD statistic, the noise floor it was compared against, the margin to the ' +
      'threshold and a confidence. The learned detector reports its posterior and the per-feature contributions ' +
      'to its logit. Generated by `npm start`; numbers match `artifacts/benchmark_report.json`.',
  )
  out.push('')
  out.push('Detectors:')
  out.push(
    `- **Baseline Energy (Gen 0)**: band-mean PSD, rectangular taper, single periodogram, fixed threshold ` +
      `${(x.baselineConfig.threshold as any).thresholdDb.toFixed(2)} dB/Hz on an assumed floor.`,
  )
  out.push(
    `- **Self-Evolved Window (Gen 3)**: \`${JSON.stringify({
      bandStatistic: x.evolvedConfig.bandStatistic,
      subWindows: x.evolvedConfig.subWindows,
      integration: x.evolvedConfig.integration,
      taper: x.evolvedConfig.taper,
      noiseFloor: x.evolvedConfig.noiseFloor,
      threshold: x.evolvedConfig.threshold,
    })}\``,
  )
  out.push(
    `- **Learned Feature Detector**: L2 logistic regression on ${FEATURE_NAMES.length} spectral features, fitted on ` +
      `${m.trainedOn.cells} dev cells (${m.trainedOn.positives} occupied); decision threshold p > ${m.threshold.toFixed(3)}.`,
  )
  out.push('')
  out.push('## Held-out benchmark')
  out.push('')
  out.push(x.comparisonMarkdown)

  pickExamples(x).forEach(({ title, i }, n) => {
    const fx = x.test[i]
    const b0 = x.baseline[i].evidence
    const ev = x.evolved[i].evidence
    const ln = x.learned[i]
    const truth = new Set(fx.occupiedBands)
    const flagged = new Set([...fx.occupiedBands, ...b0.occupiedBands, ...ev.occupiedBands, ...ln.occupiedBands])
    const bands = [...flagged].sort((a, b) => a - b)
    const verdict = (said: number[]) => {
      const s = new Set(said)
      const hits = fx.occupiedBands.filter((b) => s.has(b)).length
      const fas = said.filter((b) => !truth.has(b)).length
      return `${hits}/${fx.occupiedBands.length} detected, ${fas} false alarm(s)`
    }

    out.push('')
    out.push(`## ${n + 1}. ${title}: \`${fx.id}\``)
    out.push('')
    out.push(
      `- **Ground truth:** ` +
        (fx.emitters.length
          ? fx.emitters
              .map(
                (e) =>
                  `${e.kind} in band ${e.band} @ ${kHz(e.frequencyHz)}, in-band SNR ${sdb(e.snrDb)} dB` +
                  (e.kind === 'burst' ? ` (${e.pulses} pulse(s), duty ${(e.dutyCycle * 100).toFixed(0)}%)` : ''),
              )
              .join('; ')
          : 'no emitters (all 16 sub-bands vacant)'),
    )
    out.push(
      `- **Receiver noise floor:** true ${fx.trueNoiseFloorDb.toFixed(2)} dB/Hz (drift ${sgn(fx.noiseOffsetDb)} dB). ` +
        `Gen 0 assumes ${b0.noiseFloorDb.toFixed(2)} dB/Hz; Gen 3 estimates ${ev.noiseFloorDb.toFixed(2)} dB/Hz (median); ` +
        `learned uses ${ln.noiseFloorDb.toFixed(2)} dB/Hz (median).`,
    )
    out.push(
      `- **Outcome:** Baseline ${verdict(b0.occupiedBands)} · Self-Evolved ${verdict(ev.occupiedBands)} · ` +
        `Learned ${verdict(ln.occupiedBands)}`,
    )
    out.push(`- **SpectrumLens explanation (Gen 3):** ${ev.narrative}`)
    out.push('')
    if (bands.length) {
      out.push(
        '| Band | Center | Truth | Baseline: band-mean PSD → margin | Self-Evolved: peak-bin PSD − floor → margin (conf.) | Learned: p (top feature contributions) |',
      )
      out.push('|---:|---:|:---:|---|---|---|')
      for (const b of bands) {
        const e0 = b0.bands[b]
        const e3 = ev.bands[b]
        const lb = ln.bands[b]
        const tag = (occ: boolean) => (occ ? (truth.has(b) ? '✅ HIT' : '⚠️ FA') : truth.has(b) ? '❌ MISS' : 'vacant')
        const top = Object.entries(lb.contributions)
          .sort((p, q) => Math.abs(q[1]) - Math.abs(p[1]))
          .slice(0, 3)
          .map(([k, v]) => `${k} ${sgn(v)}`)
          .join(', ')
        out.push(
          `| ${b} | ${kHz(e3.centerFrequencyHz)} | ${truth.has(b) ? 'occupied' : 'vacant'} | ` +
            `${tag(e0.occupied)}: ${e0.measuredPsdDb.toFixed(2)} dB/Hz → ${sgn(e0.snrMarginDb)} dB | ` +
            `${tag(e3.occupied)}: ${e3.measuredPsdDb.toFixed(2)} dB/Hz, SNR ${sgn(e3.snrDb)} dB → ${sgn(e3.snrMarginDb)} dB (${(e3.confidence * 100).toFixed(0)}%) | ` +
            `${tag(lb.occupied)}: p=${lb.probability.toFixed(3)} (${top}) |`,
        )
      }
      out.push('')
      const fb = fx.occupiedBands[0] ?? ln.occupiedBands[0] ?? b0.occupiedBands[0]
      if (fb !== undefined) {
        const f = ln.bands[fb].features
        out.push(
          `Feature vector for band ${fb}: ` +
            FEATURE_NAMES.map((k) => `\`${k}\`=${f[k].toFixed(k === 'spectralEntropy' || k === 'persistence' ? 3 : 2)}`).join(', ') +
            '.',
        )
        out.push('')
      }
    }
    const rest = ev.bands.filter((e) => !flagged.has(e.band))
    if (rest.length) {
      const close = rest.reduce((a, c) => (c.snrMarginDb > a.snrMarginDb ? c : a))
      out.push(
        `The other ${rest.length} sub-bands are vacant and every detector declared them vacant. Gen 3's closest call was band ` +
          `${close.band} (${kHz(close.centerFrequencyHz)}) with a margin of ${sgn(close.snrMarginDb)} dB to threshold.`,
      )
    }
  })
  out.push('')
  return out.join('\n')
}

export async function writeText(path: string, text: string) {
  await mkdir(dirname(path), { recursive: true })
  await writeFile(path, text)
}
