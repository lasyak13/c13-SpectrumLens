/**
 * C13: SpectrumLens — entry point (Phases 1–3).
 *
 *  1. Boots a Cordis Context, mounts `dsp` (DspFixtureService) and the Gen-0 Baseline
 *     Energy Detector as `detector`.
 *  2. Mounts `SelfEvolver`, which evolves the editable signal window Gen 0 → Gen 3 on the
 *     dev split, hot-swapping `ctx.detector` behind a regression gate (Gemini when
 *     GEMINI_API_KEY is set, otherwise a deterministic optimizer).
 *  3. Mounts `LearnedSpectralDetector`, fits it on dev only, and benchmarks all three
 *     detectors on the held-out test split. Writes artifacts/benchmark_report.json and
 *     artifacts/sample_evidence.md.
 */
import { Context } from 'cordis'
import DspFixtureService, {
  type EvaluationReport,
  type FixtureSet,
} from './plugins/dsp-fixtures.ts'
import SignalWindowDetector, {
  baselineEnergyDetectorConfig,
} from './plugins/window-detector.ts'
import SelfEvolver, { MAX_PFA, type EvolutionHistory } from './plugins/self-evolver.ts'
import LearnedSpectralDetector, { FEATURE_NAMES } from './plugins/learned-detector.ts'
import {
  buildReport,
  markdownComparison,
  printComparison,
  renderSampleEvidence,
  writeText,
  type DetectorEntry,
} from './benchmark.ts'

const pct = (x: number) => `${(x * 100).toFixed(1)}%`.padStart(7)
const sdb = (x: number) => `${x >= 0 ? '+' : ''}${x}`

function printTable(title: string, report: EvaluationReport) {
  const rows = [...report.slices, report.overall].map((m) => [
    m.snrDb === 'overall' ? 'Overall' : `${sdb(m.snrDb)} dB`,
    String(m.windows),
    `${m.truePositives}/${m.occupiedTotal}`,
    pct(m.pd),
    `${m.falsePositives}/${m.vacantTotal}`,
    pct(m.pfa),
  ])
  const head = ['SNR slice', 'Windows', 'TP/Occupied', 'Pd', 'FP/Vacant', 'Pfa']
  const widths = head.map((h, i) => Math.max(h.length, ...rows.map((r) => r[i].length)))
  const line = (cells: string[]) => '│ ' + cells.map((c, i) => c.padStart(widths[i])).join(' │ ') + ' │'
  const rule = (l: string, m: string, r: string) => l + widths.map((w) => '─'.repeat(w + 2)).join(m) + r
  console.log(`\n${title}`)
  console.log(rule('┌', '┬', '┐'))
  console.log(line(head))
  console.log(rule('├', '┼', '┤'))
  rows.forEach((r, i) => {
    if (i === rows.length - 1) console.log(rule('├', '┼', '┤'))
    console.log(line(r))
  })
  console.log(rule('└', '┴', '┘'))
}

function printEvolution(h: EvolutionHistory) {
  const head = ['Gen', 'Active signal window', 'U(dev)', 'Pd -10', 'Pd -5', 'Pfa', 'Proposed candidate', 'U(cand)', 'Gate']
  const pd = (m: EvolutionHistory['generations'][number]['dev'], s: number) =>
    pct(m.slices.find((x) => x.snrDb === s)!.pd).trim()
  const rows = h.generations.map((g) => [
    String(g.generation),
    g.label,
    g.dev.utility.toFixed(3),
    pd(g.dev, -10),
    pd(g.dev, -5),
    pct(g.dev.pfa).trim(),
    g.candidate?.label ?? '— (survivor)',
    g.candidate ? g.candidate.dev.utility.toFixed(3) : '',
    g.candidate ? (g.candidate.decision === 'accepted' ? 'ACCEPT' : 'ROLLBACK') : '',
  ])
  const widths = head.map((x, i) => Math.max(x.length, ...rows.map((r) => r[i].length)))
  const line = (c: string[]) => '│ ' + c.map((x, i) => (i === 1 || i === 6 ? x.padEnd(widths[i]) : x.padStart(widths[i]))).join(' │ ') + ' │'
  const rule = (l: string, m: string, r: string) => l + widths.map((w) => '─'.repeat(w + 2)).join(m) + r
  console.log(`\nSelf-evolution summary (${h.mode} mode; U = Pd − 1.5·Pfa on dev, guardrail Pfa ≤ ${pct(MAX_PFA).trim()})`)
  console.log(rule('┌', '┬', '┐'))
  console.log(line(head))
  console.log(rule('├', '┼', '┤'))
  rows.forEach((r) => console.log(line(r)))
  console.log(rule('└', '┴', '┘'))
}

async function main() {
  const ctx = new Context()
  const dspFiber = await ctx.plugin(DspFixtureService)

  // Scoped injections are disposed right after use: a live `inject` callback would be
  // re-run by Cordis every time `ctx.detector` is hot-swapped during evolution.
  const once = async (deps: string[], fn: (ctx: Context) => unknown) => {
    const fiber = ctx.inject(deps, fn)
    await fiber
    await fiber.dispose()
  }

  let fixtures!: FixtureSet
  await once(['dsp'], (ctx) => {
    fixtures = ctx.dsp.generateFixtures()
  })
  const gen0Fiber = await ctx.plugin(SignalWindowDetector, {
    ...baselineEnergyDetectorConfig(fixtures.nominalNoiseFloorDb),
    numSubBands: fixtures.options.numSubBands,
  })

  const o = fixtures.options
  console.log('C13: SpectrumLens — Phase 2: Self-Evolution of the signal window')
  console.log(
    `Fixtures: seed=0x${o.seed.toString(16)}, fs=${o.sampleRateHz / 1e6} MS/s, window=${o.windowLength} samples, ` +
      `${o.numSubBands} sub-bands, ${o.windowsPerSlice} windows/slice/split, noise drift ±${o.noiseDriftDb / 2} dB, ` +
      `nominal floor ${fixtures.nominalNoiseFloorDb.toFixed(2)} dB/Hz`,
  )

  let gen0Dev!: EvaluationReport
  let gen0Test!: EvaluationReport
  await once(['dsp', 'detector'], (ctx) => {
    gen0Dev = ctx.dsp.evaluate(fixtures.dev, ctx.detector.detectAll(fixtures.dev))
    gen0Test = ctx.dsp.evaluate(fixtures.test, ctx.detector.detectAll(fixtures.test))
  })
  printTable('Gen-0 Baseline Energy Detector — dev split (sub-band level)', gen0Dev)

  // Mount the evolver (injects only `dsp`, so it survives detector hot-swaps) and evolve.
  console.log('')
  const evolverFiber = await ctx.plugin(SelfEvolver)
  let history!: EvolutionHistory
  await once(['evolver'], async (ctx) => {
    history = await ctx.evolver.evolve({
      fixtures,
      detectorFiber: gen0Fiber,
      log: (l) => console.log(l),
    })
  })
  printEvolution(history)

  // ---------------- Phase 3: learned detector vs. energy detection ----------------
  const learnedFiber = await ctx.plugin(LearnedSpectralDetector)
  await once(['dsp', 'detector', 'learned'], async (ctx) => {
    const { dsp, detector, learned } = ctx
    const model = learned.train(fixtures.dev) // dev only: train() rejects non-dev fixtures
    console.log(
      `\nLearned Multi-Feature Spectral Detector: L2 logistic regression on ${FEATURE_NAMES.length} features ` +
        `(${FEATURE_NAMES.join(', ')})`,
    )
    console.log(
      `  fitted on dev only: ${model.trainedOn.cells} cells (${model.trainedOn.positives} occupied), ` +
        `${model.training.iterations} GD iterations, final loss ${model.training.finalLoss.toFixed(4)}, ` +
        `threshold p > ${model.threshold.toFixed(3)} (dev Pd ${(model.training.devPd * 100).toFixed(1)}%, ` +
        `Pfa ${(model.training.devPfa * 100).toFixed(2)}%)`,
    )
    console.log(
      '  standardised weights: ' + FEATURE_NAMES.map((f, j) => `${f} ${model.weights[j] >= 0 ? '+' : ''}${model.weights[j].toFixed(2)}`).join(', '),
    )

    const baselineConfig = history.generations[0].config
    const evolvedConfig = detector.config
    const test = fixtures.test
    const baseline = detector.detectAll(test, baselineConfig) // Gen-0 signal window via override
    const evolved = detector.detectAll(test)
    const learnedDets = learned.detectAll(test)
    const entries: DetectorEntry[] = [
      {
        key: 'baseline',
        name: 'Baseline Energy',
        description: 'Gen 0 fixed-threshold energy detector (band-mean PSD, assumed floor, absolute threshold)',
        config: baselineConfig,
        report: dsp.evaluate(test, baseline),
      },
      {
        key: 'evolved',
        name: 'Self-Evolved Window',
        description: `Gen 3 survivor of the ${history.mode} self-evolution loop`,
        config: evolvedConfig,
        report: dsp.evaluate(test, evolved),
      },
      {
        key: 'learned',
        name: 'Learned Feature',
        description: 'L2 logistic regression on multi-dimensional per-sub-band spectral features, fitted on dev only',
        config: model,
        report: dsp.evaluate(test, learnedDets),
      },
    ]
    printComparison(entries)

    const comparisonMarkdown = markdownComparison(entries)
    await writeText('artifacts/benchmark_report.json', JSON.stringify(buildReport(fixtures, entries, history.mode), null, 2) + '\n')
    await writeText(
      'artifacts/sample_evidence.md',
      renderSampleEvidence({ test, baseline, evolved, learned: learnedDets, baselineConfig, evolvedConfig, model, comparisonMarkdown }),
    )
    console.log('\nArtifacts: artifacts/evolution_history.json, artifacts/benchmark_report.json, artifacts/sample_evidence.md')
    console.log('\nREADME table (Markdown):\n' + comparisonMarkdown)
  })

  await learnedFiber.dispose()
  await evolverFiber.dispose()
  await dspFiber.dispose()
  console.log('\nDone.')
}

main().then(
  () => process.exit(0),
  (err) => {
    console.error(err)
    process.exit(1)
  },
)
