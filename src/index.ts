/**
 * C13: SpectrumLens — Phase 2 entry point (Self-Evolution).
 *
 * Boots a Cordis Context, mounts `dsp` (DspFixtureService) and the Gen-0 Baseline
 * Energy Detector as `detector`, prints the Phase-1 baseline, then mounts the
 * `SelfEvolver` plugin which runs the Gen 0 → Gen 3 failure-driven evolution loop of
 * the editable signal window (Gemini when GEMINI_API_KEY is set, otherwise a
 * deterministic optimizer), hot-swapping `ctx.detector` through the Cordis plugin
 * lifecycle behind a regression gate. Finishes with dev / held-out test tables, a
 * low-SNR before/after comparison and a signal-evidence report from the survivor.
 */
import { Context } from 'cordis'
import DspFixtureService, {
  type EvaluationReport,
  type FixtureSet,
  type SignalWindowFixture,
} from './plugins/dsp-fixtures.ts'
import SignalWindowDetector, {
  baselineEnergyDetectorConfig,
  type DetectionResult,
} from './plugins/window-detector.ts'
import SelfEvolver, { MAX_PFA, type EvolutionHistory } from './plugins/self-evolver.ts'

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

function printEvidence(fx: SignalWindowFixture, det: DetectionResult) {
  const ev = det.evidence
  const truth = new Set(fx.occupiedBands)
  console.log(`\n── Signal evidence report: ${ev.windowId} ` + '─'.repeat(Math.max(0, 50 - ev.windowId.length)))
  console.log(`  Detector      : ${ev.generation}`)
  console.log(
    `  Signal window : ${ev.samplesUsed} samples @ ${(ev.sampleRateHz / 1e6).toFixed(3)} MS/s, ` +
      `${ev.config.subWindows} sub-window(s), ${ev.config.taper} taper, FFT ${ev.fftSize} ` +
      `(${ev.binWidthHz.toFixed(1)} Hz/bin), ${ev.config.numSubBands} sub-bands`,
  )
  console.log(
    `  Noise floor   : ${ev.noiseFloorDb.toFixed(2)} dB/Hz (${ev.config.noiseFloor.kind}); ` +
      `true floor ${fx.trueNoiseFloorDb.toFixed(2)} dB/Hz (drift ${sdb(+fx.noiseOffsetDb.toFixed(2))} dB)`,
  )
  console.log(`  Threshold     : ${ev.thresholdDb.toFixed(2)} dB/Hz (${ev.config.threshold.kind})`)
  console.log(
    `  Ground truth  : ` +
      (fx.emitters.length
        ? fx.emitters
            .map(
              (e) =>
                `${e.kind} in band ${e.band} @ ${(e.frequencyHz / 1e3).toFixed(1)} kHz` +
                (e.kind === 'burst' ? ` (${e.pulses} pulse(s), duty ${(e.dutyCycle * 100).toFixed(0)}%)` : ''),
            )
            .join('; ')
        : 'no emitters'),
  )
  console.log(`  Verdict       : ${ev.narrative}`)
  const shown = ev.bands.filter((b) => b.occupied || truth.has(b.band))
  const rest = ev.bands.filter((b) => !b.occupied && !truth.has(b.band))
  for (const b of shown) {
    const tag = b.occupied ? (truth.has(b.band) ? 'HIT ' : 'FA  ') : 'MISS'
    console.log(`    [${tag}] ${b.summary}`)
  }
  if (rest.length) {
    const peak = rest.reduce((a, b) => (b.snrMarginDb > a.snrMarginDb ? b : a))
    console.log(
      `    [ ok ] ${rest.length} other band(s) correctly vacant; closest call: band ${peak.band} ` +
        `margin ${peak.snrMarginDb.toFixed(2)} dB (confidence ${(peak.confidence * 100).toFixed(1)}%)`,
    )
  }
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

function printLowSnr(title: string, before: EvaluationReport, after: EvaluationReport) {
  console.log(`\n${title}`)
  for (const s of after.slices) {
    const b = before.slices.find((x) => x.snrDb === s.snrDb)!
    console.log(
      `  ${`${sdb(s.snrDb as number)} dB`.padStart(7)}: Pd ${pct(b.pd)} → ${pct(s.pd)}  (Δ ${((s.pd - b.pd) * 100).toFixed(1).padStart(5)} pp)` +
        `   Pfa ${pct(b.pfa)} → ${pct(s.pfa)}`,
    )
  }
  console.log(
    `  Overall : Pd ${pct(before.overall.pd)} → ${pct(after.overall.pd)}   Pfa ${pct(before.overall.pfa)} → ${pct(after.overall.pfa)}`,
  )
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

  await once(['dsp', 'detector'], (ctx) => {
    const { dsp, detector } = ctx
    console.log(`\nSurvivor: ${detector.config.generation}`)
    console.log(`  ${JSON.stringify(detector.config)}`)
    const devDet = detector.detectAll(fixtures.dev)
    const testDet = detector.detectAll(fixtures.test)
    const finalDev = dsp.evaluate(fixtures.dev, devDet)
    const finalTest = dsp.evaluate(fixtures.test, testDet)
    printTable(`${detector.config.generation} — dev split (sub-band level)`, finalDev)
    printTable(`${detector.config.generation} — held-out test split (sub-band level)`, finalTest)
    printLowSnr('Gen-0 → survivor, dev split, per SNR slice:', gen0Dev, finalDev)
    printLowSnr('Gen-0 → survivor, held-out test split, per SNR slice:', gen0Test, finalTest)

    // Evidence report for a −10 dB held-out window the survivor fully resolves.
    const i = fixtures.test.findIndex((f, i) => f.snrDb === -10 && f.occupiedBands.length > 0 &&
      testDet[i].occupiedBands.join() === f.occupiedBands.join())
    if (i >= 0) printEvidence(fixtures.test[i], testDet[i])
  })

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
