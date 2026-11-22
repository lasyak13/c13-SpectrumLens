/**
 * C13: SpectrumLens — Phase 1 entry point.
 *
 * Boots a Cordis Context, mounts `dsp` (DspFixtureService) and `detector`
 * (SignalWindowDetector, Gen-0 Baseline Energy Detector), evaluates Pd / Pfa per
 * SNR slice on the dev and held-out test splits, prints two signal-evidence
 * reports and exits.
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

async function main() {
  const ctx = new Context()
  const dspFiber = await ctx.plugin(DspFixtureService)

  // Generate the fixture set first so the detector's assumed noise floor and
  // sub-band layout come from the same configuration as the data.
  let fixtures!: FixtureSet
  await ctx.inject(['dsp'], (ctx) => {
    fixtures = ctx.dsp.generateFixtures()
  })
  const detectorFiber = await ctx.plugin(SignalWindowDetector, {
    ...baselineEnergyDetectorConfig(fixtures.nominalNoiseFloorDb),
    numSubBands: fixtures.options.numSubBands,
  })

  await ctx.inject(['dsp', 'detector'], (ctx) => {
    const { dsp, detector } = ctx
    const o = fixtures.options
    console.log('C13: SpectrumLens — Phase 1 / Generation 0 (First Demo)')
    console.log(
      `Fixtures: seed=0x${o.seed.toString(16)}, fs=${o.sampleRateHz / 1e6} MS/s, window=${o.windowLength} samples, ` +
        `${o.numSubBands} sub-bands, ${o.windowsPerSlice} windows/slice/split, noise drift ±${o.noiseDriftDb / 2} dB, ` +
        `nominal floor ${fixtures.nominalNoiseFloorDb.toFixed(2)} dB/Hz`,
    )
    console.log(`Detector: ${detector.config.generation} — ${JSON.stringify({
      taper: detector.config.taper,
      subWindows: detector.config.subWindows,
      noiseFloor: detector.config.noiseFloor.kind,
      threshold: detector.config.threshold,
    })}`)

    const devDet = detector.detectAll(fixtures.dev)
    const testDet = detector.detectAll(fixtures.test)
    printTable('Dev split — Pd / Pfa by SNR slice (sub-band level)', dsp.evaluate(fixtures.dev, devDet))
    printTable('Held-out test split — Pd / Pfa by SNR slice (sub-band level)', dsp.evaluate(fixtures.test, testDet))

    // Two sample evidence reports from the held-out set: a strong-SNR window with
    // emitters, and a low-SNR window illustrating baseline failure modes.
    const pick = (snr: number, pred: (i: number) => boolean) => {
      const i = fixtures.test.findIndex((f, i) => f.snrDb === snr && pred(i))
      return i >= 0 ? i : fixtures.test.findIndex((f) => f.snrDb === snr)
    }
    const strong = pick(10, (i) => fixtures.test[i].emitters.length >= 2)
    const weak = pick(-5, (i) => {
      const truth = fixtures.test[i].occupiedBands
      const said = testDet[i].occupiedBands
      return truth.length > 0 && said.join() !== truth.join()
    })
    for (const i of [strong, weak]) printEvidence(fixtures.test[i], testDet[i])
  })

  await detectorFiber.dispose()
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
