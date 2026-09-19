# Robot voice — a channel vocoder from scratch

Turns a speech recording into a robotic voice by analysing it with a bank of
band-pass filters and resynthesising it from pure sine waves. No effects
library — the filter bank, the envelope extraction and the synthesis are all
written directly.

```bash
pip install -r requirements.txt
python robot_voice.py          # reads quote.wav, writes robotic_voice.wav
```

## How it works

This is a **channel vocoder**, the same idea behind the classic robot-voice
effect and, historically, behind speech compression.

The insight is that speech is well described as a *source* (the buzz of the
vocal folds, or the hiss of a fricative) shaped by a *filter* (the resonances
of the vocal tract, which move as you articulate). Intelligibility lives almost
entirely in the filter — in how energy is distributed across frequency over
time — not in the source. So if you measure the energy in each frequency band
over time and re-impose it on a completely different source, the words survive
and the voice does not.

```
speech in
  ↓  resample to 16 kHz, take one channel
  ↓  split into 15 ms frames, 50 % overlap
  ↓  for each frame, for each band:
  ↓      band-pass filter  ->  RMS energy         ANALYSIS
  ↓      drive a sine at the band's centre        SYNTHESIS
  ↓  sum the sines, overlap-add the frames
robot out
```

Each band's RMS is the *envelope* — how loud that part of the spectrum is right
now. Replacing the original content of the band with a steady sine at the band
centre keeps that envelope and throws away everything else: the pitch, the
harmonic structure, the phase. What is left is the formant motion, which is
what carries the words. The monotone comes from every sine being at a fixed
frequency, so the resynthesised voice has no pitch contour at all.

The script also plots the original and synthesised signals in time and
frequency, so you can see the harmonic structure of the original replaced by
the discrete comb of synthesis tones.

## Known rough edges

Left as-is because the output is the intended effect, but worth knowing if you
build on it:

- **250 bands of 100 Hz each, across an 8 kHz band.** That is roughly 3x
  overlapped coverage — every part of the spectrum is counted by several bands
  at once, which is why the result needs an explicit `amplification_factor` to
  come out at a sensible level. A conventional vocoder uses 16–32 bands, often
  spaced logarithmically to match how hearing resolves frequency. Fewer, wider
  bands would sound more like a vocoder and less like a swarm.
- **Overlap-add without a window.** Frames overlap 50 % but are summed
  unwindowed, so the overlapping regions are counted twice and the boundaries
  are discontinuous. A Hann window with 50 % overlap sums to unity and is the
  standard fix.
- **Synthesis sines restart at phase 0 every frame.** That discontinuity at
  each frame boundary contributes a buzz — arguably part of the charm here, but
  it is an artefact rather than a choice. Carrying phase across frames removes
  it.

## Files

| | |
|---|---|
| `robot_voice.py` | The whole vocoder |
| `quote.wav` | Input sample |
| `robotic_voice.wav` | Example output |

## License

MIT — see [LICENSE](LICENSE).
