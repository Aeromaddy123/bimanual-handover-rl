"""Light ambient score, generated from scratch (no samples, no copyright): soft pad chords, a gentle pluck
pattern that builds while the robots learn, a sub bass, and a long synthetic reverb."""
import numpy as np, soundfile as sf

SR = 44100; TOTAL = 84.77; N = int(TOTAL * SR)
t = np.arange(N) / SR
rng = np.random.default_rng(3)
midi = lambda n: 440.0 * 2 ** ((n - 69) / 12)

# A minor: Am9 - Fmaj7 - C(add9) - G6, one chord every 5.3 s
CH = [[57, 60, 64, 67, 71], [53, 57, 60, 64, 69], [48, 55, 60, 62, 64], [55, 59, 62, 64, 67]]
BASS = [45, 41, 48, 43]
CHORD_LEN = 5.3


def lowpass(x, cutoff):
  X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR)
  return np.fft.irfft(X / (1 + (f / cutoff) ** 4), n=len(x))


def env(n, a, r):
  e = np.ones(n); na, nr = int(a * SR), int(r * SR)
  e[:na] = np.linspace(0, 1, na) ** 2; e[-nr:] *= np.linspace(1, 0, nr) ** 2
  return e


def section_level(x):  # dynamics follow the film: calm, building while learning, calm close-up, swell, resolve
  pts = [(0, .55), (8, .7), (15, .75), (30, .9), (40, 1.0), (50, .85), (58, .7), (71, .95), (79, .8), (84.77, .0)]
  xs, ys = zip(*pts); return np.interp(x, xs, ys)


pad = np.zeros((N, 2)); pluck = np.zeros((N, 2)); bass = np.zeros(N)
n_ch = int(np.ceil(TOTAL / CHORD_LEN))
for c in range(n_ch):
  s = int(c * CHORD_LEN * SR); L = min(int((CHORD_LEN + 2.0) * SR), N - s)
  if L <= 0: break
  tt = np.arange(L) / SR; chord = CH[c % 4]
  e = env(L, 1.6, 2.2)
  for n in chord:
    for ch, det in ((0, -0.08), (1, 0.08)):
      f = midi(n) * 2 ** (det / 12)
      w = sum(np.sin(2 * np.pi * f * k * tt + rng.uniform(0, 6.28)) / k ** 1.6 for k in range(1, 6))
      pad[s:s + L, ch] += w * e * 0.06
  fb = midi(BASS[c % 4]) / 2
  bass[s:s + L] += np.sin(2 * np.pi * fb * tt) * env(L, 0.8, 2.0) * 0.12
  # pluck arpeggio, 8th notes at 90 bpm, from the learning section on
  if 15 <= c * CHORD_LEN <= 80:
    step = 60 / 90 / 2
    for j in range(int(CHORD_LEN / step)):
      ts = c * CHORD_LEN + j * step; si = int(ts * SR)
      if si >= N: break
      n = chord[[0, 2, 4, 3, 1, 2, 4, 3][j % 8]] + 12
      d = min(int(1.2 * SR), N - si); td = np.arange(d) / SR
      note = (np.sin(2 * np.pi * midi(n) * td) + 0.3 * np.sin(2 * np.pi * 2 * midi(n) * td)) * np.exp(-td * 5.5)
      amp = 0.035 * (0.6 + 0.4 * (j % 2 == 0))
      pan = 0.5 + 0.3 * np.sin(j * 0.9)
      pluck[si:si + d, 0] += note * amp * (1 - pan); pluck[si:si + d, 1] += note * amp * pan

pad[:, 0] = lowpass(pad[:, 0], 1800); pad[:, 1] = lowpass(pad[:, 1], 1800)
mix = pad + pluck + bass[:, None]
# long reverb: exponentially decaying stereo noise impulse
ir_len = int(3.0 * SR); it = np.arange(ir_len) / SR
for ch in range(2):
  ir = rng.standard_normal(ir_len) * np.exp(-it / 0.9); ir = lowpass(ir, 6000); ir /= np.abs(ir).sum() ** 0.5 * 12
  wet = np.fft.irfft(np.fft.rfft(mix[:, ch], N + ir_len) * np.fft.rfft(ir, N + ir_len), n=N + ir_len)[:N]
  mix[:, ch] = mix[:, ch] * 0.75 + wet * 0.55
mix *= section_level(t)[:, None]
fade = np.ones(N); fi = int(2.0 * SR); fo = int(3.5 * SR)
fade[:fi] = np.linspace(0, 1, fi); fade[-fo:] = np.linspace(1, 0, fo) ** 1.5
mix *= fade[:, None]
mix /= np.abs(mix).max() / 0.8
sf.write("tts/music.wav", mix.astype(np.float32), SR)
print("music written", mix.shape, f"{N / SR:.1f}s")
