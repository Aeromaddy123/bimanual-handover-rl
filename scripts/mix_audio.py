"""Mix narrator + music: music ducks under the voice, then both go to the final loudness in ffmpeg."""
import numpy as np, soundfile as sf
v, sv = sf.read("tts/narration.wav"); m, sm = sf.read("tts/music.wav")
SR = 44100
x = np.arange(int(len(v) * SR / sv)) / SR
v = np.interp(x, np.arange(len(v)) / sv, v)                  # 24 kHz voice -> 44.1 kHz
n = min(len(v), len(m)); v = v[:n]; m = m[:n]
# voice envelope (50 ms window, smoothed) drives a 9 dB duck on the music
win = int(0.05 * SR); e = np.sqrt(np.convolve(v ** 2, np.ones(win) / win, mode="same"))
e = e / max(e.max(), 1e-6)
att = np.convolve((e > 0.04).astype(float), np.ones(int(0.35 * SR)) / int(0.35 * SR), mode="same")
duck = 1.0 - 0.65 * np.clip(att, 0, 1)
music_level, voice_level = 0.25, 1.0
out = m * (music_level * duck)[:, None] + (v * voice_level)[:, None] * np.array([1.0, 1.0])
out /= np.abs(out).max() / 0.95
sf.write("tts/final_mix.wav", out.astype(np.float32), SR)
print("mixed", out.shape)
