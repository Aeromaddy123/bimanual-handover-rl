"""Generate the narrator track with Kokoro (voice am_michael), each line placed at its film time."""
import json, numpy as np, soundfile as sf
from kokoro_onnx import Kokoro

SR = 24000
TOTAL = 84.77
LINES = [  # (start seconds, text)
  (1.0,  "This is a simulated robot cell. Two arms. One red block."),
  (8.6,  "To learn fast, we copy it, thousands of times. Each copy practises on its own."),
  (16.2, "At first, nothing works. They flail, they miss, they knock the block over."),
  (22.8, "Their only feedback is a reward, whenever they get closer to the goal than ever before."),
  (32.8, "About twenty-five minutes in, the first handovers start to happen."),
  (37.6, "And then, almost suddenly, they get it."),
  (43.6, "Eight thousand copies. One shared brain."),
  (50.8, "Nine attempts at the same moment. Most deliver. Some only get halfway."),
  (58.5, "And here's the skill they learned."),
  (61.3, "Pick it up."),
  (63.4, "Pass it over."),
  (64.9, "Let go."),
  (68.2, "Deliver."),
  (72.0, "Fifty million practice steps. Forty-one minutes, on a single GPU."),
  (79.8, "That's reinforcement learning."),
]
k = Kokoro("tts/kokoro-v1.0.onnx", "tts/voices-v1.0.bin")
track = np.zeros(int(TOTAL * SR) + SR, dtype=np.float32)
report = []
for i, (t0, text) in enumerate(LINES):
  audio, sr = k.create(text, voice="am_michael", speed=0.97, lang="en-us")
  assert sr == SR
  audio = np.asarray(audio, dtype=np.float32)
  nz = np.where(np.abs(audio) > 0.01)[0]                      # trim leading/trailing silence
  audio = audio[max(nz[0] - 240, 0): nz[-1] + 1200]
  s = int(t0 * SR); track[s:s + len(audio)] += audio
  end = t0 + len(audio) / SR
  nxt = LINES[i + 1][0] if i + 1 < len(LINES) else TOTAL
  report.append({"t": t0, "end": round(end, 2), "gap_to_next": round(nxt - end, 2), "text": text})
track = track[:int(TOTAL * SR)]
track /= max(1e-6, np.abs(track).max()) / 0.9
sf.write("tts/narration.wav", track, SR)
print(json.dumps(report, indent=1))
