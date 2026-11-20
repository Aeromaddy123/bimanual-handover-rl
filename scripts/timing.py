"""Exact film times for the narration: section bounds, checkpoints shown, and the hero's grab / pass / release / delivery."""
import json, numpy as np, mujoco
import video_aloha as VA
f = VA.Film("aloha1")
r = f.run.rec(f.final); h = f.hero; m = f.sc.m; d = mujoco.MjData(m)
box = r["qpos"][:, h, 16:19]; T = box.shape[0]
lgs, rgs = m.site("left/gripper").id, m.site("right/gripper").id
lg = np.zeros((T, 3)); rg = np.zeros((T, 3))
for t in range(T):
  d.qpos[:] = r["qpos"][t, h]; mujoco.mj_kinematics(m, d); lg[t] = d.site_xpos[lgs]; rg[t] = d.site_xpos[rgs]
dt = f.dt
ev = {
  "grab": int(np.argmax(box[:, 2] > 0.06)) * dt,
  "pass": int(np.argmax(box[:, 0] > -0.03)) * dt,
  "release": int(np.argmax((box[:, 0] > 0.0) & (np.linalg.norm(lg - box, axis=1) > 0.08))) * dt,
  "deliver": int(np.argmax(np.linalg.norm(box - r["target"][h], axis=1) < 0.05)) * dt,
}
segs, film = f.hero_warp()
def to_film(te):
  acc = 0.0
  for (e0, e1, sp), fl in zip(segs, film):
    if te <= e1: return acc + (te - e0) / sp
    acc += fl
  return acc
hero0 = float(f.bounds[4])
print(json.dumps({"bounds": [round(float(x), 2) for x in f.bounds], "k_steps": f.k_steps,
                  "k_minutes": [f.run.minutes.get(s) for s in f.k_steps], "k_success": [f.run.rate[s] for s in f.k_steps],
                  "k_passed": [f.run.passed[s] for s in f.k_steps], "per_ck": f.per_ck,
                  "hero_events_film_s": {k: round(hero0 + to_film(v), 2) for k, v in ev.items()},
                  "hero_events_episode_s": {k: round(v, 2) for k, v in ev.items()}}, indent=1))
