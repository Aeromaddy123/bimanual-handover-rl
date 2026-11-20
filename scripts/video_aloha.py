"""LinkedIn video (1080x1350, 60 fps) for the two-arm handover learned in DeepMind's AlohaHandOver task.

Every frame is rendered from real recordings of the trained checkpoints (see rollout_all_pg.py).
Usage: video_aloha.py <run> --start F --end F --out part.mp4  |  --still SECONDS --out frame.png  |  --info
"""
import argparse, json, math, os, pathlib, subprocess, sys
import numpy as np, mujoco
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import video as V                      # fonts, easing, overlays, camera helper (shared look)
from video import W, H, FPS, Cam, ramp, smooth, ease, lerp, clip01, finish, draw_label, draw_stats, draw_curve, fmt_steps
from video import INK, MUTED, ACCENT, F_SMALL, F_SUB, F_LABEL, F_TITLE

ROOT = pathlib.Path(__file__).parent
ROWS, COLS, DX, DY = 9, 7, 1.55, 1.05      # 63 cells, one at the origin


def theme(m):
  """Dark studio, graphite arms with a little sheen so they read against the background."""
  for t in range(m.ntex):
    adr, w, h, c = m.tex_adr[t], m.tex_width[t], m.tex_height[t], m.tex_nchannel[t]; n = w * h * c
    if m.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
      m.tex_data[adr:adr + n] = 9
    elif m.texture(t).name == "groundplane":
      img = m.tex_data[adr:adr + n].reshape(h, w, c)
      m.tex_data[adr:adr + n] = np.where(img > 128, 26, 40).astype(np.uint8).ravel()
  blk = m.material("black").id
  m.mat_rgba[blk] = [0.30, 0.31, 0.33, 1]; m.mat_specular[blk] = 0.6; m.mat_shininess[blk] = 0.7
  m.mat_reflectance[m.material("groundplane").id] = 0.12
  m.vis.headlight.ambient[:] = 0.16; m.vis.headlight.diffuse[:] = 0.5; m.vis.headlight.specular[:] = 0.15
  m.vis.map.fogstart, m.vis.map.fogend = 4.0, 14.0
  m.vis.rgba.fog[:] = [0.035, 0.037, 0.042, 1]
  m.vis.map.zfar = 60
  floor = m.geom("floor").id
  m.geom_group[floor] = 5                                 # drawn once, not per copy
  for name in ("left_thresh", "right_thresh"):
    m.geom_group[m.geom(name).id] = 4                     # helper planes: never drawn
  m.geom_rgba[m.geom("floor").id] = [1, 1, 1, 1]


class Scene:
  def __init__(self):
    from pg_tools import load_env
    self.env = load_env("AlohaHandOver")
    self.m = self.env.mj_model; theme(self.m)
    self.m.vis.global_.offwidth, self.m.vis.global_.offheight = 2160, 2700
    self.d = mujoco.MjData(self.m)
    self.r = mujoco.Renderer(self.m, H, W, max_geom=20000)
    # the wide grid shots get their own model copy whose fog starts far away (fog scales with model extent)
    import copy
    self.mg = copy.copy(self.m)
    self.mg.vis.map.fogstart, self.mg.vis.map.fogend = 16.0, 42.0
    self.mg.mat_specular[self.mg.material("groundplane").id] = 0.0
    self.mg.mat_reflectance[self.mg.material("groundplane").id] = 0.05
    self.rg = mujoco.Renderer(self.mg, H, W, max_geom=20000)
    self.opt_all = mujoco.MjvOption(); self.opt_all.geomgroup[5] = 1; self.opt_all.geomgroup[1] = 0
    self.opt_copy = mujoco.MjvOption(); self.opt_copy.geomgroup[5] = 0; self.opt_copy.geomgroup[1] = 0
    self.pert = mujoco.MjvPerturb()
    self.home = self.m.keyframe("home").qpos.copy()
    self.mocap = self.m.body("mocap_target").mocapid[0]
    self.floor = self.m.geom("floor").id
    gb = self.m.geom_bodyid
    self.geom_body = gb
    xs = np.array([[c * DX - (COLS - 1) * DX / 2, r * DY - (ROWS - 1) * DY / 2, 0.0] for r in range(ROWS) for c in range(COLS)])
    self.centers = xs
    self.center_cell = int(np.argmin(np.linalg.norm(xs, axis=1)))

  def _set(self, q, target):
    self.d.qpos[:] = q; self.d.mocap_pos[self.mocap] = target
    mujoco.mj_kinematics(self.m, self.d); mujoco.mj_camlight(self.m, self.d)

  def cell(self, q, target, cam, edit=None):
    self._set(q, target)
    self.r.update_scene(self.d, cam.mj(), scene_option=self.opt_all)
    self.r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
    if edit: edit(self.r.scene)
    return self.r.render()

  def grid(self, qs, targets, cam, alpha=None, show=None):
    """All copies composed into one scene: geoms of each copy are added and shifted to its cell."""
    c0 = self.center_cell
    self._set(qs[c0], targets[c0])
    self.rg.update_scene(self.d, cam.mj(), scene_option=self.opt_all)
    sc = self.rg.scene; sc.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
    for i in range(len(self.centers)):
      if i == c0 or (show is not None and not show[i]): continue
      a = 1.0 if alpha is None else float(alpha[i])
      if a <= 0.01: continue
      n0 = sc.ngeom
      self._set(qs[i], targets[i])
      mujoco.mjv_addGeoms(self.mg, self.d, self.opt_copy, self.pert, mujoco.mjtCatBit.mjCAT_ALL, sc)
      off = self.centers[i]
      for g in range(n0, sc.ngeom):
        geom = sc.geoms[g]; geom.pos[:] = geom.pos + off
        if a < 1.0:
          geom.rgba[3] *= a; geom.pos[2] -= (1 - a) * 0.3
    return self.rg.render()


def sample_q(qpos, t, dt):
  f = clip01(t / (dt * (qpos.shape[0] - 1))) * (qpos.shape[0] - 1)
  i = int(min(math.floor(f), qpos.shape[0] - 2)); u = f - i
  q = (1 - u) * qpos[i] + u * qpos[i + 1]
  quat = q[..., 19:23]; q[..., 19:23] = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
  return q


class Run:
  def __init__(self, name):
    self.dir = ROOT / "runs" / name; rec = self.dir / "recordings"
    self.summary = json.loads((rec / "summary.json").read_text())
    self.steps = [s["step"] for s in self.summary]
    self.rate = {s["step"]: s["success_rate"] for s in self.summary}
    self.passed = {s["step"]: s["handover_rate"] for s in self.summary}
    self.minutes = {json.loads(l)["step"]: json.loads(l)["minutes"] for l in open(self.dir / "progress.jsonl")}
    self.rec_dir = rec; self._c = {}
  def rec(self, s):
    if s not in self._c: self._c[s] = dict(np.load(self.rec_dir / f"step_{s:012d}.npz"))
    return self._c[s]
  def pick(self, k):
    idx = np.unique(np.round(np.linspace(0, len(self.steps) - 1, k)).astype(int))
    return [self.steps[i] for i in idx]


class Film:
  def __init__(self, run):
    self.run = Run(run); self.sc = Scene()
    st = self.run.steps
    self.k_steps = self.run.pick(8); self.final = st[-1]
    fin = self.run.rec(self.final)
    ok = np.where(fin["success"])[0]; self.hero = int(ok[0]) if len(ok) else 0
    self.mid = min(st, key=lambda s: abs(self.run.rate[s] - 0.45))
    self.dt = float(fin["dt"]); self.ep = self.dt * (fin["qpos"].shape[0] - 1)
    self.T_BUILD, self.T_CLONE, self.per_ck, self.xf = 8.0, 7.0, 4.4, 0.7
    self.T_LEARN = self.per_ck * len(self.k_steps)
    self.T_SPLIT, self.T_HERO, self.T_FINALGRID, self.T_END = 8.0, 12.0, 8.0, 5.5
    self.T_HERO = float(sum(self.hero_warp()[1])) + 0.6
    self.bounds = np.cumsum([0, self.T_BUILD, self.T_CLONE, self.T_LEARN, self.T_SPLIT, self.T_HERO, self.T_FINALGRID, self.T_END])
    self.total = float(self.bounds[-1])
    self.cell_cam = Cam([0, 0, 0.12], 1.55, 110, -24)
    self.high_cam = Cam([0, 0, 0.0], 12.5, 125, -42)
    self.split_tile = None

  def frame(self, t):
    b = self.bounds
    for i, f in enumerate([self.f_build, self.f_clone, self.f_learn, self.f_split, self.f_hero, self.f_finalgrid, self.f_end]):
      if t < b[i + 1] or i == 6: return f(t - b[i])

  def f_build(self, t):
    m = self.sc.m; gb = self.sc.geom_body
    times = {0: 1.0}
    for i in range(1, 11): times[i] = 2.1 + (i - 1) * 0.2
    for i in range(11, 21): times[i] = 4.1 + (i - 11) * 0.16
    times[m.body("box").id] = 5.9; times[m.body("mocap_target").id] = 6.5
    floor = self.sc.floor
    def edit(sc):
      for g in range(sc.ngeom):
        geom = sc.geoms[g]
        if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM or geom.objid == floor: continue
        b = int(gb[geom.objid]); tb = times.get(b, 1.0)
        dur = 0.9 if b == 0 else (0.7 if b == m.body("box").id else 0.55)
        u = ramp(t, tb, tb + dur)
        if u <= 0.0:
          geom.pos[2] -= 50.0                      # not there yet: no shadow either
          continue
        geom.rgba[3] *= u
        geom.pos[2] += (1 - u) * (0.3 if b == m.body("box").id else (0.0 if b == 0 else 0.04))
    cam = Cam([-0.1, 0, 0.1], 2.0, 70, -14).mix(self.cell_cam, ease(t / self.T_BUILD))
    q = self.sc.home.copy()
    img = finish(self.sc.cell(q, np.array([0.2, 0, 0.25]), cam, edit), fade=ramp(t, 0, 1.2), top_shade=True)
    return draw_label(img, "01", "The scene", "Two robot arms, one block to pass between them.", ramp(t, 1.6, 2.4) * (1 - ramp(t, 7.2, 7.9)))

  def f_clone(self, t):
    sc = self.sc; dist = np.linalg.norm(sc.centers[:, :2], axis=1)
    alpha = np.array([ramp(t, 0.6 + 0.3 * dd, 1.2 + 0.3 * dd) for dd in dist])
    r0 = self.run.rec(self.k_steps[0])
    q = np.repeat(sc.home[None], len(sc.centers), 0); q[:, 16:23] = r0["qpos"][0][:, 16:23]
    cam = self.cell_cam.mix(self.high_cam, ease(t / self.T_CLONE))
    img = finish(sc.grid(q, r0["target"], cam, alpha), top_shade=True)
    return draw_label(img, "02", "Massively parallel", "63 copies shown. Training ran 8,192 at once on one GPU.",
                      ramp(t, 1.0, 1.8) * (1 - ramp(t, 6.2, 6.9)))

  def learn_cam(self, t):
    u = t / self.T_LEARN
    return Cam([lerp(0, 0.3, u), lerp(0, 0.2, u), lerp(0.0, 0.1, u)], lerp(12.5, 6.0, ease(u)), lerp(125, 200, u), lerp(-42, -28, ease(u)))

  def grid_frame(self, s, tep, cam):
    r = self.run.rec(s); return self.sc.grid(sample_q(r["qpos"], tep, self.dt), r["target"], cam)

  def f_learn(self, t):
    k = min(int(t // self.per_ck), len(self.k_steps) - 1); tl = t - k * self.per_ck
    speed = self.ep / (self.per_ck - 0.2); cam = self.learn_cam(t)
    arr = self.grid_frame(self.k_steps[k], tl * speed, cam)
    if k > 0 and tl < self.xf:
      prev = self.grid_frame(self.k_steps[k - 1], (tl + self.per_ck) * speed, cam)
      u = smooth(tl / self.xf); arr = (prev * (1 - u) + arr * u).astype(np.uint8)
    sp = self.k_steps[k - 1] if k > 0 else 0; sn = self.k_steps[k]; u = smooth(tl / self.per_ck) if k > 0 else 1.0
    steps = lerp(sp, sn, u); mins = lerp(self.run.minutes.get(sp, 0), self.run.minutes.get(sn, 0), u)
    succ = lerp(self.run.rate[sp] if k > 0 else self.run.rate[sn], self.run.rate[sn], u)
    a = ramp(t, 0.3, 1.2) * (1 - ramp(t, self.T_LEARN - 0.8, self.T_LEARN))
    img = finish(arr, top_shade=True)
    img = draw_stats(img, steps, mins, succ, a); img = draw_curve(img, self.run, steps, a)
    return draw_label(img, "03", "Learning by trial and error", "Paid only for progress: closer than ever before.",
                      ramp(t, 0.6, 1.5) * (1 - ramp(t, 6.0, 7.0)))

  def f_split(self, t):
    if self.split_tile is None:
      self.split_tile = mujoco.Renderer(self.sc.m, 450, 360)
    r = self.run.rec(self.mid); tep = clip01((t - 0.3) / (self.T_SPLIT - 1.2)) * self.ep
    q = sample_q(r["qpos"], tep, self.dt); canvas = np.zeros((H, W, 3), np.uint8)
    cam = Cam([0, 0, 0.12], 1.7, 105, -24).mj()
    for i in range(9):
      rr, cc = divmod(i, 3)
      self.sc._set(q[i], r["target"][i])
      self.split_tile.update_scene(self.sc.d, cam, scene_option=self.sc.opt_all)
      self.split_tile.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
      canvas[rr * 450:(rr + 1) * 450, cc * 360:(cc + 1) * 360] = self.split_tile.render()
    canvas[:, 359:361] = 12; canvas[:, 719:721] = 12; canvas[449:451] = 12; canvas[899:901] = 12
    img = finish(canvas, fade=ramp(t, 0, 0.5) * (1 - ramp(t, self.T_SPLIT - 0.4, self.T_SPLIT)))
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    ta = ramp(t, self.T_SPLIT - 2.2, self.T_SPLIT - 1.6) * (1 - ramp(t, self.T_SPLIT - 0.4, self.T_SPLIT))
    for i in range(9):
      rr, cc = divmod(i, 3); ok = bool(r["success"][i]); ps = bool(r["passed"][i])
      txt = "delivered" if ok else ("passed over" if ps else "missed")
      dr.text((cc * 360 + 18, rr * 450 + 410), txt, font=F_SMALL, fill=(ACCENT if ok else (INK if ps else MUTED)) + (int(255 * ta),))
    img = Image.alpha_composite(img, ov)
    return draw_label(img, "", "Nine attempts, same moment", f"Mid-training: {self.run.rate[self.mid]*100:.0f}% of attempts succeed.",
                      ramp(t, 0.5, 1.3) * (1 - ramp(t, self.T_SPLIT - 0.6, self.T_SPLIT)))

  def hero_warp(self):
    r = self.run.rec(self.final); h = self.hero
    box_x = r["qpos"][:, h, 16]
    ti = int(np.argmax(box_x > -0.03)) if (box_x > -0.03).any() else len(box_x) // 2
    t_hand = ti * self.dt
    a0, a1 = max(t_hand - 0.7, 0.0), min(t_hand + 0.5, self.ep)          # episode window shown in slow motion
    segs = [(0.0, a0, 0.5), (a0, a1, 0.2), (a1, self.ep, 0.6)]           # (episode start, end, playback speed)
    film = [(e1 - e0) / sp for e0, e1, sp in segs]
    return segs, film

  def f_hero(self, t):
    r = self.run.rec(self.final); h = self.hero
    segs, film = self.hero_warp()
    tep, acc, slow = self.ep, 0.0, False
    for (e0, e1, sp), fl in zip(segs, film):
      if t < acc + fl:
        tep = e0 + (t - acc) * sp; slow = sp < 0.3; break
      acc += fl
    q = sample_q(r["qpos"], min(tep, self.ep), self.dt)[h]
    u = t / self.T_HERO
    cam = Cam([lerp(-0.08, 0.08, ease(u)), 0, lerp(0.12, 0.16, ease(u))], lerp(1.75, 1.4, ease(u)), lerp(125, 80, ease(u)), lerp(-18, -24, u))
    img = finish(self.sc.cell(q, r["target"][h], cam), fade=ramp(t, 0, 0.6) * (1 - ramp(t, self.T_HERO - 0.5, self.T_HERO)), top_shade=True)
    img = draw_label(img, "04", "The learned policy", "Pick up. Pass. Let go. Deliver.", ramp(t, 0.8, 1.6) * (1 - ramp(t, self.T_HERO - 1.2, self.T_HERO - 0.4)))
    s0 = film[0]; s1 = s0 + film[1]
    a = ramp(t, s0, s0 + 0.3) * (1 - ramp(t, s1 - 0.3, s1))
    if a > 0:
      ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
      ImageDraw.Draw(ov).text((W - 64 - 160, 64), "SLOW MOTION", font=F_SMALL, fill=MUTED + (int(255 * a),))
      img = Image.alpha_composite(img, ov)
    return img

  def f_finalgrid(self, t):
    u = t / self.T_FINALGRID
    cam = Cam([0, 0, 0.05], lerp(5.5, 12.0, ease(u)), lerp(100, 140, u), lerp(-26, -40, ease(u)))
    arr = self.grid_frame(self.final, clip01(t / (self.T_FINALGRID - 0.5)) * self.ep, cam)
    img = finish(arr, fade=ramp(t, 0, 0.5) * (1 - ramp(t, self.T_FINALGRID - 0.6, self.T_FINALGRID)), top_shade=True)
    s = self.final
    return draw_stats(img, s, self.run.minutes.get(s, 0), self.run.rate[s], ramp(t, 0.4, 1.2))

  def f_end(self, t):
    img = Image.new("RGBA", (W, H), (9, 10, 12, 255)); dr = ImageDraw.Draw(img)
    a1, a2, a3 = ramp(t, 0.2, 1.0), ramp(t, 0.8, 1.6), ramp(t, 1.4, 2.2); fo = 1 - ramp(t, self.T_END - 0.8, self.T_END)
    y = 500; s = self.final
    dr.rectangle([96, y - 30, 140, y - 26], fill=ACCENT + (int(255 * a1 * fo),))
    dr.text((96, y), "Two-arm handover,", font=F_TITLE, fill=INK + (int(255 * a1 * fo),))
    dr.text((96, y + 80), "learned from scratch", font=F_TITLE, fill=INK + (int(255 * a1 * fo),))
    dr.text((96, y + 200), "Reinforcement learning (PPO) in MuJoCo MJX", font=F_SUB, fill=MUTED + (int(255 * a2 * fo),))
    dr.text((96, y + 245), f"{fmt_steps(s)} simulation steps  ·  {self.run.minutes.get(s, 0):.0f} min on one GPU  ·  {self.run.rate[s]*100:.0f}% success",
            font=F_SUB, fill=MUTED + (int(255 * a2 * fo),))
    dr.text((96, y + 290), "Task design: DeepMind MuJoCo Playground", font=F_SUB, fill=MUTED + (int(200 * a2 * fo),))
    dr.text((96, y + 390), "Kishan Chhari", font=F_LABEL, fill=INK + (int(255 * a3 * fo),))
    return img


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("run"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=-1)
  ap.add_argument("--out", required=True); ap.add_argument("--still", type=float, default=None); ap.add_argument("--info", action="store_true")
  a = ap.parse_args(); film = Film(a.run); n = int(round(film.total * FPS))
  if a.info:
    print(json.dumps({"frames": n, "seconds": film.total, "checkpoints": film.k_steps, "mid": film.mid, "hero": film.hero,
                      "bounds": [float(x) for x in film.bounds]})); return
  if a.still is not None:
    film.frame(a.still).convert("RGB").save(a.out); print("saved", a.out); return
  end = n if a.end < 0 else min(a.end, n)
  ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                         "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
  for f in range(a.start, end):
    ff.stdin.write(np.asarray(film.frame(f / FPS).convert("RGB")).tobytes())
    if (f - a.start) % 300 == 0: print(f"frame {f}/{end}", flush=True)
  ff.stdin.close(); ff.wait(); print("PART_DONE", a.out, flush=True)


if __name__ == "__main__":
  main()
