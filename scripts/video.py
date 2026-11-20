"""Render the LinkedIn video (1080x1350, 60 fps) frame by frame from real training recordings.

Usage: video.py <run> --start F --end F --out part.mp4      (frame range, so parts render in parallel)
       video.py <run> --still SECONDS --out frame.png        (one frame, for previews)
"""
import argparse, json, math, os, pathlib, re, subprocess, sys
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid_scene

ROOT = pathlib.Path(__file__).parent
W, H, FPS = 1080, 1350, 60
ROWS, COLS = 9, 7                       # odd counts so one cell sits at the origin
NCELL = ROWS * COLS
FONT_DIR = "/usr/share/fonts/opentype/noto/"
INK = (232, 234, 237); MUTED = (150, 156, 164); ACCENT = (242, 115, 26)


def font(size, bold=False):
  f = FONT_DIR + ("NotoSansCJK-Bold.ttc" if bold and os.path.exists(FONT_DIR + "NotoSansCJK-Bold.ttc") else "NotoSansCJK-Regular.ttc")
  return ImageFont.truetype(f, size)


F_LABEL, F_SUB, F_SMALL, F_BIG, F_TITLE = font(26, True), font(30), font(22, True), font(58, True), font(64, True)

clip01 = lambda x: min(1.0, max(0.0, x))
def smooth(x):
  x = clip01(x); return x * x * (3 - 2 * x)
def ease(x):  # cubic in-out
  x = clip01(x); return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def ramp(t, a, b): return smooth((t - a) / (b - a)) if b > a else float(t >= a)
def lerp(a, b, u): return a + (b - a) * u


class Cam:
  def __init__(self, lookat, dist, az, el): self.lookat, self.dist, self.az, self.el = np.array(lookat, float), dist, az, el
  def mix(self, o, u):
    return Cam(lerp(self.lookat, o.lookat, u), math.exp(lerp(math.log(self.dist), math.log(o.dist), u)),
               lerp(self.az, o.az, u), lerp(self.el, o.el, u))
  def mj(self):
    c = mujoco.MjvCamera(); c.lookat[:] = self.lookat; c.distance = self.dist; c.azimuth = self.az; c.elevation = self.el
    return c


# ----------------------------------------------------------------------------- data
class Run:
  def __init__(self, name):
    self.dir = ROOT / "runs" / name
    rec = self.dir / "recordings"
    self.summary = json.loads((rec / "summary.json").read_text())
    self.steps = [s["step"] for s in self.summary]
    self.rate = {s["step"]: s["success_rate"] for s in self.summary}
    prog = [json.loads(l) for l in open(self.dir / "progress.jsonl")]
    self.minutes = {p["step"]: p["minutes"] for p in prog}
    self._cache = {}
    self.rec_dir = rec

  def rec(self, step):
    if step not in self._cache:
      self._cache[step] = dict(np.load(self.rec_dir / f"step_{step:012d}.npz"))
    return self._cache[step]

  def pick(self, k):
    """k checkpoints for the timelapse: always first and last, the rest spread over success-rate progress."""
    steps, rates = self.steps, [self.rate[s] for s in self.steps]
    chosen = [steps[0]]
    targets = np.linspace(0, max(rates), k)[1:-1]
    for tg in targets:
      cand = [s for s in steps if self.rate[s] >= tg and s not in chosen]
      if cand: chosen.append(cand[0])
    chosen.append(steps[-1])
    return sorted(set(chosen))


def sample_q(qpos, t, dt):
  """Interpolated qpos (ncell, nq) at time t seconds of a recording (T+1, ncell, nq)."""
  f = clip01(t / (dt * (qpos.shape[0] - 1))) * (qpos.shape[0] - 1)
  i = int(min(math.floor(f), qpos.shape[0] - 2)); u = f - i
  q = (1 - u) * qpos[i] + u * qpos[i + 1]
  quat = q[..., 21:25]; q[..., 21:25] = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
  return q


# ----------------------------------------------------------------------------- renderers
class Cell:
  def __init__(self, w=W, h=H):
    self.m = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "handover_scene.xml"))
    self.m.vis.global_.offwidth, self.m.vis.global_.offheight = max(w, 1080), max(h, 1350)
    self.d = mujoco.MjData(self.m)
    self.r = mujoco.Renderer(self.m, h, w)
    self.home = self.m.keyframe("home").qpos.copy()
    m = self.m
    self.geom_body = m.geom_bodyid.copy()

  def render(self, q, goal, cam, edit=None):
    self.d.qpos[:] = q; self.d.mocap_pos[0] = goal
    mujoco.mj_kinematics(self.m, self.d); mujoco.mj_camlight(self.m, self.d)
    self.r.update_scene(self.d, cam.mj())
    self.r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
    if edit: edit(self.r.scene)
    return self.r.render()


class Grid:
  def __init__(self):
    self.m, self.idx, self.centers, self.gids = grid_scene.build(ROWS, COLS)
    self.d = mujoco.MjData(self.m)
    self.r = mujoco.Renderer(self.m, H, W, max_geom=30000)
    cell_of_body = np.full(self.m.nbody, -1)
    for b in range(self.m.nbody):
      mt = re.match(r"c(\d+)_", self.m.body(b).name)
      if mt: cell_of_body[b] = int(mt.group(1))
    # attached panda bodies are named c{i}_a_link0 etc.
    self.geom_cell = cell_of_body[self.m.geom_bodyid]
    self.center_cell = int(np.argmin(np.linalg.norm(self.centers, axis=1)))

  def render(self, qcells, goals, cam, cell_alpha=None):
    grid_scene.set_state(self.m, self.d, self.idx, self.centers, self.gids, qcells, goals)
    self.r.update_scene(self.d, cam.mj())
    self.r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
    if cell_alpha is not None:
      sc = self.r.scene
      for g in range(sc.ngeom):
        geom = sc.geoms[g]
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM:
          c = self.geom_cell[geom.objid]
          if c >= 0 and cell_alpha[c] < 1.0:
            geom.rgba[3] *= cell_alpha[c]
            geom.pos[2] -= (1.0 - cell_alpha[c]) * 0.25
    return self.r.render()


# ----------------------------------------------------------------------------- overlays
def draw_label(img, num, title, sub, a):
  if a <= 0: return img
  ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
  x, y = 64, H - 190
  dr.rectangle([x, y + 6, x + 36, y + 9], fill=ACCENT + (int(255 * a),))
  dr.text((x + 52, y - 10), f"{num}   {title.upper()}", font=F_LABEL, fill=INK + (int(255 * a),))
  if sub: dr.text((x, y + 34), sub, font=F_SUB, fill=MUTED + (int(235 * a),))
  return Image.alpha_composite(img, ov)


def fmt_steps(s):
  return f"{s/1e6:.1f}M" if s >= 1e6 else f"{s/1e3:.0f}K" if s >= 1e3 else str(int(s))


def draw_stats(img, steps, minutes, success, a):
  if a <= 0: return img
  ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
  cols = [("SIMULATION STEPS", fmt_steps(steps)), ("TRAINING TIME", f"{minutes:.0f} min"), ("SUCCESS", f"{success*100:.0f}%")]
  x = 64
  for i, (k, v) in enumerate(cols):
    dr.text((x, 60), k, font=F_SMALL, fill=MUTED + (int(255 * a),))
    dr.text((x, 86), v, font=F_BIG, fill=(ACCENT if i == 2 else INK) + (int(255 * a),))
    x += 330
  return Image.alpha_composite(img, ov)


def draw_curve(img, run, upto_step, a):
  if a <= 0: return img
  ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
  x0, x1, y0, y1 = 64, W - 64, H - 330, H - 230
  smax = run.steps[-1]
  dr.line([x0, y1, x1, y1], fill=(90, 95, 102, int(200 * a)), width=1)
  if hasattr(run, "passed"):   # second, quieter line: block passed to the other arm
    pp = [(x0 + (x1 - x0) * s / smax, y1 - (y1 - y0) * run.passed[s]) for s in run.steps if s <= upto_step]
    if len(pp) > 1: dr.line(pp, fill=(150, 156, 164, int(200 * a)), width=2, joint="curve")
    lx = x1 - 360
    dr.line([lx, y0 - 20, lx + 28, y0 - 20], fill=(150, 156, 164, int(220 * a)), width=3)
    dr.text((lx + 38, y0 - 34), "PASSED", font=F_SMALL, fill=MUTED + (int(255 * a),))
    dr.line([lx + 150, y0 - 20, lx + 178, y0 - 20], fill=ACCENT + (int(255 * a),), width=4)
    dr.text((lx + 188, y0 - 34), "DELIVERED", font=F_SMALL, fill=MUTED + (int(255 * a),))
  pts = [(x0 + (x1 - x0) * s / smax, y1 - (y1 - y0) * run.rate[s]) for s in run.steps]
  # interpolate the "pen" position up to upto_step
  vis = [p for s, p in zip(run.steps, pts) if s <= upto_step]
  nxt = [(s, p) for s, p in zip(run.steps, pts) if s > upto_step]
  if nxt and vis:
    s0 = max(s for s in run.steps if s <= upto_step); s1, p1 = nxt[0]; p0 = vis[-1]
    u = (upto_step - s0) / max(1, s1 - s0); vis.append((lerp(p0[0], p1[0], u), lerp(p0[1], p1[1], u)))
  if len(vis) > 1:
    dr.line(vis, fill=ACCENT + (int(255 * a),), width=4, joint="curve")
  if vis:
    cx, cy = vis[-1]
    for r_, al in ((14, 50), (8, 110), (5, 255)):
      dr.ellipse([cx - r_, cy - r_, cx + r_, cy + r_], fill=ACCENT + (int(al * a),))
  dr.text((x0, y0 - 34), "RESULTS DURING TRAINING", font=F_SMALL, fill=MUTED + (int(255 * a),))
  return Image.alpha_composite(img, ov)


def vignette():
  yy, xx = np.mgrid[0:H, 0:W]
  r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
  return (1.0 - 0.32 * np.clip(r - 0.55, 0, 1) ** 1.5)[..., None].astype(np.float32)
VIG = None


def finish(arr, fade=1.0, top_shade=False):
  global VIG
  if VIG is None: VIG = vignette()
  f = arr.astype(np.float32) * VIG
  if top_shade:  # soft dark band behind the stats so text stays readable over robots
    band = np.clip(1 - np.linspace(0, 1, 260), 0, 1) ** 1.5
    f[:260] *= (1 - 0.65 * band)[:, None, None]
    bb = np.clip(np.linspace(0, 1, 420), 0, 1) ** 1.6
    f[H - 420:] *= (1 - 0.7 * bb)[:, None, None]
  return Image.fromarray(np.clip(f * fade, 0, 255).astype(np.uint8)).convert("RGBA")


# ----------------------------------------------------------------------------- timeline
class Film:
  def __init__(self, run_name):
    self.run = Run(run_name)
    self.cell = Cell(); self.tile = None; self.grid = None
    st = self.run.steps
    self.k_steps = self.run.pick(8)
    self.final = st[-1]
    fin = self.run.rec(self.final)
    ok = np.where(fin["success_ever"] > 0)[0]
    self.hero = int(ok[0]) if len(ok) else 0
    # mid checkpoint for the 3x3 "nine attempts" split: closest to 45% success
    self.mid = min(st, key=lambda s: abs(self.run.rate[s] - 0.45))
    self.dt = float(fin["dt"]); self.ep = self.dt * (fin["qpos"].shape[0] - 1)
    # segment boundaries (seconds)
    self.T_BUILD, self.T_CLONE = 8.0, 7.0
    self.per_ck, self.xf = 4.6, 0.7
    self.T_LEARN = self.per_ck * len(self.k_steps)
    self.T_SPLIT, self.T_HERO, self.T_FINALGRID, self.T_END = 8.0, 15.0, 8.0, 5.5
    b = np.cumsum([0, self.T_BUILD, self.T_CLONE, self.T_LEARN, self.T_SPLIT, self.T_HERO, self.T_FINALGRID, self.T_END])
    self.bounds = b
    self.total = float(b[-1])
    self.cell_cam = Cam([0, 0, 0.25], 2.9, 125, -22)
    self.high_cam = Cam([0, 0, 0.0], 15.5, 135, -40)

  def grid_r(self):
    if self.grid is None: self.grid = Grid()
    return self.grid

  def frame(self, t):
    b = self.bounds
    if t < b[1]: return self.f_build(t)
    if t < b[2]: return self.f_clone(t - b[1])
    if t < b[3]: return self.f_learn(t - b[2])
    if t < b[4]: return self.f_split(t - b[3])
    if t < b[5]: return self.f_hero(t - b[4])
    if t < b[6]: return self.f_finalgrid(t - b[5])
    return self.f_end(t - b[6])

  # 1. the scene assembles itself
  def f_build(self, t):
    m = self.cell.m
    order = [f"a_{n}" for n in ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger"]]
    times = {}
    for i, n in enumerate(order): times[m.body(n).id] = 1.3 + i * 0.2
    for i, n in enumerate(order): times[m.body(n.replace("a_", "b_", 1)).id] = 3.4 + i * 0.17
    times[m.body("bar").id] = 5.6; times[m.body("goal").id] = 6.3
    gb = self.cell.geom_body
    def edit(sc):
      for g in range(sc.ngeom):
        geom = sc.geoms[g]
        if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM: continue
        tb = times.get(int(gb[geom.objid]))
        if tb is None: continue
        u = ramp(t, tb, tb + (0.55 if gb[geom.objid] != m.body("bar").id else 0.7))
        geom.rgba[3] *= u
        geom.pos[2] += (1 - u) * (0.35 if gb[geom.objid] == m.body("bar").id else 0.05)
    cam = Cam([-0.35, 0, 0.3], 2.4, 75, -12).mix(self.cell_cam, ease(t / self.T_BUILD))
    q = self.cell.home.copy(); goal = np.array([0.3, 0.0, 0.25])
    img = finish(self.cell.render(q, goal, cam, edit), fade=ramp(t, 0.0, 1.4), top_shade=True)
    img = draw_label(img, "01", "The scene", "Two Franka Panda arms. One bar to pass.", ramp(t, 1.6, 2.4) * (1 - ramp(t, 7.2, 7.9)))
    return img

  # 2. copies ripple out from the first cell while the camera cranes up
  def f_clone(self, t):
    g = self.grid_r()
    dist = np.linalg.norm(g.centers[:, :2], axis=1)
    alpha = np.array([ramp(t, 0.6 + 0.32 * dd, 1.2 + 0.32 * dd) for dd in dist])
    alpha[g.center_cell] = 1.0
    cam = self.cell_cam.mix(self.high_cam, ease(t / self.T_CLONE))
    r0 = self.run.rec(self.k_steps[0])
    q = np.repeat(self.cell.home[None], NCELL, 0)
    q[:, 18:25] = r0["qpos"][0][:, 18:25]           # each copy gets its own random bar pose
    img = finish(g.render(q, r0["goal"], cam, alpha), top_shade=True)
    img = draw_label(img, "02", "Massively parallel", f"Each copy tries on its own. Training runs 8,192 at once.",
                     ramp(t, 1.0, 1.8) * (1 - ramp(t, 6.2, 6.9)))
    return img

  # 3. learning timelapse over checkpoints
  def learn_cam(self, t):
    u = t / self.T_LEARN
    return Cam([lerp(0, 0.6, u), 0, lerp(0.0, 0.2, u)], lerp(15.5, 8.5, ease(u)), lerp(135, 205, u), lerp(-40, -26, ease(u)))

  def grid_frame(self, step, tep, cam):
    r = self.run.rec(step)
    return self.grid_r().render(sample_q(r["qpos"], tep, self.dt), r["goal"], cam)

  def f_learn(self, t):
    k = min(int(t // self.per_ck), len(self.k_steps) - 1)
    tl = t - k * self.per_ck
    speed = self.ep / (self.per_ck - 0.2)
    cam = self.learn_cam(t)
    arr = self.grid_frame(self.k_steps[k], tl * speed, cam)
    if k > 0 and tl < self.xf:      # crossfade from the previous checkpoint's last moments
      prev = self.grid_frame(self.k_steps[k - 1], (tl + self.per_ck) * speed, cam)
      u = smooth(tl / self.xf); arr = (prev * (1 - u) + arr * u).astype(np.uint8)
    s_prev = self.k_steps[k - 1] if k > 0 else 0
    s_now = self.k_steps[k]
    u = smooth(tl / self.per_ck) if k > 0 else 1.0
    steps = lerp(s_prev, s_now, u)
    mins = lerp(self.run.minutes.get(s_prev, 0), self.run.minutes.get(s_now, 0), u)
    succ = lerp(self.run.rate[s_prev] if k > 0 else self.run.rate[s_now], self.run.rate[s_now], u)
    a = ramp(t, 0.3, 1.2) * (1 - ramp(t, self.T_LEARN - 0.8, self.T_LEARN))
    img = finish(arr, top_shade=True)
    img = draw_stats(img, steps, mins, succ, a)
    img = draw_curve(img, self.run, steps, a)
    img = draw_label(img, "03", "Learning by trial and error", "Every few minutes the policy gets a little better.",
                     ramp(t, 0.6, 1.5) * (1 - ramp(t, 6.0, 7.0)))
    return img

  # 4. nine attempts at the same moment, mid-training
  def f_split(self, t):
    if self.tile is None: self.tile = Cell(360, 450)
    r = self.run.rec(self.mid)
    order = list(range(9))
    tep = clip01((t - 0.3) / (self.T_SPLIT - 1.2)) * self.ep
    q = sample_q(r["qpos"], tep, self.dt)
    canvas = np.zeros((H, W, 3), np.uint8)
    cam = Cam([0, 0, 0.22], 2.25, 145, -22)
    for i in order:
      rr, cc = divmod(i, 3)
      im = self.tile.render(q[i], r["goal"][i], cam)
      canvas[rr * 450:(rr + 1) * 450, cc * 360:(cc + 1) * 360] = im
    canvas[:, 359:361] = 12; canvas[:, 719:721] = 12; canvas[449:451] = 12; canvas[899:901] = 12
    img = finish(canvas, fade=ramp(t, 0, 0.5) * (1 - ramp(t, self.T_SPLIT - 0.4, self.T_SPLIT)))
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    tag_a = ramp(t, self.T_SPLIT - 2.2, self.T_SPLIT - 1.6) * (1 - ramp(t, self.T_SPLIT - 0.4, self.T_SPLIT))
    for i in order:
      rr, cc = divmod(i, 3)
      ok = r["success_ever"][i] > 0; ho = r["handed"][-1, i] > 0
      txt = "delivered" if ok else ("handed over" if ho else "missed")
      col = ACCENT if ok else (INK if ho else MUTED)
      dr.text((cc * 360 + 18, rr * 450 + 410), txt, font=F_SMALL, fill=col + (int(255 * tag_a),))
    img = Image.alpha_composite(img, ov)
    rate = self.run.rate[self.mid]
    img = draw_label(img, "", f"Nine attempts, same moment", f"Mid-training: {rate*100:.0f}% of attempts succeed.",
                     ramp(t, 0.5, 1.3) * (1 - ramp(t, self.T_SPLIT - 0.6, self.T_SPLIT)))
    return img

  # 5. the learned handover, close up with slow motion around the hand-off
  def f_hero(self, t):
    r = self.run.rec(self.final); h = self.hero
    handed = r["handed"][:, h]; ti = int(np.argmax(handed > 0)) if handed.max() > 0 else len(handed) // 2
    t_hand = ti * self.dt
    # time warp: 1x until 0.9 s before hand-off, 0.35x for 2.2 s of film, then 1x
    pre = max(t_hand - 0.9, 0.0); slow_film = 2.2 / 0.35 * 0.35
    if t < pre: tep = t
    elif t < pre + 2.2 / 0.35 * 0.35 * (1 / 0.35) * 0.35: tep = pre + (t - pre) * 0.35
    else: tep = pre + 2.2 * 0.35 + (t - pre - 2.2)
    tep = min(tep, self.ep)
    q = sample_q(r["qpos"], tep, self.dt)[h]
    u = t / self.T_HERO
    cam = Cam([lerp(-0.2, 0.15, ease(u)), 0, lerp(0.22, 0.3, ease(u))], lerp(2.7, 2.2, ease(u)), lerp(150, 100, ease(u)), lerp(-14, -20, u))
    img = finish(self.cell.render(q, r["goal"][h], cam), fade=ramp(t, 0, 0.6), top_shade=True)
    slow = pre <= t < pre + 2.2
    img = draw_label(img, "04", "The learned policy", "Pick up. Pass. Release. Deliver.", ramp(t, 0.8, 1.6) * (1 - ramp(t, self.T_HERO - 1.2, self.T_HERO - 0.4)))
    if slow:
      ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
      a = ramp(t, pre, pre + 0.3) * (1 - ramp(t, pre + 1.9, pre + 2.2))
      dr.text((W - 64 - 160, 64), "SLOW MOTION", font=F_SMALL, fill=MUTED + (int(255 * a),))
      img = Image.alpha_composite(img, ov)
    return img

  # 6. all copies with the final policy
  def f_finalgrid(self, t):
    u = t / self.T_FINALGRID
    cam = Cam([0, 0, 0.1], lerp(7.5, 14.0, ease(u)), lerp(110, 150, u), lerp(-24, -38, ease(u)))
    arr = self.grid_frame(self.final, clip01(t / (self.T_FINALGRID - 0.5)) * self.ep, cam)
    img = finish(arr, fade=ramp(t, 0, 0.5) * (1 - ramp(t, self.T_FINALGRID - 0.6, self.T_FINALGRID)), top_shade=True)
    s = self.final
    img = draw_stats(img, s, self.run.minutes.get(s, 0), self.run.rate[s], ramp(t, 0.4, 1.2))
    return img

  # 7. end card
  def f_end(self, t):
    img = Image.new("RGBA", (W, H), (9, 10, 12, 255)); dr = ImageDraw.Draw(img)
    a1, a2, a3 = ramp(t, 0.2, 1.0), ramp(t, 0.8, 1.6), ramp(t, 1.4, 2.2)
    fo = 1 - ramp(t, self.T_END - 0.8, self.T_END)
    y = 520
    dr.rectangle([96, y - 30, 140, y - 26], fill=ACCENT + (int(255 * a1 * fo),))
    dr.text((96, y), "Two-arm handover,", font=F_TITLE, fill=INK + (int(255 * a1 * fo),))
    dr.text((96, y + 80), "learned from scratch", font=F_TITLE, fill=INK + (int(255 * a1 * fo),))
    s = self.final
    dr.text((96, y + 200), f"Reinforcement learning (PPO) in MuJoCo MJX", font=F_SUB, fill=MUTED + (int(255 * a2 * fo),))
    dr.text((96, y + 245), f"{fmt_steps(s)} simulation steps  ·  {self.run.minutes.get(s, 0):.0f} min on one GPU  ·  {self.run.rate[s]*100:.0f}% success",
            font=F_SUB, fill=MUTED + (int(255 * a2 * fo),))
    dr.text((96, y + 340), "Kishan Chhari", font=F_LABEL, fill=INK + (int(255 * a3 * fo),))
    return img


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("run"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=-1)
  ap.add_argument("--out", required=True); ap.add_argument("--still", type=float, default=None)
  ap.add_argument("--info", action="store_true")
  a = ap.parse_args()
  film = Film(a.run)
  n = int(round(film.total * FPS))
  if a.info:
    print(json.dumps({"frames": n, "seconds": film.total, "checkpoints": film.k_steps, "mid": film.mid, "hero": film.hero,
                      "bounds": [float(x) for x in film.bounds]})); return
  if a.still is not None:
    film.frame(a.still).convert("RGB").save(a.out); print("saved", a.out); return
  end = n if a.end < 0 else min(a.end, n)
  ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
                         "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
  for f in range(a.start, end):
    ff.stdin.write(np.asarray(film.frame(f / FPS).convert("RGB")).tobytes())
    if (f - a.start) % 300 == 0: print(f"frame {f}/{end}", flush=True)
  ff.stdin.close(); ff.wait(); print("PART_DONE", a.out, flush=True)


if __name__ == "__main__":
  main()
