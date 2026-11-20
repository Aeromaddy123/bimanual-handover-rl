"""Replay + snapshot tools for a MuJoCo Playground task (used for AlohaHandOver)."""
import argparse, json, os, pathlib, pickle, sys
import numpy as np


def load_env(name):
  from mujoco_playground import registry
  return registry.load(name, config_overrides={"impl": "jax"})


def make_policy_fn(env, run_dir):
  import jax
  from brax.training.acme import running_statistics
  from brax.training.agents.ppo import networks as ppo_networks
  a = json.loads((run_dir / "args.json").read_text())
  net = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                       preprocess_observations_fn=running_statistics.normalize,
                                       policy_hidden_layer_sizes=tuple(a["network"]["policy_hidden_layer_sizes"]),
                                       value_hidden_layer_sizes=tuple(a["network"]["value_hidden_layer_sizes"]))
  return ppo_networks.make_inference_fn(net)


def rollout(env, make_policy, params, n, seed, deterministic):
  import jax
  T = env._config.episode_length
  reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))

  @jax.jit
  def run(params, key):
    policy = make_policy(params, deterministic=deterministic)
    s = reset(jax.random.split(key, n))
    def body(c, _):
      s, k = c; k, k1 = jax.random.split(k)
      a, _ = policy(s.obs, k1); s2 = step(s, a)
      # keep the final pose after an early termination instead of jumping back
      s2 = jax.tree.map(lambda new, old: jax.numpy.where(
          jax.numpy.reshape(s.done, (-1,) + (1,) * (new.ndim - 1)) > 0, old, new), s2, s) if False else s2
      return (s2, k), (s2.data.qpos, s2.done)
    (_, _), (qpos, done) = jax.lax.scan(body, (s, key), None, length=T)
    return s.data.qpos, qpos, done, s.info["target_pos"]
  q0, qpos, done, target = jax.device_get(run(params, jax.random.PRNGKey(seed)))
  return np.concatenate([q0[None], qpos], 0), done, target


def dark_theme(m):
  """Dark studio look for the stock ALOHA scene (render-only edits)."""
  import mujoco
  for t in range(m.ntex):
    adr, w, h, c = m.tex_adr[t], m.tex_width[t], m.tex_height[t], m.tex_nchannel[t]
    n = w * h * c
    if m.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
      m.tex_data[adr:adr + n] = 10
    elif m.texture(t).name == "groundplane":
      img = m.tex_data[adr:adr + n].reshape(h, w, c).astype(np.int32)
      m.tex_data[adr:adr + n] = np.where(img > 128, 28, 44).astype(np.uint8).ravel()
  m.vis.headlight.ambient[:] = 0.12; m.vis.headlight.diffuse[:] = 0.45; m.vis.headlight.specular[:] = 0.1
  m.vis.map.fogstart, m.vis.map.fogend = 3.0, 7.0
  m.vis.rgba.fog[:] = [0.04, 0.04, 0.045, 1]
  m.mat_reflectance[m.material("groundplane").id] = 0.15


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd")
  r = sub.add_parser("rollout"); r.add_argument("run"); r.add_argument("ckpt"); r.add_argument("out")
  r.add_argument("--env", default="AlohaHandOver"); r.add_argument("--n", type=int, default=8); r.add_argument("--seed", type=int, default=0)
  r.add_argument("--stochastic", action="store_true")
  sn = sub.add_parser("snapshot"); sn.add_argument("rec"); sn.add_argument("out"); sn.add_argument("--title", default="")
  sn.add_argument("--env", default="AlohaHandOver"); sn.add_argument("--i", type=int, default=0)
  a = ap.parse_args()
  root = pathlib.Path(__file__).parent
  if a.cmd == "rollout":
    env = load_env(a.env); run_dir = root / "runs" / a.run
    mp = make_policy_fn(env, run_dir); params = pickle.load(open(a.ckpt, "rb"))
    qpos, done, target = rollout(env, mp, params, a.n, a.seed, not a.stochastic)
    np.savez_compressed(a.out, qpos=qpos, done=done, target=target, dt=env.dt)
    box = qpos[-1, :, -7:-4]
    print(f"saved {a.out}: final box x per pair {np.round(box[:, 0], 2)}  (target x {np.round(target[:, 0], 2)})")
  elif a.cmd == "snapshot":
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    env = load_env(a.env); m = env.mj_model; dark_theme(m)
    d = mujoco.MjData(m); rec = np.load(a.rec)
    rr = mujoco.Renderer(m, 540, 540)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 0.12]; cam.distance = 1.15; cam.azimuth = 90; cam.elevation = -28
    T = rec["qpos"].shape[0]; idx = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 22)
    strip = Image.new("RGB", (540 * len(idx), 540 + 50), (14, 14, 16))
    for j, t in enumerate(idx):
      d.qpos[:] = rec["qpos"][t, a.i]; d.mocap_pos[0] = rec["target"][a.i]
      mujoco.mj_forward(m, d)
      rr.update_scene(d, cam); rr.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
      im = Image.fromarray(rr.render()); ImageDraw.Draw(im).text((16, 12), f"t = {t * float(rec['dt']):.1f} s", fill=(200, 205, 210), font=font)
      strip.paste(im, (540 * j, 50))
    ImageDraw.Draw(strip).text((16, 10), a.title, fill=(230, 232, 235), font=font)
    strip.save(a.out); print("saved", a.out)
