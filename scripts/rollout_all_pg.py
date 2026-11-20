"""Replay every checkpoint of a Playground run (AlohaHandOver): success rate over many pairs + 63-pair recordings."""
import argparse, glob, json, os, pathlib, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jp, numpy as np
from pg_tools import load_env, make_policy_fn

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--env", default="AlohaHandOver")
ap.add_argument("--n", type=int, default=256); ap.add_argument("--keep", type=int, default=63)
ap.add_argument("--seed", type=int, default=7); ap.add_argument("--stochastic", action="store_true")
ap.add_argument("--only", default=None, help="comma separated step numbers")
a = ap.parse_args()
run = pathlib.Path(__file__).parent / "runs" / a.run
env = load_env(a.env); make_policy = make_policy_fn(env, run)
T = env._config.episode_length
reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
box_body = env._box_body
lg = env._left_gripper_site


@jax.jit
def episode(params, key):
  policy = make_policy(params, deterministic=not a.stochastic)
  s = reset(jax.random.split(key, a.n))
  def body(c, _):
    s, k = c; k, k1 = jax.random.split(k)
    act, _ = policy(s.obs, k1); s = step(s, act)
    return (s, k), (s.data.qpos[:a.keep], s.data.xpos[:, box_body], s.data.site_xpos[:, lg])
  (s_end, _), (qpos, box, lgrip) = jax.lax.scan(body, (s, key), None, length=T)
  return s.data.qpos[:a.keep], qpos, box, lgrip, s.info["target_pos"]


out = run / "recordings"; out.mkdir(exist_ok=True)
summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else []
done_steps = {s["step"] for s in summary}
ckpts = sorted(glob.glob(str(run / "ckpt" / "step_*.pkl")))
if a.only: ckpts = [c for c in ckpts if int(pathlib.Path(c).stem.split("_")[1]) in {int(x) for x in a.only.split(",")}]
for ck in ckpts:
  stepnum = int(pathlib.Path(ck).stem.split("_")[1])
  if stepnum in done_steps: continue
  t = time.time()
  q0, qpos, box, lgrip, target = jax.device_get(episode(pickle.load(open(ck, "rb")), jax.random.PRNGKey(a.seed)))
  # success: during the last second the block is within 5 cm of the target, in the air, and the left hand is back on its side
  last = slice(T - 50, T)
  near = np.linalg.norm(box[last] - target[None], axis=-1) < 0.05
  ok = (near & (box[last, :, 2] > 0.05) & (lgrip[last, :, 0] < 0.0)).mean(0) > 0.5
  passed = (box[:, :, 0] > 0.05).any(0)
  rate, prate = float(ok.mean()), float(passed.mean())
  np.savez_compressed(out / f"step_{stepnum:012d}.npz", qpos=np.concatenate([q0[None], qpos], 0).astype(np.float32),
                      target=target[:a.keep], success=ok[:a.keep], passed=passed[:a.keep], dt=env.dt)
  summary.append({"step": stepnum, "success_rate": rate, "handover_rate": prate})
  summary.sort(key=lambda s: s["step"]); (out / "summary.json").write_text(json.dumps(summary, indent=1))
  print(f"step {stepnum:>12,}: success {rate*100:5.1f}%  passed across {prate*100:5.1f}%  ({time.time()-t:.0f}s)", flush=True)
print("ROLLOUTS_DONE", flush=True)
