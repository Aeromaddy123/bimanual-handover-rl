"""Replay every checkpoint of a run: measure success over many pairs and keep the 63-pair recordings for the video."""
import argparse, glob, json, os, pathlib, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jp, numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import handover_env

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--n", type=int, default=256); ap.add_argument("--keep", type=int, default=63)
ap.add_argument("--seed", type=int, default=7); ap.add_argument("--deterministic", action="store_true")
a = ap.parse_args()
run = pathlib.Path(__file__).parent / "runs" / a.run
args = json.loads((run / "args.json").read_text())
env = handover_env.PandaHandover(config_overrides=json.loads(args["overrides"]))
NET = (tuple(args["network"]["policy"]), tuple(args["network"]["value"])) if "network" in args else ((512, 256, 128), (512, 256, 256))
T = env._config.episode_length
net = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                     preprocess_observations_fn=running_statistics.normalize,
                                     policy_hidden_layer_sizes=NET[0], value_hidden_layer_sizes=NET[1])
make_policy = ppo_networks.make_inference_fn(net)
reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))


@jax.jit
def episode(params, key):
  policy = make_policy(params, deterministic=a.deterministic)
  s = reset(jax.random.split(key, a.n))
  def body(carry, _):
    s, k = carry
    k, k1 = jax.random.split(k)
    act, _ = policy(s.obs, k1)
    s = step(s, act)
    return (s, k), (s.data.qpos[:a.keep], s.info["handed"], s.metrics["success"])
  (s_end, _), (qpos, handed, succ) = jax.lax.scan(body, (s, key), None, length=T)
  return s.data.qpos[:a.keep], qpos, handed, succ, s_end.info["success_ever"], s_end.info["goal"][:a.keep]


out = run / "recordings"; out.mkdir(exist_ok=True)
summary = []
for ck in sorted(glob.glob(str(run / "ckpt" / "step_*.pkl"))):
  stepnum = int(pathlib.Path(ck).stem.split("_")[1])
  params = pickle.load(open(ck, "rb"))
  t = time.time()
  q0, qpos, handed, succ, ever, goal = jax.device_get(episode(params, jax.random.PRNGKey(a.seed)))
  qpos = np.concatenate([q0[None], qpos], 0)
  rate = float(np.mean(ever)); hrate = float(np.mean(handed[-1]))
  np.savez_compressed(out / f"step_{stepnum:012d}.npz", qpos=qpos.astype(np.float32), goal=goal,
                      success_ever=ever[:a.keep], handed=handed[:, :a.keep], success=succ[:, :a.keep], dt=env.dt)
  summary.append({"step": stepnum, "success_rate": rate, "handover_rate": hrate})
  print(f"step {stepnum:>12,}: success {rate*100:5.1f}%  handover {hrate*100:5.1f}%  ({time.time()-t:.0f}s)", flush=True)
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print("ROLLOUTS_DONE", flush=True)
