"""Play a saved checkpoint in many parallel arm pairs and record every frame (qpos) for rendering."""
import argparse, json, os, pathlib, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jp, numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import handover_env


def net_sizes(ckpt_path):
  a = pathlib.Path(ckpt_path).resolve().parent.parent / "args.json"
  if a.exists():
    n = json.loads(a.read_text()).get("network")
    if n: return tuple(n["policy"]), tuple(n["value"])
  return (512, 256, 128), (512, 256, 256)


def load_policy(env, ckpt_path, deterministic=True):
  NET = net_sizes(ckpt_path)
  with open(ckpt_path, "rb") as f:
    params = pickle.load(f)
  net = ppo_networks.make_ppo_networks(
      env.observation_size, env.action_size,
      preprocess_observations_fn=running_statistics.normalize,
      policy_hidden_layer_sizes=NET[0], value_hidden_layer_sizes=NET[1])
  return jax.jit(ppo_networks.make_inference_fn(net)(params, deterministic=deterministic))


def rollout(ckpt, n=64, seed=0, T=None, overrides=None, deterministic=True, random_policy=False):
  env = handover_env.PandaHandover(config_overrides=overrides or {})
  T = T or env._config.episode_length
  reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
  key = jax.random.PRNGKey(seed)
  s = reset(jax.random.split(key, n))
  policy = None if random_policy else load_policy(env, ckpt, deterministic)
  qpos, handed, succ, grasp_a = [np.asarray(s.data.qpos)], [], [], []
  for t in range(T):
    key, k = jax.random.split(key)
    if policy is None:
      a = jax.random.uniform(k, (n, env.action_size), minval=-1, maxval=1)
    else:
      a, _ = policy(s.obs, k)
    s = step(s, a)
    qpos.append(np.asarray(s.data.qpos))
    handed.append(np.asarray(s.info["handed"])); succ.append(np.asarray(s.metrics["success"]))
    grasp_a.append(np.asarray(s.metrics["grasp_a_now"]))
  return dict(qpos=np.stack(qpos), goal=np.asarray(s.info["goal"]), handed=np.stack(handed),
              success=np.stack(succ), grasp_a=np.stack(grasp_a), success_ever=np.asarray(s.info["success_ever"]),
              dt=env.dt)


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("ckpt"); ap.add_argument("out")
  ap.add_argument("--n", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--stochastic", action="store_true"); ap.add_argument("--random", action="store_true")
  ap.add_argument("--overrides", default="{}")
  a = ap.parse_args()
  r = rollout(a.ckpt, a.n, a.seed, overrides=json.loads(a.overrides), deterministic=not a.stochastic, random_policy=a.random)
  np.savez_compressed(a.out, **r)
  print(f"saved {a.out}: success {r['success_ever'].mean()*100:.1f}%  handed {r['handed'][-1].mean()*100:.1f}%", flush=True)
