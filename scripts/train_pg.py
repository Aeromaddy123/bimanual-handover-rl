"""Train a MuJoCo Playground task with DeepMind's published PPO settings (only num_envs and the engine changed).
Saves checkpoints and progress in the same format as train.py."""
import argparse, functools, json, os, pathlib, pickle, sys, time
import jax
import jax.numpy as jnp

def _device_put_replicated(x, devices):   # Brax 0.14 still calls this; JAX 0.10 removed it
  return jax.tree.map(lambda a: jnp.stack([jnp.asarray(a)] * len(devices)), x)
jax.device_put_replicated = _device_put_replicated

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import manipulation_params

ap = argparse.ArgumentParser()
ap.add_argument("--env", default="AlohaHandOver"); ap.add_argument("--name", required=True)
ap.add_argument("--num_envs", type=int, default=8192); ap.add_argument("--steps", type=float, default=None)
ap.add_argument("--evals", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

out = pathlib.Path(__file__).parent / "runs" / args.name
(out / "ckpt").mkdir(parents=True, exist_ok=True)
overrides = {"impl": "jax"}
env = registry.load(args.env, config_overrides=overrides)
eval_env = registry.load(args.env, config_overrides=overrides)
cfg = manipulation_params.brax_ppo_config(args.env)
steps = int(args.steps or cfg.num_timesteps)
net_cfg = dict(cfg.network_factory)
(out / "args.json").write_text(json.dumps({**vars(args), "steps": steps, "overrides": json.dumps(overrides),
                                           "ppo": {k: v for k, v in cfg.to_dict().items() if k != "network_factory"},
                                           "network": net_cfg}, indent=2, default=str))
t0 = time.time(); log = open(out / "progress.jsonl", "a")

def progress(step, metrics):
  rec = {"step": int(step), "minutes": round((time.time() - t0) / 60, 2)}
  rec.update({k: float(v) for k, v in metrics.items() if k.startswith("eval/episode_")})
  log.write(json.dumps(rec) + "\n"); log.flush()
  print(f"[{rec['minutes']:6.1f} min] step {step:>12,}  reward {rec.get('eval/episode_reward', 0):8.2f}  "
        f"handover {rec.get('eval/episode_box_handover', 0):7.1f}  target {rec.get('eval/episode_handover_target', 0):7.1f}", flush=True)

def save_params(step, make_policy, params):
  with open(out / "ckpt" / f"step_{int(step):012d}.pkl", "wb") as f:
    pickle.dump(jax.device_get(params), f)

net = functools.partial(ppo_networks.make_ppo_networks,
                        policy_hidden_layer_sizes=tuple(net_cfg["policy_hidden_layer_sizes"]),
                        value_hidden_layer_sizes=tuple(net_cfg["value_hidden_layer_sizes"]))
kw = cfg.to_dict(); kw.pop("network_factory")
kw.update(num_timesteps=steps, num_evals=args.evals, num_envs=args.num_envs, seed=args.seed)
for k in ("num_resets_per_eval",): kw.pop(k, None)
ppo.train(environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training,
          network_factory=net, progress_fn=progress, policy_params_fn=save_params, **kw)
print("TRAINING_DONE", flush=True)
