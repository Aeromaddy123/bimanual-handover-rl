"""PPO training for the two-arm handover. Saves a checkpoint at every evaluation."""
import argparse, functools, json, os, pathlib, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp

# Brax 0.14 still calls jax.device_put_replicated, which JAX 0.10 removed.
# Stacking one copy per device is what it did; pmap places the copies itself.
def _device_put_replicated(x, devices):
  return jax.tree.map(lambda a: jnp.stack([jnp.asarray(a)] * len(devices)), x)
jax.device_put_replicated = _device_put_replicated

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
import handover_env

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--num_envs", type=int, default=4096)
ap.add_argument("--steps", type=float, default=200e6)
ap.add_argument("--evals", type=int, default=25)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--entropy", type=float, default=1e-2)
ap.add_argument("--discount", type=float, default=0.98)
ap.add_argument("--unroll", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--aloha_ppo", action="store_true", help="use the PPO settings DeepMind tuned for AlohaHandOver")
ap.add_argument("--restore", default=None, help="checkpoint .pkl to warm-start the policy from")
ap.add_argument("--overrides", default="{}", help='flattened config overrides, e.g. {"reward_config.scales.carry": 4}')
args = ap.parse_args()

out = pathlib.Path(__file__).parent / "runs" / args.name
(out / "ckpt").mkdir(parents=True, exist_ok=True)
overrides = json.loads(args.overrides)
(out / "args.json").write_text(json.dumps(vars(args), indent=2))

env = handover_env.PandaHandover(config_overrides=overrides)
eval_env = handover_env.PandaHandover(config_overrides=overrides)
episode_length = env._config.episode_length

t0 = time.time()
log = open(out / "progress.jsonl", "a")

def progress(step, metrics):
  rec = {"step": int(step), "minutes": round((time.time() - t0) / 60, 2)}
  rec.update({k: float(v) for k, v in metrics.items() if k.startswith("eval/episode_") or k in ("eval/sps", "training/sps")})
  log.write(json.dumps(rec) + "\n"); log.flush()
  print(f"[{rec['minutes']:6.1f} min] step {step:>12,}  reward {rec.get('eval/episode_reward', 0):8.2f}  "
        f"handed {rec.get('eval/episode_handed', 0):6.1f}  success {rec.get('eval/episode_success', 0):6.1f}", flush=True)

def save_params(step, make_policy, params):
  with open(out / "ckpt" / f"step_{int(step):012d}.pkl", "wb") as f:
    pickle.dump(jax.device_get(params), f)

POLICY, VALUE = ((256, 256, 256), (256,) * 5) if args.aloha_ppo else ((512, 256, 128), (512, 256, 256))
ppo_kw = dict(batch_size=512, num_minibatches=32, unroll_length=15, num_updates_per_batch=8, discounting=0.97,
              learning_rate=1e-3, entropy_cost=2e-2) if args.aloha_ppo else dict(
              batch_size=1024, num_minibatches=32, unroll_length=args.unroll, num_updates_per_batch=4,
              discounting=args.discount, learning_rate=args.lr, entropy_cost=args.entropy)
meta = json.loads((out / "args.json").read_text()); meta.update(network={"policy": POLICY, "value": VALUE}, ppo=ppo_kw)
(out / "args.json").write_text(json.dumps(meta, indent=2))
net = functools.partial(ppo_networks.make_ppo_networks,
                        policy_hidden_layer_sizes=POLICY, value_hidden_layer_sizes=VALUE)
ppo.train(
    environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training,
    num_timesteps=int(args.steps), num_evals=args.evals, episode_length=episode_length,
    num_envs=args.num_envs, **ppo_kw, normalize_observations=True, reward_scaling=1.0,
    clipping_epsilon=0.2, gae_lambda=0.95, max_grad_norm=1.0, num_eval_envs=128,
    network_factory=net, seed=args.seed, progress_fn=progress, policy_params_fn=save_params,
    restore_params=pickle.load(open(args.restore, "rb")) if args.restore else None,
    restore_value_fn=False)
print("TRAINING_DONE", flush=True)
