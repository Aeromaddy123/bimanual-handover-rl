"""Two Franka Panda arms pass a bar: arm A picks it up, hands it to arm B, B brings it to a goal."""
import pathlib
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import support

from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State

XML = pathlib.Path(__file__).parent / "assets" / "handover_scene.xml"
HANDOVER_POINT = np.array([0.0, 0.0, 0.35], dtype=np.float32)
ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=300,
      action_repeat=1,
      action_scale=0.04,
      # "joint": action = joint-angle deltas (16).  "cartesian": action per arm = hand move x,y,z, twist, grip (10);
      # joint targets come from damped least-squares IK, with the hand kept pointing down.
      action_mode="joint",
      cart_step=0.012,
      cart_yaw_step=0.06,
      bar_yaw_range=0.3,
      # "shaped": dense per-step terms.  "progress": DeepMind's AlohaHandOver design -- a weighted potential of
      # stage-gated distances, and the agent is only paid for beating its best potential so far this episode.
      reward_mode="shaped",
      progress_scales=config_dict.create(gripper_bar=1.0, bar_handover=4.0, handover_target=8.0, no_floor=0.3),
      reward_config=config_dict.create(
          scales=config_dict.create(
              reach_a=1.0,
              grasp_a=0.25,
              lift=8.0,
              airborne=2.0,
              level=1.0,
              hand_down=0.0,
              carry=4.0,
              b_wait=0.3,
              reach_b=2.0,
              grasp_b=3.0,
              release_a=3.0,
              a_retreat=0.5,
              goal=6.0,
              success=8.0,
              hands_apart=1.0,
              floor=0.25,
              action_rate=0.002,
          )
      ),
      impl="jax",
      naconmax=48 * 4096,
      naccdmax=48 * 4096,
      njmax=256,
  )


class PandaHandover(mjx_env.MjxEnv):

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    mj_model = mujoco.MjModel.from_xml_path(XML.as_posix())
    mj_model.opt.timestep = self.sim_dt
    self._mj_model = mj_model
    self._mjx_model = mjx.put_model(mj_model, impl=self._config.impl)
    self._action_scale = self._config.action_scale

    m = mj_model
    qadr = lambda j: m.jnt_qposadr[m.joint(j).id]
    self._a_arm = np.array([qadr(f"a_{j}") for j in ARM])
    self._b_arm = np.array([qadr(f"b_{j}") for j in ARM])
    # Actuated qpos (7 arm joints + first finger joint) per arm, in actuator order.
    self._act_qpos = np.concatenate([
        self._a_arm, [qadr("a_finger_joint1")], self._b_arm, [qadr("b_finger_joint1")]
    ])
    self._a_finger = qadr("a_finger_joint1")
    self._b_finger = qadr("b_finger_joint1")
    self._bar_body = m.body("bar").id
    self._bar_qadr = m.jnt_qposadr[m.body("bar").jntadr[0]]
    self._a_site = m.site("a_gripper").id
    self._b_site = m.site("b_gripper").id
    self._ga_site = m.site("grip_a").id
    self._gb_site = m.site("grip_b").id
    self._goal_mocap = m.body("goal").mocapid[0]
    sadr = lambda n: m.sensor_adr[m.sensor(n).id]
    self._s = {n: sadr(n) for n in
               ["a_lpad_bar", "a_rpad_bar", "b_lpad_bar", "b_rpad_bar", "a_hand_floor", "b_hand_floor"]}
    key = m.keyframe("home")
    self._init_q = jp.array(key.qpos)
    self._init_ctrl = jp.array(key.ctrl)
    self._home_a = jp.array(key.qpos[self._a_arm])
    self._home_b = jp.array(key.qpos[self._b_arm])
    self._lowers, self._uppers = m.actuator_ctrlrange.T
    self._a_dofs = np.array([m.jnt_dofadr[m.joint(f"a_{j}").id] for j in ARM])
    self._b_dofs = np.array([m.jnt_dofadr[m.joint(f"b_{j}").id] for j in ARM])
    self._a_site_body = int(m.site_bodyid[self._a_site])
    self._b_site_body = int(m.site_bodyid[self._b_site])

  # ---------------------------------------------------------------- reset
  def reset(self, rng: jax.Array) -> State:
    rng, k_bar, k_yaw, k_goal = jax.random.split(rng, 4)
    bar_xy = jax.random.uniform(k_bar, (2,), minval=jp.array([-0.36, -0.12]), maxval=jp.array([-0.22, 0.12]))
    yaw = jax.random.uniform(k_yaw, (), minval=-1.0, maxval=1.0) * self._config.bar_yaw_range
    bar_quat = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
    goal = jax.random.uniform(k_goal, (3,), minval=jp.array([0.22, -0.12, 0.18]), maxval=jp.array([0.36, 0.12, 0.30]))

    q = self._init_q.at[self._bar_qadr:self._bar_qadr + 3].set(jp.array([bar_xy[0], bar_xy[1], 0.02]))
    q = q.at[self._bar_qadr + 3:self._bar_qadr + 7].set(bar_quat)
    data = mjx_env.make_data(
        self._mj_model, qpos=q, qvel=jp.zeros(self._mjx_model.nv), ctrl=self._init_ctrl,
        impl=self._mjx_model.impl.value, naconmax=self._config.naconmax,
        naccdmax=self._config.naccdmax, njmax=self._config.njmax)
    data = data.replace(mocap_pos=data.mocap_pos.at[self._goal_mocap].set(goal))
    data = mjx.forward(self._mjx_model, data)

    info = {"rng": rng, "goal": goal, "handed": jp.array(0.0), "success_ever": jp.array(0.0),
            "prev_action": jp.zeros(self.action_size), "prev_potential": jp.array(0.0),
            "steps": jp.array(0, dtype=int), "picked": jp.array(0.0)}
    metrics = {k: jp.array(0.0) for k in self._config.reward_config.scales.keys()}
    metrics.update(success=jp.array(0.0), handed=jp.array(0.0), grasp_a_now=jp.array(0.0), lifted=jp.array(0.0))
    metrics.update({f"p_{k}": jp.array(0.0) for k in self._config.progress_scales.keys()})
    obs = self._get_obs(data, info)
    return State(data, obs, jp.array(0.0), jp.array(0.0), metrics, info)

  # ---------------------------------------------------------------- step
  def _ik_arm(self, data, act, site, body, dofs, qadr, ctrl_arm):
    """Joint targets that move one hand by (dx, dy, dz, yaw) while keeping it pointing down."""
    c = self._config
    jacp, jacr = support.jac(self._mjx_model, data, data.site_xpos[site], body)
    J = jp.concatenate([jacp[dofs].T, jacr[dofs].T], axis=0)          # 6 x 7
    z_now = data.site_xmat[site][:, 2]
    w = 0.2 * jp.cross(z_now, jp.array([0.0, 0.0, -1.0]))             # nudge the tool axis back to straight down
    w = w.at[2].add(act[3] * c.cart_yaw_step)                           # twist about the vertical
    v = act[:3] * c.cart_step                                           # hand displacement this control step (m)
    twist = jp.concatenate([v, w])
    dq = J.T @ jp.linalg.solve(J @ J.T + 1e-4 * jp.eye(6), twist)
    q = data.qpos[qadr]
    target = ctrl_arm + dq
    return jp.clip(target, q - 0.15, q + 0.15)

  def _cartesian_ctrl(self, data, action):
    a, b = action[:5], action[5:]
    ctrl = data.ctrl
    a_arm = self._ik_arm(data, a, self._a_site, self._a_site_body, self._a_dofs, self._a_arm, ctrl[0:7])
    b_arm = self._ik_arm(data, b, self._b_site, self._b_site_body, self._b_dofs, self._b_arm, ctrl[8:15])
    grip = lambda x: (x + 1.0) * 0.02                                    # -1 closed .. +1 open (0.04)
    return jp.concatenate([a_arm, grip(a[4])[None], b_arm, grip(b[4])[None]])

  def step(self, state: State, action: jax.Array) -> State:
    if self._config.action_mode == "cartesian":
      ctrl = jp.clip(self._cartesian_ctrl(state.data, action), self._lowers, self._uppers)
    else:
      ctrl = jp.clip(state.data.ctrl + action * self._action_scale, self._lowers, self._uppers)
    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

    # A single simulation that blows up must not poison the shared observation statistics:
    # give it zero reward, clean observations, and end its episode so it resets.
    bad = (jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
           | (jp.abs(data.qvel) > 1e3).any())

    raw, flags, info = self._reward(data, state.info, action)
    raw = {k: jp.where(bad, 0.0, jp.nan_to_num(v)) for k, v in raw.items()}
    flags = {k: jp.where(bad, 0.0, jp.nan_to_num(v)) for k, v in flags.items()}
    info["handed"] = jp.where(bad, 0.0, jp.nan_to_num(info["handed"]))
    info["success_ever"] = jp.where(bad, 0.0, jp.nan_to_num(info["success_ever"]))
    scales = self._config.reward_config.scales
    reward = jp.clip(sum(raw[k] * scales[k] for k in raw), -1e4, 1e4) * self.dt

    bar = data.xpos[self._bar_body]
    out = jp.any(jp.abs(bar[:2]) > 1.0) | (bar[2] < -0.05)
    dropped = jp.array(False)
    if self._config.reward_mode == "progress":
      reward, dropped, prog = self._progress_reward(data, info)
      reward = jp.where(bad, 0.0, jp.nan_to_num(reward))
      state.metrics.update({f"p_{k}": jp.where(bad, 0.0, jp.nan_to_num(v)) for k, v in prog.items()})
    done = (out | bad | dropped).astype(float)
    info["steps"] = jp.where(done > 0, 0, info["steps"] + 1)

    info["prev_action"] = jp.nan_to_num(action)
    state.metrics.update(**raw, **flags)
    obs = jp.where(bad, 0.0, jp.clip(jp.nan_to_num(self._get_obs(data, info)), -100.0, 100.0))
    return State(data, obs, reward, done, state.metrics, info)

  def _progress_reward(self, data, info):
    """Port of mujoco_playground AlohaHandOver: stage-gated exp(-10 d) terms, paid only for new progress."""
    dist = lambda x, y: jp.exp(-10.0 * jp.linalg.norm(x - y))
    barrier = lambda x, x0=0.0, k=100.0, direction=1.0: 1.0 / (1.0 + jp.exp(-k * direction * (x - x0)))
    newly_reset = info["steps"] == 0
    info["picked"] = jp.where(newly_reset, 0.0, info["picked"])
    info["prev_potential"] = jp.where(newly_reset, 0.0, info["prev_potential"])

    a = data.site_xpos[self._a_site]; b = data.site_xpos[self._b_site]
    ga_pt = data.site_xpos[self._ga_site]; gb_pt = data.site_xpos[self._gb_site]
    bar = data.xpos[self._bar_body]; hp = jp.array(HANDOVER_POINT)
    pre = (bar[0] < -0.12).astype(float)          # still on A's side
    past = (bar[0] >= 0.02).astype(float)         # moved over to B's side
    btwn = (1 - pre) * (1 - past)

    r_a = dist(ga_pt, a) * (pre + btwn)
    r_b = dist(gb_pt, b) * (btwn + past)
    r_b_bias = dist(gb_pt, b) * past
    bar_handover = jp.maximum(dist(bar, hp), dist(a, hp) * past)
    bar_target = dist(info["goal"], bar) * (r_b + r_b_bias) * barrier(a[0], -0.05, direction=-1.0)
    sd = data.sensordata
    floor = ((sd[self._s["a_hand_floor"]] > 0) | (sd[self._s["b_hand_floor"]] > 0)).astype(float)
    terms = {"gripper_bar": r_a + r_b + r_b_bias, "bar_handover": bar_handover,
             "handover_target": bar_target, "no_floor": 1.0 - floor}
    w = self._config.progress_scales
    potential = sum(terms[k] * w[k] for k in terms) / sum(w[k] for k in terms)

    reward = jp.maximum(potential - info["prev_potential"], 0.0)
    reward += 0.02 * potential * barrier(a[0], -0.05, direction=-1.0) * barrier(bar[0], 0.15)
    info["prev_potential"] = jp.maximum(potential, info["prev_potential"])
    reward = jp.where(newly_reset, 0.0, reward)

    info["picked"] = jp.maximum(info["picked"], (bar[2] > 0.15).astype(float))
    dropped = (bar[2] < 0.04) & (info["picked"] > 0)
    reward += dropped.astype(float) * -0.1
    return reward, dropped, terms

  def _grasps(self, data):
    sd = data.sensordata
    ga = ((sd[self._s["a_lpad_bar"]] > 0) & (sd[self._s["a_rpad_bar"]] > 0)).astype(float)
    gb = ((sd[self._s["b_lpad_bar"]] > 0) & (sd[self._s["b_rpad_bar"]] > 0)).astype(float)
    return ga, gb

  def _reward(self, data, info, action):
    a = data.site_xpos[self._a_site]
    b = data.site_xpos[self._b_site]
    ga_pt = data.site_xpos[self._ga_site]
    gb_pt = data.site_xpos[self._gb_site]
    bar = data.xpos[self._bar_body]
    goal = info["goal"]
    hp = jp.array(HANDOVER_POINT)
    grasp_a, grasp_b = self._grasps(data)
    lifted = (bar[2] > 0.05).astype(float)
    airborne = (jp.minimum(ga_pt[2], gb_pt[2]) > 0.05).astype(float)

    handed = jp.maximum(info["handed"], grasp_b * lifted)
    info["handed"] = handed
    near = lambda x, k: 1.0 - jp.tanh(k * jp.linalg.norm(x))
    goal_d = jp.linalg.norm(bar - goal)
    success = handed * (goal_d < 0.04).astype(float) * (1.0 - grasp_a)
    info["success_ever"] = jp.maximum(info["success_ever"], success)

    hands = jp.linalg.norm(a - b)
    sd = data.sensordata
    floor = ((sd[self._s["a_hand_floor"]] > 0) | (sd[self._s["b_hand_floor"]] > 0)).astype(float)

    raw = {
        "reach_a": near(a - ga_pt, 5.0) * (1.0 - handed),
        "grasp_a": grasp_a * (1.0 - handed),
        # paid for every centimetre the LOWER end of the bar rises while A holds it: tipping it up by one end earns nothing
        "lift": jp.clip((jp.minimum(ga_pt[2], gb_pt[2]) - 0.02) / 0.30, 0.0, 1.0) * grasp_a * (1.0 - handed),
        # nothing pays for holding the bar on the floor: these only count once both ends are off the ground
        "airborne": airborne * grasp_a * (1.0 - handed),
        "level": jp.square(data.xmat[self._bar_body][2, 2]) * airborne * grasp_a * (1.0 - handed),
        "hand_down": jp.clip(-data.site_xmat[self._a_site][2, 2], 0.0, 1.0) * (1.0 - handed),
        "carry": near(bar - hp, 4.0) * grasp_a * (1.0 - handed),
        "b_wait": near(data.qpos[self._b_arm] - self._home_b, 1.0) * (1.0 - lifted) * (1.0 - handed),
        "reach_b": near(b - gb_pt, 5.0) * jp.clip((bar[2] - 0.05) / 0.20, 0.0, 1.0),
        "grasp_b": grasp_b * lifted,
        "release_a": handed * (1.0 - grasp_a),
        "a_retreat": handed * near(data.qpos[self._a_arm] - self._home_a, 1.0),
        "goal": near(bar - goal, 4.0) * handed,
        "success": success,
        "hands_apart": -jp.clip((0.10 - hands) / 0.05, 0.0, 1.0),
        "floor": -floor,
        "action_rate": -jp.sum(jp.square(action - info["prev_action"])),
    }
    flags = {"handed": handed, "grasp_a_now": grasp_a, "lifted": lifted}
    return raw, flags, info

  def _get_obs(self, data, info):
    a = data.site_xpos[self._a_site]
    b = data.site_xpos[self._b_site]
    bar = data.xpos[self._bar_body]
    grasp_a, grasp_b = self._grasps(data)
    return jp.concatenate([
        data.qpos,
        data.qvel,
        a, b,
        data.site_xmat[self._a_site].ravel()[3:],
        data.site_xmat[self._b_site].ravel()[3:],
        data.xmat[self._bar_body].ravel()[3:],
        data.site_xpos[self._ga_site] - a,
        data.site_xpos[self._gb_site] - b,
        jp.array(HANDOVER_POINT) - bar,
        info["goal"] - bar,
        data.ctrl - data.qpos[self._act_qpos],
        jp.stack([info["handed"], grasp_a, grasp_b, (bar[2] > 0.05).astype(float),
                  info["steps"].astype(float) / self._config.episode_length]),
    ])

  @property
  def xml_path(self) -> str:
    return XML.as_posix()

  @property
  def action_size(self) -> int:
    return 10 if self._config.action_mode == "cartesian" else self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
