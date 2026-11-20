"""Physics check: from a state where the policy grips the bar, move the hand straight up with a scripted
controller (Jacobian steps) and see whether the bar comes along."""
import sys, numpy as np, mujoco, jax
sys.path.insert(0, ".")
import rollout
rec = rollout.rollout(sys.argv[1], n=4, seed=11, T=120)
m = mujoco.MjModel.from_xml_path("assets/handover_scene.xml"); m.opt.timestep = 0.005
d = mujoco.MjData(m)
site = m.site("a_gripper").id; bar = m.body("bar").id
ga, gb = m.site("grip_a").id, m.site("grip_b").id
arm = [m.jnt_dofadr[m.joint(f"a_joint{i}").id] for i in range(1, 8)]
for e in range(4):
  t0 = 110
  d.qpos[:] = rec["qpos"][t0, e]; d.qvel[:] = 0
  mujoco.mj_forward(m, d)
  d.ctrl[:] = np.concatenate([d.qpos[0:7], [0.0], d.qpos[9:16], [0.04]])   # hold pose, A fingers closed
  for _ in range(100): mujoco.mj_step(m, d)                               # settle the grip
  z0 = d.site_xpos[site][2]
  out = []
  for k in range(60):                                                     # 60 x 0.5 cm = 30 cm up
    jac = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jac, None, site)
    J = jac[:, arm]; dq = np.linalg.pinv(J) @ np.array([0, 0, 0.005])
    d.ctrl[0:7] += dq
    for _ in range(10): mujoco.mj_step(m, d)
    if k % 10 == 9:
      out.append(f"hand +{(d.site_xpos[site][2]-z0)*100:4.1f}cm: bar ends z = {d.site_xpos[ga][2]*100:4.1f} / {d.site_xpos[gb][2]*100:4.1f} cm")
  print(f"pair {e} (grip contact at start: {rec['grasp_a'][t0, e]:.0f})"); print("  " + "\n  ".join(out))
