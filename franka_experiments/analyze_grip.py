import sys, numpy as np, mujoco
rec = np.load(sys.argv[1]); m = mujoco.MjModel.from_xml_path("assets/handover_scene.xml"); d = mujoco.MjData(m)
a_site, ga = m.site("a_gripper").id, m.site("grip_a").id
bar = m.body("bar").id; fa = m.jnt_qposadr[m.joint("a_finger_joint1").id]
for e in range(rec["qpos"].shape[1]):
  print(f"--- pair {e}")
  for t in range(0, rec["qpos"].shape[0], 25):
    d.qpos[:] = rec["qpos"][t, e]; mujoco.mj_forward(m, d)
    rel = d.site_xpos[a_site] - d.site_xpos[ga]
    rel_bar = d.xmat[bar].reshape(3, 3).T @ (d.site_xpos[a_site] - d.xpos[bar])   # gripper point in bar frame
    zaxis = d.site_xmat[a_site].reshape(3, 3)[:, 2]
    tilt = np.degrees(np.arccos(abs(d.xmat[bar].reshape(3, 3)[2, 2])))
    con = [m.geom(d.contact[i].geom1).name + "/" + m.geom(d.contact[i].geom2).name for i in range(d.ncon)
           if "bar" in (m.geom(d.contact[i].geom1).name, m.geom(d.contact[i].geom2).name)]
    print(f"t={t*0.02:4.1f}s bar_z={d.xpos[bar][2]*100:5.1f}cm tilt={tilt:4.0f}deg grip_in_bar_frame={np.round(rel_bar*100,1)}cm "
          f"finger={d.qpos[fa]*100:4.1f}cm hand_z_axis={np.round(zaxis,2)} contacts={len(con)} {sorted(set(c.split('/')[0][:14] for c in con))}")
