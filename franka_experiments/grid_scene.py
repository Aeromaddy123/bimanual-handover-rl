"""Build a render-only scene with many copies of the handover cell, and map recorded qpos into it."""
import re, pathlib, numpy as np, mujoco

ROOT = pathlib.Path(__file__).parent
BASE = (ROOT / "assets" / "handover_scene.xml").read_text()


def build(rows, cols, dx=2.1, dy=1.25):
  head = BASE.split("<worldbody>")[0]
  head = head.replace('offwidth="1920" offheight="1920"', 'offwidth="2160" offheight="2700"')
  head = head.replace('<statistic center="0 0 0.3" extent="1.5"/>', '<statistic center="0 0 0.2" extent="10"/>')
  head = head.replace('<map fogstart="2.6" fogend="6.0"/>', '<map znear="0.01" zfar="20" fogstart="1.5" fogend="3.3" shadowclip="1.2"/>')
  world = BASE.split("<worldbody>")[1].split("</worldbody>")[0]
  lights = "\n".join(l for l in world.splitlines() if "<light" in l)
  floor = [l for l in world.splitlines() if 'name="floor"' in l][0]
  body = [lights, floor]
  x0, y0 = -(cols - 1) * dx / 2, -(rows - 1) * dy / 2
  for r in range(rows):
    for c in range(cols):
      i = r * cols + c
      cx, cy = x0 + c * dx, y0 + r * dy
      body.append(f'''
    <frame pos="{cx - 0.7:.3f} {cy:.3f} 0"><attach model="panda" body="link0" prefix="c{i}_a_"/></frame>
    <frame pos="{cx + 0.7:.3f} {cy:.3f} 0" euler="0 0 3.14159265359"><attach model="panda" body="link0" prefix="c{i}_b_"/></frame>
    <body name="c{i}_bar" pos="{cx - 0.28:.3f} {cy:.3f} 0.02"><freejoint name="c{i}_barj"/>
      <geom type="box" size="0.10 0.02 0.02" material="bar" contype="0" conaffinity="0"/></body>
    <body mocap="true" name="c{i}_goal" pos="{cx + 0.3:.3f} {cy:.3f} 0.25">
      <geom type="box" size="0.10 0.02 0.02" rgba="0.2 0.6 0.3 0.25" contype="0" conaffinity="0"/></body>''')
  xml = head + "<worldbody>\n" + "\n".join(body) + "\n  </worldbody>\n</mujoco>\n"
  path = ROOT / "assets" / f"grid_{rows}x{cols}.xml"
  path.write_text(xml)
  m = mujoco.MjModel.from_xml_path(str(path))
  cell = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "handover_scene.xml"))
  # qpos index map: for each cell, the grid qpos addresses matching the single-cell qpos layout.
  names = []
  for j in range(cell.njnt):
    names.append((cell.joint(j).name, cell.jnt_qposadr[j], cell.jnt_type[j]))
  idx = np.zeros((rows * cols, cell.nq), dtype=int)
  for i in range(rows * cols):
    for name, adr, typ in names:
      gname = f"c{i}_barj" if typ == mujoco.mjtJoint.mjJNT_FREE else f"c{i}_{name}"
      g = m.jnt_qposadr[m.joint(gname).id]
      n = 7 if typ == mujoco.mjtJoint.mjJNT_FREE else 1
      idx[i, adr:adr + n] = np.arange(g, g + n)
  centers = np.array([[x0 + (i % cols) * dx, y0 + (i // cols) * dy, 0.0] for i in range(rows * cols)])
  goal_ids = np.array([m.body(f"c{i}_goal").mocapid[0] for i in range(rows * cols)])
  return m, idx, centers, goal_ids


def set_state(m, d, idx, centers, goal_ids, qpos_cells, goals):
  """qpos_cells: (ncell, nq) in single-cell coordinates; bar xy and goal are shifted to each cell."""
  q = qpos_cells.copy()
  bar = 18  # bar freejoint qpos address in the single-cell model
  q[:, bar:bar + 2] += centers[:, :2]
  d.qpos[idx.ravel()] = q.ravel()
  d.mocap_pos[goal_ids] = goals + centers
  mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
