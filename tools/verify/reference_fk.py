"""Independent URDF forward kinematics, standard library only.

Reads the URDF itself (not the imported project) and composes 4x4 matrices:
  T_child = T_parent * Origin(xyz, rpy) * Motion(axis, q)
with URDF's fixed-axis roll-pitch-yaw, R = Rz(yaw) Ry(pitch) Rx(roll).
Then runs the product's `axisforge fk` on the imported project for the same
joint values and compares every link's position and orientation.
"""
import json, math, random, subprocess, sys
import xml.etree.ElementTree as ET

def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]]

def homogeneous(rot, t):
    return [rot[0] + [t[0]], rot[1] + [t[1]], rot[2] + [t[2]], [0, 0, 0, 1]]

def axis_angle(axis, q):
    n = math.sqrt(sum(v * v for v in axis)); x, y, z = (v / n for v in axis)
    c, s, C = math.cos(q), math.sin(q), 1 - math.cos(q)
    return [[c + x*x*C, x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, c + z*z*C]]

def numbers(text, default):
    return [float(v) for v in text.split()] if text else default

def load(path):
    root = ET.parse(path).getroot()
    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = numbers(o.get("xyz") if o is not None else None, [0, 0, 0])
        rpy = numbers(o.get("rpy") if o is not None else None, [0, 0, 0])
        a = j.find("axis")
        lim = j.find("limit")
        joints.append(dict(name=j.get("name"), type=j.get("type"),
            parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            origin=homogeneous(rpy_matrix(*rpy), xyz),
            axis=numbers(a.get("xyz") if a is not None else None, [1, 0, 0]),
            lower=float(lim.get("lower", 0)) if lim is not None else 0.0,
            upper=float(lim.get("upper", 0)) if lim is not None else 0.0))
    links = [l.get("name") for l in root.findall("link")]
    children = {j["child"] for j in joints}
    (base,) = [l for l in links if l not in children]
    return base, joints

def fk(base, joints, values):
    world = {base: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
    pending = list(joints)
    while pending:
        for j in list(pending):
            if j["parent"] not in world:
                continue
            q = values.get(j["name"], 0.0)
            if j["type"] in ("revolute", "continuous"):
                motion = homogeneous(axis_angle(j["axis"], q), [0, 0, 0])
            elif j["type"] == "prismatic":
                n = math.sqrt(sum(v * v for v in j["axis"]))
                motion = homogeneous(axis_angle([0, 0, 1], 0), [v / n * q for v in j["axis"]])
            else:
                motion = homogeneous(axis_angle([0, 0, 1], 0), [0, 0, 0])
            world[j["child"]] = matmul(matmul(world[j["parent"]], j["origin"]), motion)
            pending.remove(j)
    return world

def quat_to_matrix(x, y, z, w):
    return [[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]

def rotation_error(a, b):
    # ||A - B||_F = 2*sqrt(2)*sin(theta/2): well conditioned near zero, where
    # acos of the trace loses half its digits.
    frob = math.sqrt(sum((a[i][k] - b[i][k]) ** 2 for i in range(3) for k in range(3)))
    return 2 * math.asin(min(1.0, frob / (2 * math.sqrt(2))))

def main(cli, urdf, project, samples, seed):
    base, joints = load(urdf)
    actuated = [j for j in joints if j["type"] in ("revolute", "continuous", "prismatic")]
    rng = random.Random(seed)
    worst_position = worst_angle = 0.0
    compared = 0
    for _ in range(samples):
        values = {}
        for j in actuated:
            lo, hi = (-math.pi, math.pi) if j["type"] == "continuous" else (j["lower"], j["upper"])
            values[j["name"]] = rng.uniform(lo, hi)
        reference = fk(base, joints, values)
        args = [cli, "fk", project] + [f"--set={n}={v!r}" for n, v in values.items()]
        product = json.loads(subprocess.run(args, check=True, capture_output=True, text=True).stdout)
        for link, m in reference.items():
            x, y, z, qx, qy, qz, qw = product[link]
            worst_position = max(worst_position, math.dist((x, y, z), (m[0][3], m[1][3], m[2][3])))
            worst_angle = max(worst_angle, rotation_error([row[:3] for row in m[:3]], quat_to_matrix(qx, qy, qz, qw)))
            compared += 1
    print(json.dumps(dict(robot=urdf.split("/")[-2], postures=samples, link_poses_compared=compared,
        actuated_joints=len(actuated), worst_position_mm=worst_position * 1000,
        worst_rotation_deg=math.degrees(worst_angle))))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
