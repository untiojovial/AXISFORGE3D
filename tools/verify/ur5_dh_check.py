"""UR5: product FK against the manufacturer's published DH parameters.

Universal Robots publishes standard DH parameters for the UR5; its `base`
and `tool0` frames are the controller's base and flange frames, i.e. DH
frames 0 and 6. This check shares nothing with the URDF or the importer.
"""
import json, math, random, subprocess, sys

D = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
A = [0, -0.425, -0.39225, 0, 0, 0]
ALPHA = [math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0]
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

def dh(theta, d, a, alpha):
    ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
    return [[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st], [0, sa, ca, d], [0, 0, 0, 1]]

def pose(x, y, z, qx, qy, qz, qw):
    r = [[1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
         [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
         [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]]
    return [r[0] + [x], r[1] + [y], r[2] + [z], [0, 0, 0, 1]]

def inverse(m):
    r = [[m[j][i] for j in range(3)] for i in range(3)]
    t = [-sum(r[i][k] * m[k][3] for k in range(3)) for i in range(3)]
    return [r[0] + [t[0]], r[1] + [t[1]], r[2] + [t[2]], [0, 0, 0, 1]]

cli, project, samples, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rng = random.Random(seed)
worst_position = worst_rotation = 0.0
for _ in range(samples):
    q = [rng.uniform(-math.pi, math.pi) for _ in JOINTS]
    expected = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    for i in range(6):
        expected = matmul(expected, dh(q[i], D[i], A[i], ALPHA[i]))
    args = [cli, "fk", project] + [f"--set={n}={v!r}" for n, v in zip(JOINTS, q)]
    out = json.loads(subprocess.run(args, check=True, capture_output=True, text=True).stdout)
    got = matmul(inverse(pose(*out["base"])), pose(*out["tool0"]))
    worst_position = max(worst_position, math.dist([got[i][3] for i in range(3)], [expected[i][3] for i in range(3)]))
    frob = math.sqrt(sum((got[i][k] - expected[i][k]) ** 2 for i in range(3) for k in range(3)))
    worst_rotation = max(worst_rotation, 2 * math.asin(min(1.0, frob / (2 * math.sqrt(2)))))
print(json.dumps(dict(check="UR5 tool0 in base vs published DH", postures=samples,
    worst_position_mm=worst_position * 1000, worst_rotation_deg=math.degrees(worst_rotation))))
