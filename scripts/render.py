import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armlab.collision import random_scene
from armlab.model import Robot, forward_kinematics, joint_origins
from armlab.planner import rrt_connect, straight_line

FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
         (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5))


def box_faces(box):
    c, h = box.center, box.half
    corners = np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                        for sx, sy, sz in
                        ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                         (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))])
    return [corners[list(f)] for f in FACES]


def densify(path, step=0.05):
    out = []
    for a, b in zip(path, path[1:]):
        n = max(int(np.linalg.norm(b - a) / step), 1)
        out.extend(a + (b - a) * (i / n) for i in range(n))
    out.append(path[-1])
    return out


def main():
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ap = argparse.ArgumentParser()
    ap.add_argument("--obstacles", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("docs/plan.gif"))
    args = ap.parse_args()

    robot = Robot()
    rng = np.random.default_rng(args.seed)
    scene = random_scene(args.obstacles, rng)

    def free():
        for _ in range(5000):
            q = robot.random_configuration(rng)
            if not scene.collides(robot, q):
                return q
        raise SystemExit("no free configuration")

    start, goal = free(), free()
    for _ in range(60):
        if not straight_line(robot, scene, start, goal).success:
            break
        start, goal = free(), free()

    result = rrt_connect(robot, scene, start, goal, rng)
    if not result.success:
        raise SystemExit("no plan found, try another seed")

    frames = densify(result.path, step=0.11)

    naive = [start + (goal - start) * (i / (len(frames) - 1))
             for i in range(len(frames))]
    hits = [scene.collides(robot, q) for q in naive]

    fig = plt.figure(figsize=(6.4, 5.2), facecolor="#1e1e22")
    ax = fig.add_subplot(111, projection="3d", facecolor="#1e1e22")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94)

    for box in scene.obstacles:
        ax.add_collection3d(Poly3DCollection(
            box_faces(box), facecolor="#d24b4b", alpha=0.35,
            edgecolor="#e08585", linewidths=0.4))

    traced = [(2, "#5a8cff"), (3, "#73d9f2"), (4, "#8cf299"),
              (5, "#f2d95a"), (6, "#4ce08a")]
    tracks = np.array([joint_origins(robot, q) for q in frames])
    lines = [ax.plot([], [], [], color=c,
                     linewidth=2.4 if j == 6 else 1.3,
                     alpha=1.0 if j == 6 else 0.75)[0] for j, c in traced]

    arm, = ax.plot([], [], [], color="#4c9ae0", linewidth=8,
                   marker="o", markersize=9, markerfacecolor="#f0bf3a",
                   markeredgecolor="#f0bf3a", solid_capstyle="round")

    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_zlim(0.0, 1.05)
    ax.set_box_aspect((1, 1, 0.72))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.11, 0.11, 0.13, 1.0))
        axis.line.set_color("#2c2c33")
        axis.set_ticklabels([])
        axis._axinfo["grid"]["color"] = (0.22, 0.22, 0.26, 1.0)
        axis.set_ticks([])
    title = ax.set_title("", color="#e6e6ec", fontsize=11, pad=2)

    def update(i):
        points = joint_origins(robot, frames[i])
        arm.set_data(points[:, 0], points[:, 1])
        arm.set_3d_properties(points[:, 2])

        for line, (j, _) in zip(lines, traced):
            line.set_data(tracks[:i + 1, j, 0], tracks[:i + 1, j, 1])
            line.set_3d_properties(tracks[:i + 1, j, 2])

        ax.view_init(elev=24, azim=-60 + 35 * i / len(frames))
        title.set_text(
            f"planned {result.joint_length:.2f} rad    "
            f"vs straight line, {sum(hits)} of {len(hits)} steps in collision")
        return (arm, *lines)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=70)

    print(f"{len(frames)} frames -> {args.out}")
    print(f"planned in {result.plan_ms:.0f} ms, "
          f"shortcut {result.shortcut_ms:.0f} ms, "
          f"joint length {result.joint_length:.2f} rad "
          f"(raw {result.raw_joint_length:.2f})")


if __name__ == "__main__":
    main()
