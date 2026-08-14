import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armlab.collision import random_scene
from armlab.model import Robot
from armlab.planner import rrt_connect, straight_line

DENSITIES = (0, 10, 20, 40)


def sample_free(robot: Robot, scene, rng, attempts: int = 4000):
    for _ in range(attempts):
        q = robot.random_configuration(rng)
        if not scene.collides(robot, q):
            return q
    return None


def run(runs: int, seed: int, densities=DENSITIES) -> dict:
    robot = Robot()
    out = {}

    for density in densities:
        rows = []
        for trial in range(runs):
            rng = np.random.default_rng(seed + trial)
            scene = random_scene(density, rng)
            start = sample_free(robot, scene, rng)
            goal = sample_free(robot, scene, rng)
            if start is None or goal is None:
                continue

            plan = rrt_connect(robot, scene, start, goal, rng)
            direct = straight_line(robot, scene, start, goal)

            rows.append({
                "success": bool(plan.success),
                "plan_ms": plan.plan_ms,
                "shortcut_ms": plan.shortcut_ms,
                "iterations": plan.iterations,
                "waypoints": len(plan.path),
                "joint_length": plan.joint_length,
                "raw_joint_length": plan.raw_joint_length,
                "cartesian_length": plan.cartesian_length(robot),
                "direct_success": bool(direct.success),
            })
        out[str(density)] = rows
    return {"runs": runs, "seed": seed, "results": out}


def summarise(data: dict) -> None:
    print(f"{data['runs']} planning queries per density, seed {data['seed']}")
    print()
    header = (f"{'obstacles':>10}{'RRT':>8}{'direct':>8}{'plan ms':>10}"
              f"{'short ms':>10}{'iters':>8}{'joint':>8}{'cart m':>8}"
              f"{'saved':>8}")
    print(header)
    print("-" * len(header))

    for density, rows in data["results"].items():
        ok = [r for r in rows if r["success"]]
        if not ok:
            continue
        saved = np.mean([1.0 - r["joint_length"] / r["raw_joint_length"]
                         for r in ok if r["raw_joint_length"] > 0])
        print(f"{density:>10}"
              f"{len(ok) / len(rows):>7.0%}"
              f"{sum(r['direct_success'] for r in rows) / len(rows):>8.0%}"
              f"{np.median([r['plan_ms'] for r in ok]):>10.0f}"
              f"{np.median([r['shortcut_ms'] for r in ok]):>10.0f}"
              f"{np.median([r['iterations'] for r in ok]):>8.0f}"
              f"{np.median([r['joint_length'] for r in ok]):>8.2f}"
              f"{np.median([r['cartesian_length'] for r in ok]):>8.2f}"
              f"{saved:>8.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("docs/planner_benchmark.json"))
    args = ap.parse_args()

    data = run(args.runs, args.seed)
    summarise(data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1))
    print()
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
