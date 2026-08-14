import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from armlab.ik import analytical_ik, numerical_ik
from armlab.model import (Robot, condition_number, forward_kinematics,
                          manipulability, pose_error)


def run(samples: int, seed: int) -> dict:
    robot = Robot()
    rng = np.random.default_rng(seed)

    rows = []
    for _ in range(samples):
        q = robot.random_configuration(rng)
        target = forward_kinematics(robot, q)
        w = manipulability(robot, q)
        kappa = condition_number(robot, q)

        began = time.perf_counter()
        solutions = analytical_ik(robot, target)
        analytic_ms = (time.perf_counter() - began) * 1000.0
        errors = [pose_error(forward_kinematics(robot, s), target)
                  for s in solutions]
        exact = [e for e in errors if e[0] < 1e-6 and e[1] < 1e-6]
        analytic_error = min((e[0] for e in exact), default=float("nan"))

        seed_q = robot.clamp(q + rng.normal(0.0, 0.5, robot.n))
        began = time.perf_counter()
        found, iterations = numerical_ik(robot, target, seed_q)
        numeric_ms = (time.perf_counter() - began) * 1000.0
        numeric_error = pose_error(forward_kinematics(robot, found),
                                   target)[0] if found is not None \
            else float("nan")

        rows.append({
            "manipulability": w,
            "condition": kappa,
            "analytic_solutions": len(exact),
            "analytic_ok": bool(exact),
            "analytic_ms": analytic_ms,
            "analytic_error": analytic_error,
            "numeric_ok": found is not None,
            "numeric_ms": numeric_ms,
            "numeric_iterations": iterations,
            "numeric_error": numeric_error,
        })
    return {"samples": samples, "seed": seed, "rows": rows}


def summarise(data: dict) -> None:
    rows = data["rows"]
    n = len(rows)
    a_ok = [r for r in rows if r["analytic_ok"]]
    n_ok = [r for r in rows if r["numeric_ok"]]

    print(f"{n} random reachable poses, seed {data['seed']}")
    print()
    print(f"{'':<22}{'analytical':>14}{'numerical':>14}")
    print("-" * 50)
    print(f"{'success rate':<22}{len(a_ok) / n:>13.1%}{len(n_ok) / n:>14.1%}")
    print(f"{'median time (ms)':<22}"
          f"{np.median([r['analytic_ms'] for r in rows]):>13.3f}"
          f"{np.median([r['numeric_ms'] for r in rows]):>14.3f}")
    print(f"{'95th pct time (ms)':<22}"
          f"{np.percentile([r['analytic_ms'] for r in rows], 95):>13.3f}"
          f"{np.percentile([r['numeric_ms'] for r in rows], 95):>14.3f}")
    print(f"{'median error (m)':<22}"
          f"{np.median([r['analytic_error'] for r in a_ok]):>13.2e}"
          f"{np.median([r['numeric_error'] for r in n_ok]):>14.2e}")
    print(f"{'mean solutions':<22}"
          f"{np.mean([r['analytic_solutions'] for r in rows]):>13.2f}"
          f"{1.0:>14.2f}")
    print()

    quartiles = np.quantile([r["manipulability"] for r in rows],
                            [0.0, 0.25, 0.5, 0.75, 1.0])
    print("success rate by manipulability quartile")
    print(f"{'range':<26}{'analytical':>12}{'numerical':>12}{'n':>6}")
    print("-" * 56)
    for i in range(4):
        lo, hi = quartiles[i], quartiles[i + 1]
        block = [r for r in rows if lo <= r["manipulability"] <= hi]
        if not block:
            continue
        a = sum(r["analytic_ok"] for r in block) / len(block)
        b = sum(r["numeric_ok"] for r in block) / len(block)
        print(f"{lo:.4f} - {hi:.4f}{'':<10}{a:>11.1%}{b:>12.1%}"
              f"{len(block):>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("docs/ik_benchmark.json"))
    args = ap.parse_args()

    data = run(args.samples, args.seed)
    summarise(data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1))
    print()
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
