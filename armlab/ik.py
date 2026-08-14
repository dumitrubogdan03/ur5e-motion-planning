import numpy as np

from .model import (Robot, dh_transform, forward_kinematics,
                    geometric_jacobian, pose_error)

ZERO = 1e-9


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def analytical_ik(robot: Robot, target: np.ndarray) -> list[np.ndarray]:
    l = robot.links
    d1, d4, d5, d6 = l[0].d, l[3].d, l[4].d, l[5].d
    a2, a3 = l[1].a, l[2].a

    solutions = []
    t06 = np.asarray(target, dtype=float)
    t60 = np.linalg.inv(t06)

    p05 = t06 @ np.array([0.0, 0.0, -d6, 1.0])
    radius = float(np.hypot(p05[0], p05[1]))
    if radius < abs(d4) - ZERO:
        return solutions

    psi = float(np.arctan2(p05[1], p05[0]))
    phi = float(np.arccos(np.clip(d4 / radius, -1.0, 1.0)))

    for t1 in (psi + phi + np.pi / 2, psi - phi + np.pi / 2):
        s1, c1 = np.sin(t1), np.cos(t1)

        c5 = (t06[0, 3] * s1 - t06[1, 3] * c1 - d4) / d6
        if abs(c5) > 1.0 + ZERO:
            continue

        for t5 in (float(np.arccos(np.clip(c5, -1.0, 1.0))),
                   -float(np.arccos(np.clip(c5, -1.0, 1.0)))):
            s5 = np.sin(t5)
            if abs(s5) < 1e-7:
                t6 = 0.0
            else:
                t6 = float(np.arctan2(
                    (-t60[1, 0] * s1 + t60[1, 1] * c1) / s5,
                    (t60[0, 0] * s1 - t60[0, 1] * c1) / s5))

            t01 = dh_transform(t1, l[0])
            t45 = dh_transform(t5, l[4])
            t56 = dh_transform(t6, l[5])
            t14 = np.linalg.inv(t01) @ t06 @ np.linalg.inv(t45 @ t56)

            p13 = (t14 @ np.array([0.0, -d4, 0.0, 1.0]))[:3]
            reach = float(np.linalg.norm(p13))
            if reach > abs(a2) + abs(a3) - ZERO:
                continue

            c3 = (reach ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3)
            if abs(c3) > 1.0 + ZERO:
                continue

            for t3 in (float(np.arccos(np.clip(c3, -1.0, 1.0))),
                       -float(np.arccos(np.clip(c3, -1.0, 1.0)))):
                t2 = float(np.arctan2(-p13[1], -p13[0])
                           + np.arcsin(np.clip(a3 * np.sin(t3) / reach,
                                               -1.0, 1.0)))
                t12 = dh_transform(t2, l[1])
                t23 = dh_transform(t3, l[2])
                t34 = np.linalg.inv(t12 @ t23) @ t14
                t4 = float(np.arctan2(t34[1, 0], t34[0, 0]))

                solutions.append(np.array([_wrap(t1), _wrap(t2), _wrap(t3),
                                           _wrap(t4), _wrap(t5), _wrap(t6)]))
    return solutions


def best_solution(robot: Robot, target: np.ndarray, seed=None,
                  tol_pos: float = 1e-4, tol_rot: float = 1e-3):
    valid = []
    for q in analytical_ik(robot, target):
        pos, rot = pose_error(forward_kinematics(robot, q), target)
        if pos < tol_pos and rot < tol_rot and robot.within_limits(q):
            valid.append(q)
    if not valid:
        return None
    if seed is None:
        return valid[0]
    seed = np.asarray(seed, dtype=float)
    return min(valid, key=lambda q: float(np.linalg.norm(q - seed)))


def _pose_residual(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    err = np.zeros(6)
    err[:3] = target[:3, 3] - current[:3, 3]
    r = target[:3, :3] @ current[:3, :3].T
    angle = np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return err
    axis = np.array([r[2, 1] - r[1, 2],
                     r[0, 2] - r[2, 0],
                     r[1, 0] - r[0, 1]]) / (2.0 * np.sin(angle))
    err[3:] = axis * angle
    return err


def numerical_ik(robot: Robot, target: np.ndarray, seed,
                 damping: float = 0.05, max_iterations: int = 200,
                 tol_pos: float = 1e-4, tol_rot: float = 1e-3):
    q = np.asarray(seed, dtype=float).copy()
    target = np.asarray(target, dtype=float)

    for iteration in range(max_iterations):
        current = forward_kinematics(robot, q)
        pos, rot = pose_error(current, target)
        if pos < tol_pos and rot < tol_rot:
            return q, iteration

        err = _pose_residual(current, target)
        jac = geometric_jacobian(robot, q)
        step = jac.T @ np.linalg.solve(
            jac @ jac.T + damping ** 2 * np.eye(6), err)

        norm = float(np.linalg.norm(step))
        if norm > 0.4:
            step *= 0.4 / norm
        q = robot.clamp(q + step)

    return None, max_iterations
