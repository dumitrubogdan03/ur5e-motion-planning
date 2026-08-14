import numpy as np
import pytest

from armlab.ik import analytical_ik, best_solution, numerical_ik
from armlab.model import (Robot, forward_kinematics, geometric_jacobian,
                          joint_origins, manipulability, pose_error)

ROBOT = Robot()


def finite_difference_jacobian(q, h=1e-7):
    jac = np.zeros((6, 6))
    base = forward_kinematics(ROBOT, q)
    for i in range(6):
        shifted = np.array(q, dtype=float)
        shifted[i] += h
        moved = forward_kinematics(ROBOT, shifted)
        jac[:3, i] = (moved[:3, 3] - base[:3, 3]) / h
        r = moved[:3, :3] @ base[:3, :3].T
        jac[3:, i] = np.array([r[2, 1] - r[1, 2],
                               r[0, 2] - r[2, 0],
                               r[1, 0] - r[0, 1]]) / (2 * h)
    return jac


def test_zero_configuration_reaches_expected_height():
    pose = forward_kinematics(ROBOT, np.zeros(6))
    assert pose.shape == (4, 4)
    assert np.allclose(pose[3], [0, 0, 0, 1])


def test_forward_kinematics_is_a_rigid_transform():
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = forward_kinematics(ROBOT, ROBOT.random_configuration(rng))[:3, :3]
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(r), 1.0, atol=1e-12)


def test_joint_origins_start_at_the_base():
    rng = np.random.default_rng(1)
    points = joint_origins(ROBOT, ROBOT.random_configuration(rng))
    assert len(points) == 7
    assert np.allclose(points[0], np.zeros(3))


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(2)
    for _ in range(30):
        q = ROBOT.random_configuration(rng)
        assert np.allclose(geometric_jacobian(ROBOT, q),
                           finite_difference_jacobian(q), atol=1e-5)


def test_analytical_ik_round_trip():
    rng = np.random.default_rng(3)
    solved = 0
    for _ in range(200):
        q = ROBOT.random_configuration(rng)
        target = forward_kinematics(ROBOT, q)
        exact = [s for s in analytical_ik(ROBOT, target)
                 if pose_error(forward_kinematics(ROBOT, s), target)[0] < 1e-8]
        if exact:
            solved += 1
    assert solved == 200


def test_analytical_ik_returns_multiple_branches():
    rng = np.random.default_rng(4)
    counts = []
    for _ in range(50):
        target = forward_kinematics(ROBOT, ROBOT.random_configuration(rng))
        counts.append(len([
            s for s in analytical_ik(ROBOT, target)
            if pose_error(forward_kinematics(ROBOT, s), target)[0] < 1e-8]))
    assert np.mean(counts) > 4.0
    assert max(counts) <= 8


def test_unreachable_pose_yields_no_solution():
    far = np.eye(4)
    far[:3, 3] = [5.0, 0.0, 0.0]
    assert analytical_ik(ROBOT, far) == []


def test_best_solution_prefers_the_nearest_branch():
    rng = np.random.default_rng(5)
    q = ROBOT.random_configuration(rng)
    target = forward_kinematics(ROBOT, q)
    picked = best_solution(ROBOT, target, seed=q)
    assert picked is not None
    assert np.linalg.norm(picked - q) < 1e-6


def test_numerical_ik_converges_from_a_nearby_seed():
    rng = np.random.default_rng(6)
    hits = 0
    for _ in range(60):
        q = ROBOT.random_configuration(rng)
        target = forward_kinematics(ROBOT, q)
        seed = ROBOT.clamp(q + rng.normal(0.0, 0.15, 6))
        found, _ = numerical_ik(ROBOT, target, seed)
        if found is not None:
            assert pose_error(forward_kinematics(ROBOT, found),
                              target)[0] < 1e-4
            hits += 1
    assert hits >= 55


def test_manipulability_drops_at_a_singular_configuration():
    straight = np.zeros(6)
    rng = np.random.default_rng(7)
    generic = [manipulability(ROBOT, ROBOT.random_configuration(rng))
               for _ in range(200)]
    assert manipulability(ROBOT, straight) < np.median(generic)


def test_pose_error_is_zero_for_identical_poses():
    rng = np.random.default_rng(8)
    pose = forward_kinematics(ROBOT, ROBOT.random_configuration(rng))
    position, rotation = pose_error(pose, pose)
    assert position == pytest.approx(0.0, abs=1e-12)
    assert rotation == pytest.approx(0.0, abs=1e-9)
