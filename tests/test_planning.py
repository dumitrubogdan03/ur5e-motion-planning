import numpy as np

from armlab.collision import Box, Scene, random_scene
from armlab.model import Robot
from armlab.planner import rrt_connect, shortcut, straight_line

ROBOT = Robot()


def free_configuration(scene, rng, attempts=5000):
    for _ in range(attempts):
        q = ROBOT.random_configuration(rng)
        if not scene.collides(ROBOT, q):
            return q
    return None


def test_box_distance_is_negative_inside():
    box = Box(np.zeros(3), np.array([0.1, 0.1, 0.1]))
    assert box.distance(np.array([[0.0, 0.0, 0.0]]))[0] < 0
    assert box.distance(np.array([[0.5, 0.0, 0.0]]))[0] > 0
    assert np.isclose(box.distance(np.array([[0.3, 0.0, 0.0]]))[0], 0.2)


def test_empty_scene_still_rejects_configurations_below_the_floor():
    scene = Scene()
    down = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    down[1] = np.pi / 2
    assert scene.collides(ROBOT, down)


def test_base_link_does_not_trigger_the_floor_check():
    scene = Scene()
    upright = np.array([0.0, -np.pi / 2, 0.0, 0.0, 0.0, 0.0])
    assert not scene.collides(ROBOT, upright)


def test_obstacle_density_reduces_the_free_fraction():
    rng = np.random.default_rng(0)
    fractions = []
    for count in (0, 20, 40):
        scene = random_scene(count, rng)
        free = sum(not scene.collides(ROBOT, ROBOT.random_configuration(rng))
                   for _ in range(200))
        fractions.append(free / 200)
    assert fractions[0] > fractions[1] > fractions[2]


def test_planner_finds_a_path_in_an_empty_scene():
    rng = np.random.default_rng(1)
    scene = Scene()
    start = free_configuration(scene, rng)
    goal = free_configuration(scene, rng)
    result = rrt_connect(ROBOT, scene, start, goal, rng)
    assert result.success
    assert np.allclose(result.path[0], start)
    assert np.allclose(result.path[-1], goal)


def test_planned_path_is_collision_free():
    rng = np.random.default_rng(2)
    scene = random_scene(20, rng)
    start = free_configuration(scene, rng)
    goal = free_configuration(scene, rng)
    result = rrt_connect(ROBOT, scene, start, goal, rng)
    assert result.success
    for a, b in zip(result.path, result.path[1:]):
        assert scene.segment_free(ROBOT, a, b)


def test_planner_reports_failure_when_the_goal_is_in_collision():
    rng = np.random.default_rng(3)
    scene = random_scene(10, rng)
    start = free_configuration(scene, rng)
    blocked = None
    for _ in range(5000):
        q = ROBOT.random_configuration(rng)
        if scene.collides(ROBOT, q):
            blocked = q
            break
    result = rrt_connect(ROBOT, scene, start, blocked, rng)
    assert not result.success
    assert result.path == []


def test_shortcut_never_lengthens_the_path():
    rng = np.random.default_rng(4)
    scene = random_scene(15, rng)
    start = free_configuration(scene, rng)
    goal = free_configuration(scene, rng)
    result = rrt_connect(ROBOT, scene, start, goal, rng)
    assert result.success
    assert result.joint_length <= result.raw_joint_length + 1e-9


def test_straight_line_baseline_agrees_with_the_collision_checker():
    rng = np.random.default_rng(5)
    scene = random_scene(10, rng)
    start = free_configuration(scene, rng)
    goal = free_configuration(scene, rng)
    direct = straight_line(ROBOT, scene, start, goal)
    assert direct.success == scene.segment_free(ROBOT, start, goal)


def test_shortcut_keeps_the_endpoints():
    rng = np.random.default_rng(6)
    scene = random_scene(10, rng)
    start = free_configuration(scene, rng)
    goal = free_configuration(scene, rng)
    result = rrt_connect(ROBOT, scene, start, goal, rng)
    trimmed = shortcut(ROBOT, scene, result.raw_path, rng)
    assert np.allclose(trimmed[0], result.raw_path[0])
    assert np.allclose(trimmed[-1], result.raw_path[-1])
