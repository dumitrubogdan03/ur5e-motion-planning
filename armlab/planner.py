from dataclasses import dataclass, field

import numpy as np

from .collision import Scene
from .model import Robot, forward_kinematics


@dataclass
class PlannerResult:
    path: list = field(default_factory=list)
    raw_path: list = field(default_factory=list)
    success: bool = False
    plan_ms: float = 0.0
    shortcut_ms: float = 0.0
    iterations: int = 0
    collision_checks: int = 0

    @property
    def milliseconds(self) -> float:
        return self.plan_ms + self.shortcut_ms

    @staticmethod
    def _length(path: list) -> float:
        if len(path) < 2:
            return 0.0
        return float(sum(np.linalg.norm(b - a)
                         for a, b in zip(path, path[1:])))

    @property
    def joint_length(self) -> float:
        return self._length(self.path)

    @property
    def raw_joint_length(self) -> float:
        return self._length(self.raw_path)

    def cartesian_length(self, robot: Robot) -> float:
        if len(self.path) < 2:
            return 0.0
        points = [forward_kinematics(robot, q)[:3, 3] for q in self.path]
        return float(sum(np.linalg.norm(b - a)
                         for a, b in zip(points, points[1:])))


class _Tree:

    def __init__(self, root: np.ndarray):
        self.nodes = [np.asarray(root, dtype=float)]
        self.parents = [-1]

    def nearest(self, q: np.ndarray) -> int:
        data = np.array(self.nodes)
        return int(np.argmin(np.linalg.norm(data - q, axis=1)))

    def add(self, q: np.ndarray, parent: int) -> int:
        self.nodes.append(np.asarray(q, dtype=float))
        self.parents.append(parent)
        return len(self.nodes) - 1

    def branch(self, index: int) -> list:
        out = []
        while index != -1:
            out.append(self.nodes[index])
            index = self.parents[index]
        return out[::-1]


def _steer(a: np.ndarray, b: np.ndarray, step: float) -> np.ndarray:
    delta = b - a
    dist = float(np.linalg.norm(delta))
    return b if dist <= step else a + delta * (step / dist)


def rrt_connect(robot: Robot, scene: Scene, start, goal,
                rng: np.random.Generator, step: float = 0.35,
                max_iterations: int = 4000, goal_bias: float = 0.1,
                resolution: float = 0.05) -> PlannerResult:
    import time

    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    result = PlannerResult()
    began = time.perf_counter()

    checks = [0]

    def free(a, b) -> bool:
        checks[0] += 1
        return scene.segment_free(robot, a, b, resolution)

    if scene.collides(robot, start) or scene.collides(robot, goal):
        result.plan_ms = (time.perf_counter() - began) * 1000.0
        result.collision_checks = checks[0]
        return result

    trees = [_Tree(start), _Tree(goal)]
    swapped = False

    for iteration in range(max_iterations):
        result.iterations = iteration + 1
        target = goal if (not swapped and rng.random() < goal_bias) \
            else robot.random_configuration(rng)

        a, b = trees
        near = a.nearest(target)
        stepped = _steer(a.nodes[near], target, step)
        if not free(a.nodes[near], stepped):
            trees.reverse()
            swapped = not swapped
            continue
        grown = a.add(stepped, near)

        other = b.nearest(stepped)
        reached = b.nodes[other]
        while True:
            advanced = _steer(reached, stepped, step)
            if not free(reached, advanced):
                break
            other = b.add(advanced, other)
            reached = advanced
            if float(np.linalg.norm(reached - stepped)) < 1e-9:
                first = a.branch(grown)
                second = b.branch(other)[::-1]
                path = first + second[1:]
                if swapped:
                    path = path[::-1]
                result.plan_ms = (time.perf_counter() - began) * 1000.0
                result.raw_path = path
                trimmed = time.perf_counter()
                result.path = shortcut(robot, scene, path, rng, resolution)
                result.shortcut_ms = (time.perf_counter() - trimmed) * 1000.0
                result.success = True
                result.collision_checks = checks[0]
                return result

        trees.reverse()
        swapped = not swapped

    result.plan_ms = (time.perf_counter() - began) * 1000.0
    result.collision_checks = checks[0]
    return result


def shortcut(robot: Robot, scene: Scene, path: list,
             rng: np.random.Generator, resolution: float = 0.05,
             attempts: int = 120) -> list:
    path = [np.asarray(q, dtype=float) for q in path]
    for _ in range(attempts):
        if len(path) <= 2:
            break
        i = int(rng.integers(0, len(path) - 2))
        j = int(rng.integers(i + 2, len(path)))
        if scene.segment_free(robot, path[i], path[j], resolution):
            path = path[:i + 1] + path[j:]
    return path


def straight_line(robot: Robot, scene: Scene, start, goal,
                  resolution: float = 0.05) -> PlannerResult:
    import time

    began = time.perf_counter()
    result = PlannerResult()
    ok = scene.segment_free(robot, start, goal, resolution)
    result.success = ok
    result.path = [np.asarray(start, dtype=float),
                   np.asarray(goal, dtype=float)] if ok else []
    result.raw_path = list(result.path)
    result.collision_checks = 1
    result.plan_ms = (time.perf_counter() - began) * 1000.0
    return result
