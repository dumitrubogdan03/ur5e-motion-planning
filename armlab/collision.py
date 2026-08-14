from dataclasses import dataclass, field

import numpy as np

from .model import Robot, joint_origins

SAMPLES_PER_LINK = 10


@dataclass(frozen=True)
class Box:
    center: np.ndarray
    half: np.ndarray

    def distance(self, points: np.ndarray) -> np.ndarray:
        delta = np.abs(points - self.center) - self.half
        outside = np.linalg.norm(np.maximum(delta, 0.0), axis=-1)
        inside = np.minimum(np.max(delta, axis=-1), 0.0)
        return outside + inside


@dataclass
class Scene:
    obstacles: list = field(default_factory=list)
    floor_z: float = 0.0
    floor_clearance: float = 0.02

    def __post_init__(self):
        if self.obstacles:
            self._centers = np.array([b.center for b in self.obstacles])
            self._halves = np.array([b.half for b in self.obstacles])
        else:
            self._centers = np.zeros((0, 3))
            self._halves = np.zeros((0, 3))

    def clearance(self, robot: Robot, q) -> float:
        points, radii, index = _link_samples(robot, q)
        above = index > 0
        margin = float(np.min(points[above, 2] - radii[above])) \
            - self.floor_z - self.floor_clearance
        if len(self._centers):
            delta = np.abs(points[None, :, :] - self._centers[:, None, :]) \
                - self._halves[:, None, :]
            outside = np.linalg.norm(np.maximum(delta, 0.0), axis=-1)
            inside = np.minimum(np.max(delta, axis=-1), 0.0)
            margin = min(margin, float(np.min(outside + inside - radii)))
        return margin

    def collides(self, robot: Robot, q) -> bool:
        return self.clearance(robot, q) < 0.0

    def segment_free(self, robot: Robot, a, b, step: float = 0.05) -> bool:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        n = max(int(np.linalg.norm(b - a) / step), 1)
        for i in range(n + 1):
            if self.collides(robot, a + (b - a) * (i / n)):
                return False
        return True


def _link_samples(robot: Robot, q):
    origins = joint_origins(robot, q)
    points = []
    radii = []
    index = []
    for i in range(len(origins) - 1):
        a, b = origins[i], origins[i + 1]
        for t in np.linspace(0.0, 1.0, SAMPLES_PER_LINK):
            points.append(a + (b - a) * t)
            radii.append(robot.radii[i])
            index.append(i)
    return np.array(points), np.array(radii), np.array(index)


def random_scene(n_obstacles: int, rng: np.random.Generator,
                 inner: float = 0.25, outer: float = 0.85,
                 size: tuple = (0.06, 0.16)) -> Scene:
    obstacles = []
    while len(obstacles) < n_obstacles:
        angle = rng.uniform(0.0, 2 * np.pi)
        radius = rng.uniform(inner, outer)
        center = np.array([radius * np.cos(angle),
                           radius * np.sin(angle),
                           rng.uniform(0.10, 0.80)])
        half = rng.uniform(*size, size=3) / 2.0
        obstacles.append(Box(center, half))
    return Scene(obstacles)
