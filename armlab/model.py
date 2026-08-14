from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DHLink:
    d: float
    a: float
    alpha: float
    lower: float
    upper: float


UR5E = (
    DHLink(0.1625, 0.0, np.pi / 2, -np.pi, np.pi),
    DHLink(0.0, -0.425, 0.0, -np.pi, np.pi),
    DHLink(0.0, -0.3922, 0.0, -np.pi, np.pi),
    DHLink(0.1333, 0.0, np.pi / 2, -np.pi, np.pi),
    DHLink(0.0997, 0.0, -np.pi / 2, -np.pi, np.pi),
    DHLink(0.0996, 0.0, 0.0, -np.pi, np.pi),
)


@dataclass(frozen=True)
class Robot:
    links: tuple = UR5E
    radii: tuple = (0.075, 0.075, 0.06, 0.05, 0.05, 0.05)
    name: str = "UR5e"

    @property
    def n(self) -> int:
        return len(self.links)

    @property
    def lower(self) -> np.ndarray:
        return np.array([l.lower for l in self.links])

    @property
    def upper(self) -> np.ndarray:
        return np.array([l.upper for l in self.links])

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.lower, self.upper)

    def within_limits(self, q: np.ndarray) -> bool:
        return bool(np.all(q >= self.lower) and np.all(q <= self.upper))

    def random_configuration(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lower, self.upper)


def dh_transform(theta: float, link: DHLink) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(link.alpha), np.sin(link.alpha)
    return np.array([
        [ct, -st * ca, st * sa, link.a * ct],
        [st, ct * ca, -ct * sa, link.a * st],
        [0.0, sa, ca, link.d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def link_transforms(robot: Robot, q) -> list[np.ndarray]:
    q = np.asarray(q, dtype=float)
    out = []
    t = np.eye(4)
    for theta, link in zip(q, robot.links):
        t = t @ dh_transform(theta, link)
        out.append(t.copy())
    return out


def forward_kinematics(robot: Robot, q) -> np.ndarray:
    return link_transforms(robot, q)[-1]


def joint_origins(robot: Robot, q) -> np.ndarray:
    frames = link_transforms(robot, q)
    points = [np.zeros(3)]
    points.extend(t[:3, 3] for t in frames)
    return np.array(points)


def geometric_jacobian(robot: Robot, q) -> np.ndarray:
    frames = link_transforms(robot, q)
    p_end = frames[-1][:3, 3]

    z_axes = [np.array([0.0, 0.0, 1.0])]
    origins = [np.zeros(3)]
    for t in frames[:-1]:
        z_axes.append(t[:3, 2])
        origins.append(t[:3, 3])

    jac = np.zeros((6, robot.n))
    for i in range(robot.n):
        jac[:3, i] = np.cross(z_axes[i], p_end - origins[i])
        jac[3:, i] = z_axes[i]
    return jac


def manipulability(robot: Robot, q) -> float:
    jac = geometric_jacobian(robot, q)
    return float(np.sqrt(max(np.linalg.det(jac @ jac.T), 0.0)))


def condition_number(robot: Robot, q) -> float:
    sv = np.linalg.svd(geometric_jacobian(robot, q), compute_uv=False)
    return float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else float("inf")


def pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    position = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    r = a[:3, :3].T @ b[:3, :3]
    skew = np.array([r[2, 1] - r[1, 2],
                     r[0, 2] - r[2, 0],
                     r[1, 0] - r[0, 1]])
    return position, float(np.arctan2(np.linalg.norm(skew) / 2.0,
                                      (np.trace(r) - 1.0) / 2.0))
