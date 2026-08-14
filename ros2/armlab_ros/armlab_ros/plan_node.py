import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from armlab.collision import random_scene
from armlab.model import Robot, joint_origins, link_transforms
from armlab.planner import rrt_connect, straight_line

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow",
               "wrist_1", "wrist_2", "wrist_3"]

TRACED = (2, 3, 4, 5, 6)

TRACE_COLOURS = {
    2: (0.35, 0.55, 1.00),
    3: (0.45, 0.85, 0.95),
    4: (0.55, 0.95, 0.60),
    5: (0.95, 0.85, 0.35),
    6: (0.30, 0.95, 0.50),
}

TRACE_LABELS = {2: "elbow", 3: "wrist 1", 4: "wrist 2",
                5: "wrist 3", 6: "tool"}


def quaternion_from_matrix(r: np.ndarray):
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return ((r[2, 1] - r[1, 2]) * s, (r[0, 2] - r[2, 0]) * s,
                (r[1, 0] - r[0, 1]) * s, 0.25 / s)
    i = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
    q = [0.0, 0.0, 0.0, 0.0]
    q[3] = (r[k, j] - r[j, k]) / s
    q[i] = 0.25 * s
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    return tuple(q)


def segment_pose(a: np.ndarray, b: np.ndarray):
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        return (a + b) / 2.0, (0.0, 0.0, 0.0, 1.0), 1e-6
    z = axis / length
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 \
        else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return (a + b) / 2.0, quaternion_from_matrix(np.column_stack([x, y, z])), \
        length


class PlanNode(Node):

    def __init__(self):
        super().__init__("armlab_plan")

        self.declare_parameter("obstacles", 20)
        self.declare_parameter("seed", 0)
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("speed", 0.6)
        self.declare_parameter("frame_id", "world")

        self.frame = self.get_parameter("frame_id").value
        seed = self.get_parameter("seed").value
        self.robot = Robot()
        rng = np.random.default_rng(seed)
        self.scene = random_scene(
            self.get_parameter("obstacles").value, rng)

        start, goal = self._sample_query(rng)
        result = rrt_connect(self.robot, self.scene, start, goal, rng)
        if not result.success:
            raise RuntimeError("no plan found, try another seed")

        self.trajectory = self._densify(result.path)
        self.index = 0
        self.sweep = len(self.trajectory) // 2
        outbound = self.trajectory[:self.sweep + 1]
        self.traces = {i: [joint_origins(self.robot, q)[i] for q in outbound]
                       for i in TRACED}
        self.drawn = 1

        self.get_logger().info(
            f"planned in {result.plan_ms:.0f} ms, "
            f"{result.iterations} iterations, "
            f"{len(result.path)} waypoints, "
            f"joint length {result.joint_length:.2f} rad "
            f"(raw {result.raw_joint_length:.2f})")

        self.pub_arm = self.create_publisher(MarkerArray, "~/arm", 1)
        self.pub_scene = self.create_publisher(MarkerArray, "~/obstacles", 1)
        self.pub_path = self.create_publisher(MarkerArray, "~/joint_paths", 1)
        self.pub_joints = self.create_publisher(JointState, "~/joint_states",
                                                10)
        self.tf = TransformBroadcaster(self)

        rate = self.get_parameter("rate_hz").value
        self.timer = self.create_timer(1.0 / rate, self.step)

    def _sample_free(self, rng):
        for _ in range(5000):
            q = self.robot.random_configuration(rng)
            if not self.scene.collides(self.robot, q):
                return q
        raise RuntimeError("could not sample a free configuration")

    def _sample_query(self, rng, attempts: int = 60):
        pair = None
        for _ in range(attempts):
            pair = (self._sample_free(rng), self._sample_free(rng))
            if not straight_line(self.robot, self.scene, *pair).success:
                return pair
        self.get_logger().warn(
            "no query the straight-line baseline fails on, showing an easy one")
        return pair

    def _densify(self, path):
        speed = float(self.get_parameter("speed").value)
        rate = float(self.get_parameter("rate_hz").value)
        step = speed / rate
        out = []
        for a, b in zip(path, path[1:]):
            n = max(int(np.linalg.norm(b - a) / step), 1)
            out.extend(a + (b - a) * (i / n) for i in range(n))
        out.append(path[-1])
        return out + out[-2:0:-1]

    def step(self):
        q = self.trajectory[self.index]
        self.index = (self.index + 1) % len(self.trajectory)
        if self.index == 1:
            self.drawn = 1
        elif self.index <= self.sweep:
            self.drawn = self.index + 1

        now = self.get_clock().now().to_msg()
        origins = joint_origins(self.robot, q)

        state = JointState()
        state.header.stamp = now
        state.name = JOINT_NAMES
        state.position = [float(v) for v in q]
        self.pub_joints.publish(state)

        for i, t in enumerate(link_transforms(self.robot, q)):
            tf = TransformStamped()
            tf.header.stamp = now
            tf.header.frame_id = self.frame
            tf.child_frame_id = f"link_{i + 1}"
            tf.transform.translation.x = float(t[0, 3])
            tf.transform.translation.y = float(t[1, 3])
            tf.transform.translation.z = float(t[2, 3])
            x, y, z, w = quaternion_from_matrix(t[:3, :3])
            tf.transform.rotation.x = float(x)
            tf.transform.rotation.y = float(y)
            tf.transform.rotation.z = float(z)
            tf.transform.rotation.w = float(w)
            self.tf.sendTransform(tf)

        self.pub_arm.publish(self._arm_markers(origins, now))
        self.pub_scene.publish(self._scene_markers(now))

        self.pub_path.publish(self._path_markers(now))

    def _arm_markers(self, origins, stamp) -> MarkerArray:
        out = MarkerArray()
        for i in range(len(origins) - 1):
            center, quat, length = segment_pose(origins[i], origins[i + 1])
            m = Marker()
            m.header.frame_id = self.frame
            m.header.stamp = stamp
            m.ns = "links"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = \
                (float(v) for v in center)
            m.pose.orientation.x, m.pose.orientation.y, \
                m.pose.orientation.z, m.pose.orientation.w = \
                (float(v) for v in quat)
            radius = self.robot.radii[i]
            m.scale.x = m.scale.y = float(2 * radius)
            m.scale.z = float(max(length, 1e-3))
            m.color = ColorRGBA(r=0.25, g=0.6, b=0.95, a=0.9)
            out.markers.append(m)

        for i, p in enumerate(origins):
            m = Marker()
            m.header.frame_id = self.frame
            m.header.stamp = stamp
            m.ns = "joints"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = \
                (float(v) for v in p)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.09
            m.color = ColorRGBA(r=0.95, g=0.75, b=0.2, a=1.0)
            out.markers.append(m)
        return out

    def _scene_markers(self, stamp) -> MarkerArray:
        out = MarkerArray()
        for i, box in enumerate(self.scene.obstacles):
            m = Marker()
            m.header.frame_id = self.frame
            m.header.stamp = stamp
            m.ns = "obstacles"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = \
                (float(v) for v in box.center)
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = \
                (float(2 * v) for v in box.half)
            m.color = ColorRGBA(r=0.85, g=0.3, b=0.3, a=0.55)
            out.markers.append(m)
        return out

    def _path_markers(self, stamp) -> MarkerArray:
        out = MarkerArray()
        for i in TRACED:
            r, g, b = TRACE_COLOURS[i]
            m = Marker()
            m.header.frame_id = self.frame
            m.header.stamp = stamp
            m.ns = TRACE_LABELS[i]
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.014 if i == TRACED[-1] else 0.008
            m.color = ColorRGBA(r=r, g=g, b=b,
                                a=1.0 if i == TRACED[-1] else 0.8)
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                        for p in self.traces[i][:self.drawn]]
            out.markers.append(m)
        return out


def main():
    rclpy.init()
    node = PlanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
