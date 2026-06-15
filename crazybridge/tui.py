"""Textual TUI for driving the crazybridge node.

Spins a small rclpy node in a background thread that subscribes to the
bridge's odometry and holds service clients for takeoff, land and go_to.
Hotkeys:

    t           takeoff (uses takeoff_height_m / takeoff_duration_s params)
    l           land    (uses land_duration_s)
    w / s       nudge +x / -x by nudge_step_m
    a / d       nudge +y / -y
    r / f       nudge +z / -z
    z / x       yaw     +nudge_yaw_deg / -nudge_yaw_deg
    q          quit

Nudges fan out as relative go_to calls with goto_duration_s seconds each.
"""
from __future__ import annotations

import threading
from collections import deque
from queue import Empty, Queue

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry

from crazybridge_interfaces.srv import GoTo, Land, Takeoff

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header, Static


def _seconds_to_duration(seconds: float) -> Duration:
    d = Duration()
    d.sec = int(seconds)
    d.nanosec = int((seconds - int(seconds)) * 1e9)
    return d


class BridgeClient(Node):
    def __init__(self) -> None:
        super().__init__('crazybridge_tui')

        self.declare_parameter('odom_topic', '/crazybridge/odometry')
        self.declare_parameter('takeoff_srv', '/crazybridge/takeoff')
        self.declare_parameter('land_srv', '/crazybridge/land')
        self.declare_parameter('goto_srv', '/crazybridge/go_to')

        self.declare_parameter('takeoff_height_m', 1.0)
        self.declare_parameter('takeoff_duration_s', 2.0)
        self.declare_parameter('land_duration_s', 3.0)
        self.declare_parameter('goto_duration_s', 1.5)
        self.declare_parameter('nudge_step_m', 0.2)
        self.declare_parameter('nudge_yaw_deg', 15.0)

        odom_topic = self._sp('odom_topic')
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        self._takeoff_cli = self.create_client(Takeoff, self._sp('takeoff_srv'))
        self._land_cli = self.create_client(Land, self._sp('land_srv'))
        self._goto_cli = self.create_client(GoTo, self._sp('goto_srv'))

        self._state_lock = threading.Lock()
        self._latest_odom: Odometry | None = None
        self.events: Queue[str] = Queue()

    def _sp(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _dp(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _odom_cb(self, msg: Odometry) -> None:
        with self._state_lock:
            self._latest_odom = msg

    def snapshot(self) -> Odometry | None:
        with self._state_lock:
            return self._latest_odom

    def service_status(self) -> dict[str, bool]:
        return {
            'takeoff': self._takeoff_cli.service_is_ready(),
            'land': self._land_cli.service_is_ready(),
            'go_to': self._goto_cli.service_is_ready(),
        }

    def _on_response(self, label: str, future) -> None:
        try:
            r = future.result()
            self.events.put(
                f'{label} -> success={r.success}'
                + (f' msg="{r.message}"' if r.message else '')
            )
        except Exception as exc:
            self.events.put(f'{label} -> exception: {exc}')

    def call_takeoff(self) -> None:
        if not self._takeoff_cli.service_is_ready():
            self.events.put('takeoff: service not available')
            return
        req = Takeoff.Request()
        req.height = float(self._dp('takeoff_height_m'))
        req.duration = _seconds_to_duration(self._dp('takeoff_duration_s'))
        req.group_mask = 0
        fut = self._takeoff_cli.call_async(req)
        fut.add_done_callback(
            lambda f: self._on_response(f'takeoff h={req.height:.2f}', f)
        )

    def call_land(self) -> None:
        if not self._land_cli.service_is_ready():
            self.events.put('land: service not available')
            return
        req = Land.Request()
        req.height = 0.0
        req.duration = _seconds_to_duration(self._dp('land_duration_s'))
        req.group_mask = 0
        fut = self._land_cli.call_async(req)
        fut.add_done_callback(lambda f: self._on_response('land', f))

    def call_nudge(self, dx: float, dy: float, dz: float, dyaw_deg: float = 0.0) -> None:
        if not self._goto_cli.service_is_ready():
            self.events.put('go_to: service not available')
            return
        req = GoTo.Request()
        req.relative = True
        req.goal = Point(x=float(dx), y=float(dy), z=float(dz))
        req.yaw = float(dyaw_deg)
        req.duration = _seconds_to_duration(self._dp('goto_duration_s'))
        req.group_mask = 0
        fut = self._goto_cli.call_async(req)
        fut.add_done_callback(
            lambda f: self._on_response(
                f'nudge dx={dx:+.2f} dy={dy:+.2f} dz={dz:+.2f} dyaw={dyaw_deg:+.1f}',
                f,
            )
        )

    @property
    def step(self) -> float:
        return float(self._dp('nudge_step_m'))

    @property
    def yaw_step(self) -> float:
        return float(self._dp('nudge_yaw_deg'))


class CrazyBridgeTUI(App):
    CSS = """
    Screen { layout: vertical; }
    #status {
        height: 11;
        border: round green;
        padding: 0 1;
    }
    #log {
        border: round blue;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding('t', 'takeoff', 'Takeoff'),
        Binding('l', 'land', 'Land'),
        Binding('w', 'nudge_fwd', '+X'),
        Binding('s', 'nudge_back', '-X'),
        Binding('a', 'nudge_left', '+Y'),
        Binding('d', 'nudge_right', '-Y'),
        Binding('r', 'nudge_up', '+Z'),
        Binding('f', 'nudge_down', '-Z'),
        Binding('z', 'yaw_left', '+Yaw'),
        Binding('x', 'yaw_right', '-Yaw'),
        Binding('q', 'quit', 'Quit'),
    ]

    def __init__(self, client: BridgeClient) -> None:
        super().__init__()
        self._client = client
        self._log: deque[str] = deque(maxlen=14)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Static('', id='status'),
            Static('', id='log'),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.1, self._refresh)

    def _drain_events(self) -> None:
        while True:
            try:
                self._log.append(self._client.events.get_nowait())
            except Empty:
                return

    def _refresh(self) -> None:
        self._drain_events()
        odom = self._client.snapshot()
        svc = self._client.service_status()
        svc_line = '  '.join(
            f'{k}={"OK" if v else "--"}' for k, v in svc.items()
        )
        if odom is None:
            body = (
                'waiting for odometry...\n'
                f'step={self._client.step:.2f}m  yaw_step={self._client.yaw_step:.1f}deg\n'
                f'services: {svc_line}\n'
            )
        else:
            p = odom.pose.pose.position
            q = odom.pose.pose.orientation
            stamp = odom.header.stamp
            body = (
                f'pos   x={p.x:+8.3f}  y={p.y:+8.3f}  z={p.z:+8.3f}\n'
                f'quat  x={q.x:+8.3f}  y={q.y:+8.3f}  z={q.z:+8.3f}  w={q.w:+8.3f}\n'
                f'frame={odom.header.frame_id} child={odom.child_frame_id} '
                f'stamp={stamp.sec}.{stamp.nanosec:09d}\n'
                f'step={self._client.step:.2f}m  yaw_step={self._client.yaw_step:.1f}deg\n'
                f'services: {svc_line}\n'
            )
        self.query_one('#status', Static).update(body)
        self.query_one('#log', Static).update('\n'.join(self._log))

    def action_takeoff(self) -> None:
        self._client.call_takeoff()

    def action_land(self) -> None:
        self._client.call_land()

    def action_nudge_fwd(self) -> None:
        self._client.call_nudge(self._client.step, 0.0, 0.0)

    def action_nudge_back(self) -> None:
        self._client.call_nudge(-self._client.step, 0.0, 0.0)

    def action_nudge_left(self) -> None:
        self._client.call_nudge(0.0, self._client.step, 0.0)

    def action_nudge_right(self) -> None:
        self._client.call_nudge(0.0, -self._client.step, 0.0)

    def action_nudge_up(self) -> None:
        self._client.call_nudge(0.0, 0.0, self._client.step)

    def action_nudge_down(self) -> None:
        self._client.call_nudge(0.0, 0.0, -self._client.step)

    def action_yaw_left(self) -> None:
        self._client.call_nudge(0.0, 0.0, 0.0, dyaw_deg=self._client.yaw_step)

    def action_yaw_right(self) -> None:
        self._client.call_nudge(0.0, 0.0, 0.0, dyaw_deg=-self._client.yaw_step)


def main(args=None) -> None:
    rclpy.init(args=args)
    client = BridgeClient()
    executor = MultiThreadedExecutor()
    executor.add_node(client)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    app = CrazyBridgeTUI(client)
    try:
        app.run()
    finally:
        executor.shutdown()
        client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
