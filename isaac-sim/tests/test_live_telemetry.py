from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import live_telemetry  # noqa: E402
from coordinate_mapper import map_trajectory_to_plane  # noqa: E402
from drawing_state_machine import build_motion_sequence  # noqa: E402
from run_replay import ReplayRun  # noqa: E402

SIM_CONFIG = "config/xarm7_drawing.yaml"
TRAJECTORY = {
    "drawing_id": "square_and_tick",
    "strokes": [
        {"stroke_id": 0, "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.7], [0.2, 0.7], [0.2, 0.2]]},
        {"stroke_id": 1, "points": [[0.35, 0.45], [0.45, 0.6], [0.7, 0.3]]},
    ],
}


def _write_trajectory(directory: Path) -> Path:
    path = directory / "square_and_tick.json"
    path.write_text(json.dumps(TRAJECTORY), encoding="utf-8")
    return path


class MappingTests(unittest.TestCase):
    def test_world_canvas_round_trip_matches_the_plane_mapping(self):
        for flip in (True, False):
            mapped = map_trajectory_to_plane(TRAJECTORY, center_xy=[0.45, 0.0], size_xy=[0.16, 0.12], flip_image_y=flip, max_step=1.0)
            mapping = mapped["mapping"]
            for stroke, source in zip(mapped["strokes"], TRAJECTORY["strokes"]):
                # Resampling keeps every source vertex, so the first and last map exactly.
                for world, uv in ((stroke["points"][0], source["points"][0]), (stroke["points"][-1], source["points"][-1])):
                    np.testing.assert_allclose(live_telemetry.canvas_to_world(mapping, uv), world, atol=1e-12)
                    np.testing.assert_allclose(live_telemetry.world_to_canvas(mapping, world), uv, atol=1e-12)
            for uv in ([0.0, 0.0], [1.0, 1.0], [0.31, 0.77]):
                back = live_telemetry.world_to_canvas(mapping, live_telemetry.canvas_to_world(mapping, uv))
                np.testing.assert_allclose(back, uv, atol=1e-12)

    def test_canvas_y_down_is_world_y_negative_when_flipped(self):
        mapping = map_trajectory_to_plane(TRAJECTORY, center_xy=[0.0, 0.0], size_xy=[0.2, 0.2], max_step=1.0)["mapping"]
        top = live_telemetry.canvas_to_world(mapping, [0.5, 0.2])
        bottom = live_telemetry.canvas_to_world(mapping, [0.5, 0.7])
        self.assertGreater(top[1], bottom[1])


class FileTests(unittest.TestCase):
    def test_atomic_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = live_telemetry.status_path(Path(tmp) / "live", "studio_x")
            self.assertIsNone(live_telemetry.read_status(path))
            self.assertTrue(live_telemetry.write_status(path, {"status": "drawing", "progress": 0.25}))
            self.assertTrue(live_telemetry.write_status(path, {"status": "finished", "progress": 1.0}))
            self.assertEqual(live_telemetry.read_status(path), {"status": "finished", "progress": 1.0})
            self.assertEqual(os.listdir(path.parent), [path.name])  # no temp file left behind
            path.write_text('{"status": "draw', encoding="utf-8")  # torn write
            self.assertIsNone(live_telemetry.read_status(path))

    def test_read_trace_resumes_and_skips_partial_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t_trace.ndjson"
            path.write_text('{"u": 0.1}\n{"u": 0.2}\n{"u": 0.', encoding="utf-8")
            samples, offset = live_telemetry.read_trace(path)
            self.assertEqual([s["u"] for s in samples], [0.1, 0.2])
            with open(path, "a", encoding="utf-8") as handle:
                handle.write('3}\n')
            more, _ = live_telemetry.read_trace(path, offset)
            self.assertEqual([s["u"] for s in more], [0.3])
            self.assertEqual(live_telemetry.read_trace(Path(tmp) / "missing", 7), ([], 7))


class LiveTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.mapping = map_trajectory_to_plane(TRAJECTORY, center_xy=[0.45, 0.0], size_xy=[0.16, 0.12], max_step=1.0)["mapping"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_trace_is_thinned_to_spacing_and_keeps_segment_ends(self):
        telemetry = live_telemetry.LiveTelemetry(self.dir, "p", self.mapping)
        t = 0.0
        for segment_start in (0.40, 0.46):
            for x in np.arange(0.0, 0.0101, 0.0002):  # 10 mm at 0.2 mm per tick
                t += 1 / 60
                telemetry.sample(t, "DRAW_STROKE_0", [segment_start + x, 0.0, 0.201], pen_down=True, stroke_id=0)
            t += 1 / 60
            telemetry.sample(t, "LIFT_PEN_STROKE_0", [segment_start + 0.0101, 0.0, 0.22], pen_down=False, stroke_id=0)
        telemetry.finish(t, "finished")
        samples, _ = live_telemetry.read_trace(telemetry.trace_file)
        for segment in (0, 1):
            xs = np.asarray([s["x"] for s in samples if s["segment"] == segment])
            self.assertGreaterEqual(len(xs), 8)
            self.assertLessEqual(len(xs), 12)
            self.assertTrue(np.all(np.diff(xs)[:-1] >= 0.001 - 1e-9))
            self.assertAlmostEqual(xs[-1] - xs[0], 0.0100, places=4)  # segment end kept
        self.assertEqual(len(live_telemetry.trace_polylines(samples)), 2)
        self.assertTrue(all(s["u"] is not None for s in samples))

    def test_status_written_at_period_progress_monotonic_and_final(self):
        telemetry = live_telemetry.LiveTelemetry(self.dir, "p", self.mapping, extra={"source": "test"})
        writes = 0
        for tick in range(1, 121):  # 2 s at 60 Hz
            progress = 0.5 if tick != 60 else 0.1  # one backwards blip
            writes += telemetry.sample(tick / 60, "DRAW_STROKE_0", [0.45, 0.0, 0.201], error_m=0.0004,
                                       progress=progress * tick / 120, pen_down=True, stroke_id=0)
        self.assertTrue(18 <= writes <= 21)
        status = live_telemetry.read_status(telemetry.status_file)
        self.assertEqual(status["status"], "drawing")
        self.assertEqual(status["source"], "test")
        self.assertAlmostEqual(status["error_mm"], 0.4)
        np.testing.assert_allclose(status["tip_canvas"], live_telemetry.world_to_canvas(self.mapping, [0.45, 0.0]), atol=1e-6)
        for key in ("prefix", "phase", "progress", "tip_world", "tip_canvas", "error_mm", "sim_time", "status", "updated_at"):
            self.assertIn(key, status)
        telemetry.finish(2.0, "failed", "IK failed")
        telemetry.sample(2.1, "DRAW_STROKE_0", [0.45, 0.0, 0.201], progress=0.9)
        final = live_telemetry.read_status(telemetry.status_file)
        self.assertEqual((final["status"], final["phase"], final["failure_reason"]), ("failed", "FAILED", "IK failed"))
        self.assertLessEqual(final["progress"], 0.5)

    def test_pen_down_progress_follows_draw_phases(self):
        mapped = map_trajectory_to_plane(TRAJECTORY, center_xy=[0.45, 0.0], size_xy=[0.16, 0.12], max_step=0.002)
        phases = build_motion_sequence(mapped["strokes"], pen_down_z=0.201, pen_up_z=0.235, approach_height=0.27, max_cartesian_step=0.002)
        progress = live_telemetry.PenDownProgress(phases)
        values = [progress.at(p, t) for p, phase in enumerate(phases) for t in range(len(phase.targets))]
        self.assertEqual(values[0], 0.0)
        self.assertAlmostEqual(values[-1], 1.0)
        self.assertTrue(np.all(np.diff(values) >= 0))
        first_lift = next(i for i, phase in enumerate(phases) if phase.state.startswith("LIFT"))
        self.assertTrue(0.6 < progress.at(first_lift, 0) < 0.95)  # the square is most of the ink


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.trajectory = _write_trajectory(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, **kwargs):
        return ReplayRun(self.trajectory, prefix="replay_t", live_dir=self.dir / "live", config=SIM_CONFIG, **kwargs)

    def test_progress_is_monotonic_and_finishes(self):
        run = self._run()
        progress, sim_time = [], 0.0
        while not run.finished:
            sim_time += 0.37
            status = run.step(sim_time)
            progress.append(status["progress"])
            self.assertEqual(live_telemetry.read_status(run.status_file)["progress"], status["progress"])
        self.assertTrue(np.all(np.diff(progress) >= 0))
        self.assertEqual(progress[-1], 1.0)
        final = run.status()
        self.assertEqual((final["status"], final["source"], final["mode"]), ("finished", "replay", "planned"))
        self.assertAlmostEqual(final["sim_time"], run.path.duration_s, places=3)
        samples, _ = live_telemetry.read_trace(run.trace_file)
        uv = np.asarray([[s["u"], s["v"]] for s in samples])
        self.assertTrue(np.all((uv >= 0.2 - 1e-6) & (uv <= 0.8 + 1e-6)))
        self.assertEqual({s["stroke_id"] for s in samples}, {0, 1})

    def test_background_thread_finishes_and_stop_marks_failed(self):
        run = self._run(speedup=1000.0, tick_s=0.01).start()
        run.join(10)
        self.assertFalse(run.running)
        self.assertEqual(run.status()["status"], "finished")

        slow = self._run(speedup=0.01, tick_s=0.01).start()
        self.assertEqual(slow.status()["status"], "drawing")
        slow.stop()
        self.assertEqual((slow.status()["status"], slow.status()["failure_reason"]), ("failed", "replay stopped"))

    def test_frames_follow_recorded_progress(self):
        frames_dir = self.dir / "frames"
        for camera in ("WideFront", "PaperTop"):
            (frames_dir / camera).mkdir(parents=True)
            for index in range(60):
                (frames_dir / camera / f"{index:05d}.jpg").write_bytes(b"")
        # A recorded run: 1 s approach, then the same square drawn over 6 s at 60 Hz.
        csv_path = self.dir / "run_executed_path.csv"
        rows = ["simulation_time_s,state,stroke_id,waypoint_index,pen_down,tip_x_m,tip_y_m,tracking_error_m"]
        for tick in range(60):
            rows.append(f"{tick / 60:.4f},APPROACH_STROKE_0,0,0,0,0.45,0.0,0.002")
        for tick in range(360):
            rows.append(f"{1 + tick / 60:.4f},DRAW_STROKE_0,0,{tick},1,{0.40 + 0.1 * tick / 360:.6f},0.0,0.0005")
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        run = self._run(frames_dir=frames_dir, executed_csv=csv_path, frame_every=4)
        indices, sim_time = [], 0.0
        while not run.finished:
            sim_time += 0.5
            frame = run.step(sim_time)["frame"]
            self.assertEqual(set(frame), {"WideFront", "PaperTop"})
            indices.append(int(Path(frame["PaperTop"]).stem))
        self.assertTrue(np.all(np.diff(indices) >= 0))
        self.assertEqual(indices[-1], 59)
        self.assertEqual(run.status()["mode"], "recorded")
        self.assertIn(run.status()["error_mm"], (0.5, 2.0))


def _load_server():
    """Import the MCP server with a stand-in FastMCP when `mcp` is not installed."""
    try:
        import mcp.server.fastmcp  # noqa: F401
    except ImportError:
        class FastMCP:
            def __init__(self, *args, **kwargs):
                pass

            def tool(self, *args, **kwargs):
                return lambda fn: fn

            def run(self):
                raise RuntimeError("stub")

        for name in ("mcp", "mcp.server", "mcp.server.fastmcp"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["mcp.server.fastmcp"].FastMCP = FastMCP
    spec = importlib.util.spec_from_file_location("wildtrace_sim_server_test", PROJECT_ROOT / "mcp_server" / "wildtrace_sim_server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DrawTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _load_server()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.trajectory = _write_trajectory(self.dir)
        self.calls = []
        server = self.server
        server._LAUNCHES.clear()
        patches = [
            mock.patch.object(server, "MCP_SESSIONS_DIR", self.dir / "sessions"),
            mock.patch.object(server, "VIDEO_ROOT_WSL", self.dir / "videos"),
            mock.patch.object(server, "VIDEO_OUT", self.dir / "video_out"),
            mock.patch.object(server, "STUDIO_WAIT_S", 0),
            mock.patch.object(server, "POLL_S", 0.05),
            mock.patch.object(server, "sim_start", lambda wait_s=300: json.dumps({"status": "already_running"})),
            mock.patch.object(server, "setup_scene", lambda **kw: json.dumps({"status": "already_built"})),
            mock.patch.object(server, "_scene_config", lambda: server.TWIN_CONFIG),
            mock.patch.object(server, "_ensure_policy_server", lambda checkpoint, args=(): self.calls.append(("policy", checkpoint)) or {"status": "already_running"}),
            mock.patch.object(server, "_attach_capture", lambda prefix, config, every: self.calls.append(("capture", prefix)) or {"status": "success"}),
            mock.patch.object(server, "_detach_capture", lambda: self.calls.append(("detach",))),
            mock.patch.object(server, "_set_timeline", lambda command, wanted, attempts=2: self.calls.append((command,)) or True),
            mock.patch.object(server, "_isaac", lambda *a, **kw: {"status": "error", "message": "no Isaac in tests"}),
            mock.patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu-24.04"}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        (self.dir / "sessions").mkdir()
        self.addCleanup(self.tmp.cleanup)

    def test_starts_in_background_and_writes_the_job(self):
        reply = json.loads(self.server.draw_trajectory(str(self.trajectory), title="Square"))
        self.assertEqual(reply["status"], "started")
        prefix = reply["prefix"]
        self.assertRegex(prefix, r"^studio_[0-9a-f]{8}$")
        self.assertEqual((reply["stroke_count"], reply["point_count"], reply["cameras"]), (2, 8, ["WideFront", "PaperTop"]))
        self.server._LAUNCH_THREADS[prefix].join(10)

        job = json.loads((self.dir / "sessions" / "job.json").read_text(encoding="utf-8"))
        self.assertEqual(job["prefix"], prefix)
        self.assertEqual(job["controller"], "act")
        self.assertEqual(job["policy_address"], f"127.0.0.1:{self.server.ACT_PORT}")
        self.assertEqual(job["title"], "Square")
        self.assertEqual(job["trajectory_path"], "\\\\wsl.localhost\\Ubuntu-24.04" + str(self.trajectory.resolve()).replace("/", "\\"))
        self.assertIn(("policy", self.server.ACT_CHECKPOINTS[self.server.TWIN_CONFIG]), self.calls)
        self.assertIn(("capture", prefix), self.calls)
        self.assertEqual([c for c in self.calls if c[0].startswith("simulation.")], [("simulation.stop",), ("simulation.play",)])
        self.assertFalse(self.server._RUN_LOCK.locked())

        # The controller's live status and report, as run_status reports them.
        status = json.loads(self.server.run_status(prefix))
        self.assertIsNone(status["telemetry"])  # the controller has not started (no Isaac)
        live = live_telemetry.LiveTelemetry(live_telemetry.live_dir(self.dir / "sessions"), prefix)
        live.finish(42.0, "finished")
        (self.dir / "sessions" / f"{prefix}_run_metrics.json").write_text(json.dumps({
            "status": "finished", "stroke_count": 2, "simulation_duration_s": 42.0,
            "desired_to_executed_nearest_path_error": {"rmse_m": 0.0004, "mean_error_m": 0.0003, "max_error_m": 0.001},
        }), encoding="utf-8")
        frames = self.dir / "videos" / prefix / "PaperTop"
        frames.mkdir(parents=True)
        for index in range(3):
            (frames / f"{index:05d}.jpg").write_bytes(b"")
        status = json.loads(self.server.run_status(prefix))
        self.assertEqual(status["status"], "finished")
        self.assertEqual(status["telemetry"]["progress"], 1.0)
        self.assertEqual(status["metrics"]["rmse_mm"], 0.4)
        self.assertEqual(status["cameras"]["PaperTop"]["frame_count"], 3)
        self.assertEqual(status["cameras"]["PaperTop"]["latest_frame"], str(frames / "00001.jpg"))
        self.assertEqual(status["cameras"]["WideFront"]["frame_count"], 0)
        self.assertEqual(status["job"]["title"], "Square")

    def test_ik_without_video_detaches_capture(self):
        reply = json.loads(self.server.draw_trajectory(str(self.trajectory), controller="IK", record_video=False))
        self.server._LAUNCH_THREADS[reply["prefix"]].join(10)
        job = json.loads((self.dir / "sessions" / "job.json").read_text(encoding="utf-8"))
        self.assertEqual(job["controller"], "ik")
        self.assertNotIn("policy_address", job)
        self.assertIn(("detach",), self.calls)
        self.assertFalse(any(c[0] in ("policy", "capture") for c in self.calls))

    def test_rejects_bad_input_and_concurrent_runs(self):
        self.assertEqual(json.loads(self.server.draw_trajectory(str(self.dir / "nope.json")))["status"], "failed")
        bad = self.dir / "bad.json"
        bad.write_text(json.dumps({"drawing_id": "x", "strokes": []}), encoding="utf-8")
        self.assertIn("invalid trajectory", json.loads(self.server.draw_trajectory(str(bad)))["reason"])
        self.assertEqual(json.loads(self.server.draw_trajectory(str(self.trajectory), controller="rl"))["status"], "failed")
        self.assertEqual(json.loads(self.server.run_status("../etc"))["status"], "failed")
        self.server._RUN_LOCK.acquire()
        try:
            self.assertEqual(json.loads(self.server.draw_trajectory(str(self.trajectory)))["status"], "busy")
            self.assertEqual(json.loads(self.server.draw("Cat"))["status"], "busy")
        finally:
            self.server._RUN_LOCK.release()
        self.assertFalse((self.dir / "sessions" / "job.json").exists())

    def test_launch_failure_is_reported_in_live_status(self):
        with mock.patch.object(self.server, "setup_scene", lambda **kw: json.dumps({"status": "failed", "step": "scene.clear"})):
            reply = json.loads(self.server.draw_trajectory(str(self.trajectory)))
            self.server._LAUNCH_THREADS[reply["prefix"]].join(10)
        status = json.loads(self.server.run_status(reply["prefix"]))
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["telemetry"]["phase"], "LAUNCH")
        self.assertIn("setup_scene", status["telemetry"]["failure_reason"])

    def test_sync_wait_and_unc_input(self):
        unc = "\\\\wsl.localhost\\Ubuntu-24.04" + str(self.trajectory).replace("/", "\\")
        reply = json.loads(self.server.draw_trajectory(unc, wait_s=1))
        self.assertEqual(reply["status"], "running")  # no Isaac, so no report within the wait
        self.assertEqual(reply["trajectory_path"], str(self.trajectory.resolve()))
        self.assertFalse(self.server._RUN_LOCK.locked())

    def test_draw_still_queues_gold_jobs(self):
        def provision(category, allow_repeat=False):
            job = {"trajectory_path": "C:\\x.json", "prefix": "cat_12345678"}
            (self.dir / "sessions" / "job.json").write_text(json.dumps(job), encoding="utf-8")
            return {"status": "ok", "category": "Cat", "sample_id": "12345678abcd", **job}

        with mock.patch.object(self.server, "provision", provision):
            reply = json.loads(self.server.draw("Cat", wait_s=0))
        self.assertEqual(reply, {"status": "running", "category": "Cat", "sample_id": "12345678abcd", "prefix": "cat_12345678"})
        job = json.loads((self.dir / "sessions" / "job.json").read_text(encoding="utf-8"))
        self.assertEqual((job["controller"], job["prefix"], job["trajectory_path"]), ("ik", "cat_12345678", "C:\\x.json"))
        self.assertIn(("detach",), self.calls)


if __name__ == "__main__":
    unittest.main()
