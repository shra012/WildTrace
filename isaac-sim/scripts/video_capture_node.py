"""ScriptNode: record a drawing run from several cameras for videos.

Attach next to the drawing controller (OnPlaybackTick -> ScriptNode); the
wildtrace-sim MCP server's draw(record_video=True) does this. Every `every`
ticks it saves one JPEG per camera, and it keeps a live "ink" BasisCurves
under /World/LiveInk that follows the pen tip whenever the pen is on the
sheet, so the drawing appears on camera as it is drawn (the controller only
authors its result curves at the end of a run).

Everything geometry-specific comes from the capture job written by the MCP
server from the active scene config (outputs/mcp_sessions/capture_job.json):
    {"out_dir": "C:\\\\wildtrace\\\\videos\\\\cat_70bc30f8", "every": 4,
     "cameras": ["/World/VideoCams/WideFront", "/World/VideoCams/PaperTop"],
     "resolution": [1280, 720], "robot_prim": "/World/xarm7",
     "sheet_xy": [x0, x1, y0, y1], "pen_down_z": [z_min, z_max], "ink_z": z}

State lives on `builtins`, and every module-level name is prefixed
_wtcap_/_WTCAP_ because ScriptNodes can share one globals dict (a plain
JOB_FILE here once clobbered the controller's Path JOB_FILE); delete
builtins._wt_capture to reset between runs.
"""
import builtins
import json
import os
from concurrent.futures import ThreadPoolExecutor

_WTCAP_JOB_FILE = r"\\wsl.localhost\Ubuntu-24.04\home\shravan\Workspace\WildTrace\isaac-sim\outputs\mcp_sessions\capture_job.json"
_WTCAP_INK_PRIM = "/World/LiveInk"


def _wtcap_find_pen_tip(stage, robot_prim):
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(robot_prim) and prim.GetName() == "pen_tip":
            return path
    raise RuntimeError(f"No pen_tip prim under {robot_prim}")


def _wtcap_init():
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.prims import XFormPrim

    with open(_WTCAP_JOB_FILE, encoding="utf-8") as handle:
        job = json.load(handle)
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(_WTCAP_INK_PRIM):
        stage.RemovePrim(_WTCAP_INK_PRIM)
    cams = []
    for cam in job["cameras"]:
        name = cam.rsplit("/", 1)[-1]
        os.makedirs(os.path.join(job["out_dir"], name), exist_ok=True)
        render_product = rep.create.render_product(cam, tuple(job.get("resolution", [1280, 720])))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([render_product])
        cams.append((name, annotator))
    state = type("CaptureState", (), {})()
    state.job, state.cams, state.tick, state.frame = job, cams, 0, 0
    state.tip = XFormPrim(_wtcap_find_pen_tip(stage, job.get("robot_prim", "/World/xarm7")))
    state.strokes, state.down = [], False
    state.pool = ThreadPoolExecutor(max_workers=3)
    builtins._wt_capture = state
    print(f"[Capture] recording {len(cams)} cameras every {job['every']} ticks -> {job['out_dir']}")
    return state


def _wtcap_update_ink(state):
    from pxr import Gf, UsdGeom
    import omni.usd

    strokes = [s for s in state.strokes if len(s) >= 2]
    if not strokes:
        return
    stage = omni.usd.get_context().get_stage()
    ink_z = float(state.job["ink_z"])
    curve = UsdGeom.BasisCurves.Define(stage, _WTCAP_INK_PRIM)
    curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    curve.CreateCurveVertexCountsAttr().Set([len(s) for s in strokes])
    curve.CreatePointsAttr().Set([Gf.Vec3f(x, y, ink_z) for s in strokes for x, y in s])
    curve.CreateWidthsAttr().Set([0.0018])
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curve.CreateDisplayColorAttr().Set([Gf.Vec3f(0.03, 0.03, 0.03)])


def _wtcap_save(path, data):
    from PIL import Image

    Image.fromarray(data[..., :3]).save(path, quality=92)


def setup(db):
    if hasattr(builtins, "_wt_capture"):
        del builtins._wt_capture


def compute(db):
    state = getattr(builtins, "_wt_capture", None) or _wtcap_init()
    state.tick += 1

    if state.tick % 2 == 0:
        positions, _ = state.tip.get_world_poses()
        x, y, z = (float(v) for v in positions[0])
        x0, x1, y0, y1 = state.job["sheet_xy"]
        z_min, z_max = state.job["pen_down_z"]
        down = z_min < z < z_max and x0 < x < x1 and y0 < y < y1
        if down:
            if not state.down:
                state.strokes.append([])
            state.strokes[-1].append((x, y))
        state.down = down

    if state.tick % int(state.job["every"]) == 0:
        _wtcap_update_ink(state)
        for name, annotator in state.cams:
            data = annotator.get_data()
            if data is not None and getattr(data, "size", 0):
                path = os.path.join(state.job["out_dir"], name, f"{state.frame:05d}.jpg")
                state.pool.submit(_wtcap_save, path, data.copy())
        state.frame += 1
    return True
