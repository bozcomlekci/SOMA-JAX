"""tools/pipeline/demo_motion.py: Acclaim ASF/AMC kinematics, clips, the CMU ->
SOMA retarget, staging and the gallery, without the network (the demo itself
downloads CMU trials)."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from ._helpers import UPSTREAM_ASSETS, need_assets, need_module

ASF = """\
# a two-bone leg in CMU's conventions
:version 1.10
:name test
:units
  mass 1.0
  length 0.45
  angle deg
:root
   order TX TY TZ RX RY RZ
   axis XYZ
   position 0 0 0
   orientation 0 0 0
:bonedata
  begin
     id 1
     name lfemur
     direction 0.34202 -0.939693 0
     length 10
     axis 0 0 20  XYZ
    dof rx ry rz
    limits (-160.0 20.0)
           (-70.0 70.0)
           (-60.0 70.0)
  end
  begin
     id 2
     name ltibia
     direction 0 -1 0
     length 5
     axis 0 0 0  XYZ
    dof rx
    limits (-10.0 170.0)
  end
:hierarchy
  begin
    root lfemur
    lfemur ltibia
  end
"""
AMC = """\
:FULLY-SPECIFIED\r
:DEGREES\r
1\r
root 1 2 3 0 0 0\r
lfemur 0 0 0\r
ltibia 0\r
2\r
root 4 5 6 10 90 -20\r
lfemur 30 -15 5\r
ltibia 70\r
"""


def _demo():
    from soma_jax.assets import REPO_ROOT
    spec = importlib.util.spec_from_file_location(
        "_demo_motion", REPO_ROOT / "tools" / "pipeline" / "demo_motion.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_examples_and_capture_rates():
    """Every example names a CMU trial and a sensible window; trials of the
    subjects CMU captured at 60 fps (and trial 136_19) read at 60, the rest at 120."""
    dw = _demo()
    assert "walk" in dw.EXAMPLES and len(dw.EXAMPLES) >= 10
    for name, (trial, start, seconds, caption) in dw.EXAMPLES.items():
        subject, number = trial.split("_")
        assert subject.isdigit() and number.isdigit(), name
        assert start >= 0 and (seconds is None or seconds > 0) and caption, name
    assert dw.cmu_fps("35_08") == 120 and dw.cmu_fps("87_03") == 60
    assert dw.cmu_fps("136_19") == 60 and dw.cmu_fps("136_18") == 120
    assert dw.cmu_fps("my_walk") == 120


def test_load_clip_cuts_and_resamples(tmp_path):
    """--start / --seconds cut the trial at its capture rate, then the clip is
    resampled to the output rate (every 4th frame of 120 fps, every 2nd of 60)."""
    dw = _demo()
    (tmp_path / "leg.asf").write_text(ASF)
    lines = [":FULLY-SPECIFIED", ":DEGREES"]
    for i in range(240):                          # root x counts frames
        lines += [str(i + 1), f"root {i} 0 0 0 0 0", "lfemur 0 0 0", "ltibia 0"]
    (tmp_path / "leg.amc").write_text("\n".join(lines) + "\n")
    s = 0.0254 / 0.45
    asf, W, end, fps = dw.load_clip(tmp_path / "leg.asf", tmp_path / "leg.amc", 120.0, 30.0,
                                    start=0.5, seconds=1.0)
    assert fps == 30 and W["lfemur"].shape == (30, 3, 3)
    np.testing.assert_allclose(end["root"][:, 0] / s, np.arange(60, 180, 4))
    _, W, end, fps = dw.load_clip(tmp_path / "leg.asf", tmp_path / "leg.amc", 60.0, 30.0)
    assert fps == 30 and len(end["root"]) == 120
    with pytest.raises(SystemExit):
        dw.load_clip(tmp_path / "leg.asf", tmp_path / "leg.amc", 120.0, 30.0, start=5.0)


def test_stage_yaw():
    """A motion that travels turns to run along X, front rather than back to
    the camera; one in place turns to face 35 degrees past the camera, judged
    from the frames where the pelvis is upright (not mid-flip)."""
    from scipy.spatial.transform import Rotation
    dw = _demo()
    T, z = 20, np.array([0.0, 0.0, 1.0])

    def yaw(a):
        return Rotation.from_euler("y", a, degrees=True).as_matrix()

    path = np.linspace([0, 0.9, 0], [2.0, 0.9, -3.0], T)        # travels 3.6 m
    for facing, expect_x in ((np.degrees(np.arctan2(2.0, -3.0)), 1),   # walks forwards
                             (np.degrees(np.arctan2(2.0, -3.0)) + 90, None),  # sideways
                             (np.degrees(np.arctan2(2.0, -3.0)) - 90, None)):
        root = np.broadcast_to(yaw(facing), (T, 3, 3))
        G, travels = dw.stage_yaw(path, root, azimuth_deg=15.0)
        d = G @ (path[-1] - path[0])
        assert travels and abs(d[2]) < 1e-9 and abs(abs(d[0]) - np.hypot(2, 3)) < 1e-9
        if expect_x:
            assert d[0] > 0                                       # left to right
        assert (G @ root[0] @ z)[2] > -1e-9                       # never its back
    still = np.zeros((T, 3))
    root = np.broadcast_to(yaw(-120.0), (T, 3, 3)).copy()
    root[::3] = Rotation.from_euler("x", 180, degrees=True).as_matrix()   # upside down
    G, travels = dw.stage_yaw(still, root, azimuth_deg=15.0)
    face = G @ root[1] @ z
    assert not travels and np.degrees(np.arctan2(face[0], face[2])) == pytest.approx(50.0)


def test_strip_offsets():
    """Strip poses keep their order along the path, at least 0.8 median body
    widths apart: a motion in place lines up evenly, a fast one is left alone,
    and one travelling to -X spreads towards -X."""
    dw = _demo()
    np.testing.assert_allclose(dw.strip_offsets(np.zeros(5), np.ones(5)), [0, 0.8, 1.6, 2.4, 3.2])
    np.testing.assert_allclose(dw.strip_offsets([0.0, 1.0, 2.0], [0.5] * 3), [0, 0, 0])
    x = np.array([0.0, -0.1, -0.2])
    moved = x + dw.strip_offsets(x, [1.0] * 3)
    np.testing.assert_allclose(moved, [0.0, -0.8, -1.6])


def test_compose_gallery(tmp_path):
    """Clips of different lengths and rates loop side by side in gallery.gif,
    at the gallery's rate, for as long as the longest; gallery.png holds the strips."""
    Image = need_module("PIL.Image")
    dw = _demo()
    rng = np.random.default_rng(0)
    clips = [(f"clip {k}", [rng.integers(0, 255, (18, 32, 3), np.uint8) for _ in range(n)],
              rate, rng.integers(0, 255, (36, 64, 3), np.uint8))
             for k, (n, rate) in enumerate(((4, 10.0), (9, 30.0), (12, 20.0)))]
    written = dw.compose_gallery(clips, tmp_path, cols=2, tile=(32, 18), fps=10.0)
    assert {p.name for p in written} >= {"gallery.gif", "gallery.png"}
    gif = Image.open(tmp_path / "gallery.gif")
    assert gif.size == (64, 36) and gif.n_frames == 6          # 12 frames at 20 fps = 0.6 s
    assert Image.open(tmp_path / "gallery.png").size == (128, 72)


def test_asf_amc_forward_kinematics():
    """Bone ends follow W_b = W_parent C M C^T, checked against SciPy's
    extrinsic XYZ Euler angles (the ASF/AMC convention)."""
    from scipy.spatial.transform import Rotation
    dw = _demo()
    asf, frames = dw.parse_asf(ASF), dw.parse_amc(AMC)
    assert len(frames) == 2 and set(frames[1]) == {"root", "lfemur", "ltibia"}
    s = 0.0254 / 0.45
    assert asf["meters_per_unit"] == pytest.approx(s)
    W, end = dw.cmu_forward_kinematics(asf, frames)

    def R(xyz):
        return Rotation.from_euler("xyz", xyz, degrees=True).as_matrix()
    femur, tibia = asf["bones"]["lfemur"], asf["bones"]["ltibia"]
    for t, f in enumerate(frames):
        root = R(f["root"][3:])
        c = R(femur["axis"])
        w_femur = root @ c @ R(f["lfemur"]) @ c.T
        w_tibia = w_femur @ R([f["ltibia"][0], 0, 0])
        knee = f["root"][:3] * s + femur["length"] * s * w_femur @ femur["direction"]
        ankle = knee + tibia["length"] * s * w_tibia @ tibia["direction"]
        np.testing.assert_allclose(W["root"][t], root, atol=1e-12)
        np.testing.assert_allclose(end["lfemur"][t], knee, atol=1e-12)
        np.testing.assert_allclose(end["ltibia"][t], ankle, atol=1e-12)


@pytest.mark.assets
@pytest.mark.slow
def test_retarget_turns_bones_with_the_capture(layer_procedural):
    """Every captured bone turns at random and rests up to 25 degrees off
    SOMA's bone here. Each posed SOMA bone turns from its T-pose as its CMU
    bone turns from rest, and the legs, whose rest poses differ (CMU's rest 20
    degrees apart), point along the captured bones. At the capture's rest pose
    SOMA keeps its T-pose outside the legs: matching every rest bone instead
    tipped its neck back and pressed the chin into the throat."""
    need_module("pxr")
    need_assets(*UPSTREAM_ASSETS)
    import jax.numpy as jnp
    from scipy.spatial.transform import Rotation

    from soma_jax import SOMAParams
    dw = _demo()
    layer = layer_procedural
    names, parents = list(layer.public_joint_names), np.asarray(layer.parents)
    J, T = len(names), 3
    zero = layer(SOMAParams(poses=jnp.zeros((1, J, 3)), transl=jnp.zeros((1, 3)),
                            identity_coeffs=jnp.zeros((1, 128))), fk_only=True)
    tpose = np.asarray(zero.joints[0], np.float64)
    index = {n: i for i, n in enumerate(names)}

    rng = np.random.default_rng(0)
    asf = {"bones": {}, "parent": {}, "meters_per_unit": 1.0}
    W = {"root": Rotation.random(T, random_state=1).as_matrix()}
    end = {"root": rng.normal(size=(T, 3))}
    for k, (name, (bone, child)) in enumerate(dw.SOMA_FROM_CMU.items()):
        if child is None:
            continue
        rest = tpose[index[child]] - tpose[index[name]]
        tilt = Rotation.from_rotvec(rng.normal(size=3) * np.radians(25) / np.sqrt(3))
        direction = tilt.apply(rest / np.linalg.norm(rest))
        asf["bones"][bone] = {"direction": direction, "length": 0.4}
        asf["parent"][bone] = "root"
        W[bone] = Rotation.random(T, random_state=10 + k).as_matrix()
    for bone in W:                      # frame 0: the capture at rest, at the origin
        W[bone][0] = np.eye(3)
    end["root"][0] = 0.0
    local, hips, _ = dw.retarget_to_soma(asf, W, end, names, parents, tpose)
    out = layer(SOMAParams(poses=jnp.asarray(local, jnp.float32),
                           transl=jnp.asarray(hips, jnp.float32),
                           identity_coeffs=jnp.zeros((T, 128))), fk_only=True)
    joints = np.asarray(out.joints, np.float64)
    err = dw.bone_direction_error(joints, asf, W, np.eye(3), names, tpose)
    assert err < 0.05, f"posed bones off the capture by {err:.3f} degrees"

    legs = set()                        # the thighs and everything below them
    for j in range(J):
        if names[j] in ("LeftLeg", "RightLeg") or (parents[j] != j and parents[j] in legs):
            legs.add(j)
    rest_of_body = [j for j in range(J) if j not in legs]
    assert {"Neck2", "Head", "HeadEnd", "LeftArm"} <= {names[j] for j in rest_of_body}
    np.testing.assert_allclose(joints[0, rest_of_body], tpose[rest_of_body], atol=1e-5)
