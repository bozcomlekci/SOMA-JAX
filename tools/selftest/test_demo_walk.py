"""tools/pipeline/demo_walk.py: Acclaim ASF/AMC kinematics and the CMU -> SOMA
retarget, without the network (the demo itself downloads a CMU trial)."""
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
        "_demo_walk", REPO_ROOT / "tools" / "pipeline" / "demo_walk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
def test_retarget_follows_the_captured_bones(layer_procedural):
    """Posed SOMA bones point where the captured bones do, although the source
    rests differently (each rest direction here is tilted up to 25 degrees, as
    CMU's thighs rest 20 degrees apart) and every bone turns at random."""
    need_module("pxr")
    need_assets(*UPSTREAM_ASSETS)
    import jax.numpy as jnp
    from soma_jax import SOMAParams
    from scipy.spatial.transform import Rotation
    dw = _demo()
    layer = layer_procedural
    names, parents = list(layer.public_joint_names), layer.parents
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
        end[bone] = end["root"] + W[bone] @ direction
    local, hips, _ = dw.retarget_to_soma(asf, W, end, names, parents, tpose)
    out = layer(SOMAParams(poses=jnp.asarray(local, jnp.float32),
                           transl=jnp.asarray(hips, jnp.float32),
                           identity_coeffs=jnp.zeros((T, 128))), fk_only=True)
    err = dw.bone_direction_error(np.asarray(out.joints, np.float64), asf, end, np.eye(3), names)
    assert err < 0.05, f"posed bones off the capture by {err:.3f} degrees"
