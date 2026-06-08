"""
Verification tests for the differentiable optical-flow backward pass
(`GaussianRasterizerWithFlow`).

Requires a CUDA GPU and the built extension (`pip install -e .`). Run with:

    python tests/test_flow_backward.py

Covers the three checks from docs/flow-forward-pass.md:
  1. Zero-flow  : prev params == current params  -> flow == 0 and all prev grads == 0
  2. Translation: shift prev_means3D            -> foreground flow is nonzero & coherent
  3. Finite-difference gradient check on means3D and prev_means3D under a flow loss.

These use float32 (the kernels are single precision), so the FD check uses a
loose tolerance rather than torch.autograd.gradcheck (which assumes float64).
"""

import math
import torch

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    rasterize_gaussians_with_flow,
)

DEVICE = "cuda"


def make_settings(H=64, W=64, device_id=0):
    fovx = fovy = math.radians(60.0)
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)

    # Camera at origin looking down +z (column-major 4x4, as the lib expects).
    viewmatrix = torch.eye(4, device=DEVICE)
    znear, zfar = 0.01, 100.0
    P = torch.zeros(4, 4, device=DEVICE)
    P[0, 0] = 1.0 / tanfovx
    P[1, 1] = 1.0 / tanfovy
    P[2, 2] = zfar / (zfar - znear)
    P[3, 2] = -(zfar * znear) / (zfar - znear)
    P[2, 3] = 1.0
    projmatrix = viewmatrix @ P

    return GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros(3, device=DEVICE),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=torch.zeros(3, device=DEVICE),
        prefiltered=False,
        device_id=device_id,
    )


def make_scene(n=8, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    means3D = torch.randn(n, 3, generator=g, device=DEVICE) * 0.5
    means3D[:, 2] = means3D[:, 2].abs() + 4.0  # in front of the camera
    scales = torch.rand(n, 3, generator=g, device=DEVICE) * 0.1 + 0.05
    rotations = torch.zeros(n, 4, device=DEVICE)
    rotations[:, 0] = 1.0  # identity quaternion
    opacities = torch.rand(n, 1, generator=g, device=DEVICE) * 0.5 + 0.5
    colors = torch.rand(n, 3, generator=g, device=DEVICE)
    return means3D, scales, rotations, opacities, colors


def render(means3D, prev_means3D, scales, prev_scales, rotations, prev_rotations,
           opacities, colors, settings):
    n = means3D.shape[0]
    empty = torch.zeros(0, device=DEVICE)
    means2D = torch.zeros_like(means3D, requires_grad=True)
    color, radii, depth, flow = rasterize_gaussians_with_flow(
        means3D, prev_means3D, means2D, empty, colors, opacities,
        scales, prev_scales, rotations, prev_rotations, empty, empty, settings)
    return color, depth, flow, radii


def test_zero_flow():
    settings = make_settings()
    means3D, scales, rotations, opacities, colors = make_scene()
    means3D = means3D.clone().requires_grad_(True)
    prev_means3D = means3D.detach().clone().requires_grad_(True)
    prev_scales = scales.clone().requires_grad_(True)
    prev_rotations = rotations.clone().requires_grad_(True)

    color, depth, flow, radii = render(
        means3D, prev_means3D, scales, prev_scales, rotations, prev_rotations,
        opacities, colors, settings)

    max_flow = flow.abs().max().item()
    print(f"[zero-flow] max|flow| = {max_flow:.3e} (expect ~0)")
    assert max_flow < 1e-3, "flow should vanish when prev == curr"

    loss = flow.pow(2).sum()
    loss.backward()
    for name, t in [("prev_means3D", prev_means3D), ("prev_scales", prev_scales),
                    ("prev_rotations", prev_rotations)]:
        gmax = 0.0 if t.grad is None else t.grad.abs().max().item()
        print(f"[zero-flow] max|d {name}| = {gmax:.3e} (expect ~0)")
        assert gmax < 1e-3, f"{name} grad should vanish at zero flow"
    print("[zero-flow] OK")


def test_translation():
    settings = make_settings()
    means3D, scales, rotations, opacities, colors = make_scene()
    shift = torch.tensor([0.05, 0.0, 0.0], device=DEVICE)
    prev_means3D = means3D + shift  # previous frame was shifted in +x

    color, depth, flow, radii = render(
        means3D, prev_means3D, scales, scales, rotations, rotations,
        opacities, colors, settings)

    fg = color.sum(0) > 1e-3  # foreground mask
    mean_fx = flow[0][fg].mean().item()
    mean_fy = flow[1][fg].mean().item()
    print(f"[translate] foreground mean flow = ({mean_fx:.3f}, {mean_fy:.3f}) px")
    assert abs(mean_fx) > 1e-2, "x-flow should be nonzero under x-translation"
    print("[translate] OK")


def _flow_loss(target_grad, *tensors_and_args):
    means3D, prev_means3D, scales, prev_scales, rotations, prev_rotations, \
        opacities, colors, settings = tensors_and_args
    _, _, flow, _ = render(means3D, prev_means3D, scales, prev_scales,
                           rotations, prev_rotations, opacities, colors, settings)
    return (flow * target_grad).sum()


def test_finite_difference(eps=1e-3, tol=2e-2):
    settings = make_settings()
    means3D0, scales, rotations, opacities, colors = make_scene(seed=1)
    prev_means3D0 = means3D0 + torch.tensor([0.04, -0.03, 0.0], device=DEVICE)

    torch.manual_seed(0)
    target = torch.randn(2, settings.image_height, settings.image_width, device=DEVICE)

    for name, base, is_prev in [("means3D", means3D0, False),
                                ("prev_means3D", prev_means3D0, True)]:
        means3D = means3D0.clone().requires_grad_(True)
        prev_means3D = prev_means3D0.clone().requires_grad_(True)
        param = prev_means3D if is_prev else means3D
        loss = _flow_loss(target, means3D, prev_means3D, scales, scales,
                          rotations, rotations, opacities, colors, settings)
        loss.backward()
        ana = param.grad.clone()

        # finite difference on a few random entries
        idxs = [(0, 0), (base.shape[0] // 2, 1), (base.shape[0] - 1, 0)]
        max_rel = 0.0
        for (i, j) in idxs:
            p = base.clone(); p[i, j] += eps
            if is_prev:
                lp = _flow_loss(target, means3D0, p, scales, scales, rotations,
                                rotations, opacities, colors, settings).item()
                p2 = base.clone(); p2[i, j] -= eps
                lm = _flow_loss(target, means3D0, p2, scales, scales, rotations,
                                rotations, opacities, colors, settings).item()
            else:
                lp = _flow_loss(target, p, prev_means3D0, scales, scales, rotations,
                                rotations, opacities, colors, settings).item()
                p2 = base.clone(); p2[i, j] -= eps
                lm = _flow_loss(target, p2, prev_means3D0, scales, scales, rotations,
                                rotations, opacities, colors, settings).item()
            fd = (lp - lm) / (2 * eps)
            denom = max(1.0, abs(fd))
            rel = abs(fd - ana[i, j].item()) / denom
            max_rel = max(max_rel, rel)
            print(f"[FD {name}] ({i},{j}) analytic={ana[i,j].item():+.4f} fd={fd:+.4f} rel={rel:.3f}")
        assert max_rel < tol, f"{name}: FD mismatch {max_rel:.3f} > {tol}"
        print(f"[FD {name}] OK (max rel {max_rel:.3f})")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "these tests require a CUDA GPU"
    test_zero_flow()
    test_translation()
    test_finite_difference()
    print("\nAll flow-backward tests passed.")
