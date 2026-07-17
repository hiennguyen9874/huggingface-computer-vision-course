# Unit 8 — 3D Computer Vision

Small, executable implementations matching `chapters-vi/unit08.md` and
`chapters-en/unit8/`. Every file is independent, states its tensor contract,
and prints each important intermediate shape from `__main__`.

The paper-inspired neural networks preserve the core data flow for teaching;
they are intentionally small random-weight models, not pretrained paper
reproductions or substitutes for Depth Anything V2, PixelNeRF, or Zero123.

## Design map

| File | Section / module | Main contract |
|---|---|---|
| `transformations_3d.py` | homogeneous translation, scale, rotation | `[P,3] -> [4,P] -> [P,3]` |
| `pinhole_camera.py` | intrinsics, extrinsics, pixel projection | world `[P,3] -> pixels [P,2]` |
| `representations_3d.py` | point cloud, mesh, voxel grid, sphere SDF | representation-specific queries |
| `stereo_reconstruction.py` | rectified projection and triangulation | two `[P,2] -> XYZ [P,3]` |
| `depth_objectives.py` | invariant loss, valid mask, depth metrics | depth `[N,H,W] -> scalars` |
| `dpt_depth.py` | tiny DPT-style monocular depth model | RGB `[N,3,H,W] -> depth [N,H,W]` |
| `pixelnerf.py` | image-conditioned radiance field | image + queries -> RGB/density |
| `zero123_conditioning.py` | viewpoint-conditioned latent denoising | image + pose + latent -> noise |
| `nerf.py` | rays, Fourier encoding, field, volume rendering | rays `[R,3] -> RGB/depth` |

Notation: `N` batch, `P` points, `R` rays, `S` samples per ray, `Q` field
queries, `C` channels, and `H/W` image height/width. Coordinates and neural
inputs are floating point; masks are boolean and mesh/pair indices are int64.

## Run

Run one lesson from the repository root:

```bash
uv run python codes/unit08/nerf.py
```

Run every lesson:

```bash
for file in codes/unit08/*.py; do uv run python "$file"; done
```

The demos use fixed random seeds where neural weights are involved, require no
downloads, and run on CPU.
