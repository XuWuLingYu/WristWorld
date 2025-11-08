## WristWorld: Generating Wrist-Views via 4D World Models for Robotic Manipulation

- Paper: [arXiv](https://arxiv.org/html/2510.07313v1)
- Project page: [wrist-world.github.io](https://wrist-world.github.io/)

### Overview
WristWorld is a two-stage 4D world model that synthesizes realistic wrist-view videos from anchor views for robotic manipulation. It closes the gap between abundant third-person (anchor) views and scarce wrist-view recordings by explicitly modeling geometry and temporal consistency.

- Reconstruction stage: extends VGGT with a wrist head and proposes Spatial Projection Consistency (SPC) to predict geometrically consistent wrist poses and point clouds, plus wrist-view projections used as condition maps.
- Generation stage: a diffusion transformer (DiT) synthesizes wrist-view videos conditioned on wrist-view projections and CLIP-encoded anchor-view semantics, producing temporally coherent and geometrically faithful sequences.

Results on Droid, Calvin, and Franka Panda show state-of-the-art wrist-view video generation and improved VLA performance.

### Repository layout
- Stage 1 — Reconstruction (VGGT + Wrist head): `stage1_reconstruction/`
  - Installation, training and commands are documented in `stage1_reconstruction/README.md`.
- Stage 2 — Generation (Video DiT): `stage2_generation/`
  - Environment and usage are documented in `stage2_generation/README.md`.

### Installation
- For the reconstruction stage, follow the official VGGT training branch instructions to set up dependencies and environment: https://github.com/facebookresearch/vggt/tree/training
- For the generation stage, see `stage2_generation/requirements.txt` and `stage2_generation/README.md`.

### Quick start
- Reconstruction: see `stage1_reconstruction/README.md` for environment setup and training commands (multiple configs provided for Droid/Calvin/RealBot variants).
- Generation: see `stage2_generation/README.md` for preparing condition maps and running the wrist-view video generator.

### Checklist
- [x] Open-sourced inference code
- [x] Released weights
- [ ] Open-sourced training code

### Acknowledgements
This repository builds upon and is inspired by VGGT and related geometry-grounded vision works. We thank the community for open-sourcing their research and tools.

### Citation
If you find WristWorld useful, please cite our paper:

```
@article{qian2025wristworld,
  title   = {WristWorld: Generating Wrist-Views via 4D World Models for Robotic Manipulation},
  author  = {Qian, Zezhong and Chi, Xiaowei and Li, Yuming and Wang, Shizun and Qin, Zhiyuan and Ju, Xiaozhu and Han, Sirui and Zhang, Shanghang},
  journal = {arXiv preprint arXiv:2510.07313},
  year    = {2025}
}
```