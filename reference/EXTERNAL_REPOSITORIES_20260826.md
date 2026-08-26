# External reference repositories

These repositories are literature references, not runtime dependencies of the
Stage-3 CMU-to-H36M pipeline. They are separate Git working trees locally, so
the parent repository intentionally does not commit them as broken gitlinks.
Clone the pinned revisions below only if their reference implementations are
needed.

| Local directory | Upstream | Pinned commit |
|---|---|---|
| `PointDSC-official` | <https://github.com/XuyangBai/PointDSC.git> | `b009d536ac10b570853833f2178397c154745da9` |
| `Probabilistic-Monocular-3D-Human-Pose-Estimation-with-Normalizing-Flows` | <https://github.com/twehrbein/Probabilistic-Monocular-3D-Human-Pose-Estimation-with-Normalizing-Flows.git> | `ad2fdf21da1dfb689a5c2a531ebda6e23d95ccc0` |
| `SimVQ-official` | <https://github.com/youngsheen/SimVQ.git> | `d8bd94d9ca30273c8bab1b4194d8e1dc0b9d3f70` |
| `differentiable-ransac-official` | <https://github.com/weitong8591/differentiable_ransac.git> | `d128128c1038f1eb8fbde94c2779bfc41092e994` |

Example:

```bash
git clone https://github.com/XuyangBai/PointDSC.git reference/PointDSC-official
git -C reference/PointDSC-official checkout b009d536ac10b570853833f2178397c154745da9
```
