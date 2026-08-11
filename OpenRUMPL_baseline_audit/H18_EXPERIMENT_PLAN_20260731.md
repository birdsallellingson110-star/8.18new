# H18: paper-aligned single-frame GBT screening

All arms use real Human3.6M detections, balanced random two-camera pairs,
single frames, no synthetic cameras, no temporal module, and no scene
centering. Formal evaluation averages all camera combinations for V2/V3/V4.

| Arm | Harmonic Plucker | Confidence | Loss | Token dropout |
|---|---|---|---|---|
| P0 | 15 frequencies | attention bias only | MSE | 0% |
| P1 | 15 frequencies | attention bias only | MSE | 20% |

Queued paper-centering follow-ups:

| Arm | H18 token setup | Triangulated single-frame center | Token dropout |
|---|---|---|---|
| P2 | yes | yes | 0% |
| P3 | yes | yes | 20% |

The comparison isolates whether paper-style harmonic ray tokens and training
objective repair the observed fixed-V2 to V3/V4 cardinality shift, and whether
token dropout only becomes useful after the token definition matches the
paper.
