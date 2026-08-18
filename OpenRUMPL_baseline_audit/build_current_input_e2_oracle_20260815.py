#!/usr/bin/env python3
"""Append confidence/IRLS triangulation candidates to the current 11-candidate H76 cache.

This is deliberately an inference-only oracle diagnostic.  It does not use the
training target to generate candidates; targets are used only to report the
offline upper bound and therefore must never be used for model selection.
"""
import argparse, itertools, json
from pathlib import Path
import numpy as np
from diagnose_h76_candidate_pool_20260812 import ray_solver

COMBOS = tuple(c for k in (2, 3, 4) for c in itertools.combinations(range(4), k))

def action_equal(x, actions):
    names = np.unique(actions)
    return float(np.mean([x[actions == a].mean() for a in names]))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--irls-iters', type=int, default=3); p.add_argument('--huber-threshold-m', type=float, default=.03)
    a = p.parse_args(); src = np.load(a.input)
    pred, target, rays, actions = (src[k].astype(np.float32) for k in ('predictions','targets','rays','actions'))
    if pred.shape[1:] != (11,17,3): raise ValueError(f'expected 11 candidates, got {pred.shape}')
    conf = np.stack([ray_solver(rays, c, 'confidence', a.irls_iters, a.huber_threshold_m) for c in COMBOS], 1)
    irls = np.stack([ray_solver(rays, c, 'irls', a.irls_iters, a.huber_threshold_m) for c in COMBOS], 1)
    expanded = np.concatenate((pred, conf, irls), 1)
    errors = np.linalg.norm(expanded - target[:,None], axis=-1) * 1000
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'validation_current_e2_33c.npz', **{**{k:src[k] for k in src.files if k != 'predictions'}, 'predictions':expanded})
    rows = {}
    for k in (2,3,4):
        vals=[]
        for task in itertools.combinations(range(4), k):
            ids=[i for i,c in enumerate(COMBOS*3) if set(c).issubset(task)]
            vals.append(errors[:,ids].min(1))
        oracle=np.stack(vals,1).mean(1)
        rows[f'V{k}']={'oracle_action_equal_all17_mm':action_equal(oracle,actions),'oracle_frame_weighted_all17_mm':float(oracle.mean())}
    result={'input':str(Path(a.input).resolve()),'candidate_count':33,'candidate_order':[list(c) for c in COMBOS]*3,'oracle':rows}
    (out/'oracle.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result, indent=2))
if __name__=='__main__': main()
