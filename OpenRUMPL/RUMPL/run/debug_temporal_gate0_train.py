#!/usr/bin/env python3
"""One real train window: compare H76 and gate-zero temporal with V=2."""
import os, sys, torch
from torchvision import transforms
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from core.config import config, update_config
import dataset
from dataset.temporal_h36m_rumpl import TemporalH36MRUMPL, collate_temporal_h36m
from models.multiview_rumpl import get_multiview_rumpl_net
from models.temporal_gbt_rumpl import TemporalJointViewRUMPL

def main():
    os.environ.update(RUMPL_TRI_ANCHOR='1', RUMPL_TRI_ANCHOR_REG='1e-4', RUMPL_TRI_ANCHOR_CONF_EPS='0.05', RUMPL_ANCHOR_CENTERED_RAYS='1', RUMPL_INPUT_PLUCKER='1', RUMPL_INPUT_HARMONIC_L='0')
    for n in ('GBT_GLOBAL_JV_DEPTH','GBT_LEARNABLE_BIAS','RUMPL_GBT_SET_DECODER','RUMPL_RELATIVE_VIEW_FUSION','RUMPL_GEOMETRY_UNCERTAINTY_TOKEN','RUMPL_PER_JOINT_RESIDUAL_GATE','RUMPL_POST_PFT_GRAPH_RESIDUAL','RUMPL_JOINT_SPECIFIC_HEAD'): os.environ[n]='0'
    update_config('/mnt/data/cjyoutput/open_source_fusion_audit_20260731/H35_a1d_h21_refined_rumpl_tri_anchor.yaml')
    config.DATASET.TRAIN_MMPOSE_TYPE='mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap'; config.DATASET.USE_MMPOSE_TRAIN=True
    base=get_multiview_rumpl_net(config,is_train=False)
    p='/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_2026-08-03_03-00-38/model_best.pth.tar'
    s=torch.load(p,map_location='cpu'); s=s.get('state_dict',s); s={(k[7:] if k.startswith('module.') else k):v for k,v in s.items()}; base.load_state_dict(s,strict=False)
    d=dataset.multiview_h36m_rumpl(config,'train',True,transforms.ToTensor()); t=TemporalH36MRUMPL(d,9,5); loader=DataLoader(t,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_temporal_h36m); _,_,target,rays,_,_=next(iter(loader)); dev=torch.device('cuda:0'); base=base.to(dev).eval(); w=TemporalJointViewRUMPL(base,3,8,False,0.2, residual_gate=True).to(dev).eval(); w.global_gate.data.zero_(); rays=rays.to(dev); target=target.to(dev)
    torch.manual_seed(123)
    with torch.no_grad():
        out, idx=w(rays,num_views=2); print('idx',idx.tolist()); print('gate',float(w.global_gate)); print('out_mm',float((out-target).norm(dim=-1).mean()*1000)); print('latest_mm',float((out[:,-1]-target[:,-1]).norm(dim=-1).mean()*1000)); print('out_abs',float(out.abs().max()),'target_abs',float(target.abs().max()));
        flat=[]
        for ti in range(9): flat.append(base(rays[:,ti,:,idx[0]],is_training=False))
        flat=torch.stack(flat,1); print('identity_max',float((out-flat).abs().max()),'flat_mm',float((flat-target).norm(dim=-1).mean()*1000))
if __name__=='__main__': main()
