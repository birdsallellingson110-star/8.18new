#!/usr/bin/env python3
# 在抽好的 CMU 帧上跑 HRNet(td-hm_hrnet-w32_384x288), 输出与官方/乖崽一致的 per-frame json:
#   <out>/<seq>/00_<cam>/00_<cam>_<frame>.json = [{keypoints,keypoint_scores,bbox,bbox_score}, ...]
# 单人序列: 整图检测通常只得 1 个人, 后续 preprocess mmpose 模式做(平凡的)匹配。
import os, json, glob, argparse, numpy as np
os.environ.setdefault('TORCH_HOME','/mnt/data/dataset/c2i/torch')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', default='/mnt/data/cjydata/cmu_singleperson')
    ap.add_argument('--out',  default='/mnt/data/cjydata/cmu_singleperson/MPL_data/mmpose_outputs')
    ap.add_argument('--seqs', nargs='+', default=['171204_pose5','171204_pose6'])
    ap.add_argument('--cams', nargs='+', default=['03','06','12','13','23'])
    ap.add_argument('--pose2d', default='td-hm_hrnet-w32_8xb64-210e_coco-384x288')
    args=ap.parse_args()
    from mmpose.apis import MMPoseInferencer
    inferencer=MMPoseInferencer(pose2d=args.pose2d, device='cuda:0')
    for seq in args.seqs:
        for cam in args.cams:
            imgdir=os.path.join(args.root,seq,'hdImgs',f'00_{cam}')
            imgs=sorted(glob.glob(os.path.join(imgdir,'*.jpg')))
            if not imgs: print(f"[skip] {seq}/00_{cam} 无图"); continue
            od=os.path.join(args.out,seq,f'00_{cam}'); os.makedirs(od,exist_ok=True)
            n=0
            for im in imgs:
                jf=os.path.join(od, os.path.basename(im).replace('.jpg','.json'))
                if os.path.exists(jf): n+=1; continue
                res=next(inferencer(im, return_vis=False))
                preds=res['predictions'][0]   # list of persons
                out=[]
                for p in preds:
                    out.append({
                        'keypoints': np.array(p['keypoints']).tolist(),
                        'keypoint_scores': np.array(p['keypoint_scores']).tolist(),
                        'bbox': np.array(p['bbox']).reshape(-1).tolist() if 'bbox' in p else [],
                        'bbox_score': float(p.get('bbox_score',1.0)),
                    })
                json.dump(out, open(jf,'w')); n+=1
            print(f"[{seq}/00_{cam}] HRNet 完成 {n} 帧 -> {od}")
    print("=== HRNet 全部完成 ===")

if __name__=='__main__': main()
