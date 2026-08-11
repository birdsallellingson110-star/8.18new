#!/usr/bin/env python3
# 用 opencv 抽帧, 和官方 ffmpeg(-start_number 0)同样按帧号顺序对齐 GT。
# 为省盘只写每 skip 帧(真实帧号命名), 且只写"有GT且非空(1个body)"的帧。
# 输出: <seq>/hdImgs/00_<cam>/00_<cam>_<frame:08d>.jpg  (与 preprocess 期望一致)
import cv2, os, json, glob, argparse, numpy as np

def gt_frames_with_body(seq_dir):
    """返回有 GT 且 bodies>=1 的帧号集合"""
    s=set()
    for jf in glob.glob(os.path.join(seq_dir,'hdPose3d_stage1_coco19','body3DScene_*.json')):
        try:
            d=json.load(open(jf))
            if len(d.get('bodies',[]))>=1:
                s.add(int(jf.split('_')[-1].split('.')[0]))
        except Exception: pass
    return s

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', default='/mnt/data/cjydata/cmu_singleperson')
    ap.add_argument('--seqs', nargs='+', default=['171204_pose5','171204_pose6'])
    ap.add_argument('--cams', nargs='+', default=['03','06','12','13','23'])
    ap.add_argument('--skip', type=int, default=64)   # 与官方 skip_step 一致
    ap.add_argument('--max-frames', type=int, default=0, help='限连续帧数(时序评估用,取前N有body帧); 0=不限')
    args=ap.parse_args()
    for seq in args.seqs:
        sd=os.path.join(args.root,seq)
        gtf=gt_frames_with_body(sd)
        keep=sorted(f for f in gtf if f % args.skip == 0)
        if args.max_frames > 0:
            keep = keep[:args.max_frames]
        print(f"[{seq}] 有body的GT帧 {len(gtf)}, 每{args.skip}帧保留 {len(keep)} (帧号 {keep[0] if keep else '-'}..{keep[-1] if keep else '-'})")
        keepset=set(keep)
        for cam in args.cams:
            vid=os.path.join(sd,'hdVideos',f'hd_00_{cam}.mp4')
            outd=os.path.join(sd,'hdImgs',f'00_{cam}'); os.makedirs(outd,exist_ok=True)
            if len(glob.glob(os.path.join(outd,'*.jpg'))) >= len(keep):
                print(f"  cam{cam}: 已抽 {len(keep)} 帧, 跳过"); continue
            if not os.path.exists(vid): print(f"  [skip] 无视频 {vid}"); continue
            cap=cv2.VideoCapture(vid); i=0; w=0
            while True:
                ret=cap.grab()                  # grab 不解码, 快进
                if not ret: break
                if i in keepset:
                    ok,frame=cap.retrieve()     # 只在需要时解码
                    if ok:
                        cv2.imwrite(os.path.join(outd,f'00_{cam}_{i:08d}.jpg'),frame,[cv2.IMWRITE_JPEG_QUALITY,100]); w+=1
                i+=1
            cap.release()
            print(f"  cam{cam}: 解码{i}帧, 写出{w}张 -> {outd}")
    print("=== 抽帧完成 ===")

if __name__=='__main__': main()
