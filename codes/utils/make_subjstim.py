import numpy as np
import scipy.io
from tqdm import tqdm
import argparse
import os

def main():
    # 设置参数
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--featname",
        type=str,
        default='',
        help="Target variable",
    )
    parser.add_argument(
        "--use_stim",
        type=str,
        default='',
        help="ave or each",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    # 参数赋值
    opt = parser.parse_args()
    subject=opt.subject
    use_stim = opt.use_stim
    featname = opt.featname
    topdir = '../../nsdfeat/'
    savedir = f'{topdir}/subjfeat/'
    featdir = f'{topdir}/{featname}/'

    # NSD实验设计，通过索引来获取数据
    nsd_expdesign = scipy.io.loadmat('../../nsd/nsddata/experiments/nsd/nsd_expdesign.mat')

    # Note that most of them are 1-base index!
    # This is why I subtract 1
    sharedix = nsd_expdesign['sharedix'] -1 

    # 在make_subjmri数据预处理时保存的所有刺激和特有刺激
    if use_stim == 'ave':# 特有刺激
        stims = np.load(f'../../mrifeat/{subject}/{subject}_stims_ave.npy')
    else: # Each # 所有刺激
        stims = np.load(f'../../mrifeat/{subject}/{subject}_stims.npy')
    
    feats = []
    tr_idx = np.zeros(len(stims))

    # 读取刺激对应的大脑响应的特征
    for idx, s in tqdm(enumerate(stims)): 
        if s in sharedix:
            tr_idx[idx] = 0
        else:
            tr_idx[idx] = 1    
        # init_latent 或 c
        feat = np.load(f'{featdir}/{s:06}.npy')
        feats.append(feat)

    feats = np.stack(feats)    

    os.makedirs(savedir, exist_ok=True)

    feats_tr = feats[tr_idx==1,:]
    feats_te = feats[tr_idx==0,:]
    # 训练集的id
    np.save(f'../../mrifeat/{subject}/{subject}_stims_tridx.npy',tr_idx)
    # 训练集和测试集的特征
    np.save(f'{savedir}/{subject}_{use_stim}_{featname}_tr.npy',feats_tr)
    np.save(f'{savedir}/{subject}_{use_stim}_{featname}_te.npy',feats_te)


if __name__ == "__main__":
    main()
