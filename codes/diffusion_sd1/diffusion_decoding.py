import h5py
from PIL import Image
import scipy.io
import argparse, os
import pandas as pd
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import trange
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import sys
sys.path.append("../utils/")
from nsd_access.nsda import NSDAccess
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

'''加载ckpt模型'''
def load_model_from_config(config, ckpt, gpu, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda(f"cuda:{gpu}")
    model.eval()
    return model

'''加载COCO图像'''
def load_img_from_arr(img_arr):
    image = Image.fromarray(img_arr).convert("RGB")
    w, h = 512, 512
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def main():
    ### 添加参数
    parser = argparse.ArgumentParser()
    # 图像id
    parser.add_argument(
        "--imgidx",
        required=True,
        type=int,
        help="img idx"
    )
    # 运行gpu
    parser.add_argument(
        "--gpu",
        required=True,
        type=int,
        help="gpu"
    )
    # 随机数种子
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    # 主体subject
    parser.add_argument(
        "--subject",
        required=True,
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )
    # 算法流程
    parser.add_argument(
        "--method",
        required=True,
        type=str,
        help="cvpr or text or gan",
    )

    # Set parameters
    # 设置参数
    opt = parser.parse_args()
    seed_everything(opt.seed)
    imgidx = opt.imgidx
    gpu = opt.gpu
    method = opt.method
    subject=opt.subject
    gandir = f'../../decoded/gan_recon_img/all_layers/{subject}/streams/'
    captdir = f'../../decoded/{subject}/captions/'

    # Load NSD information
    # 加载NSD实验设计信息( ../../ 相对于当前文件的上一级的上一级 )
    nsd_expdesign = scipy.io.loadmat('../../nsd/nsddata/experiments/nsd/nsd_expdesign.mat')

    # Note that mos of them are 1-base index!
    # This is why I subtract 1
    # 以1为基准的索引
    sharedix = nsd_expdesign['sharedix'] -1 

    # 获取刺激（图像）数据集
    nsda = NSDAccess('../../nsd/')
    sf = h5py.File(nsda.stimuli_file, 'r')
    sdataset = sf.get('imgBrick')

    stims_ave = np.load(f'../../mrifeat/{subject}/{subject}_stims_ave.npy')

    # 拆分数据集（训练+测试）
    tr_idx = np.zeros_like(stims_ave)
    for idx, s in enumerate(stims_ave):
        if s in sharedix:
            tr_idx[idx] = 0
        else:
            tr_idx[idx] = 1

    # Load Stable Diffusion Model
    # 加载 sd-v1-4 模型和参数
    config = './stable-diffusion/configs/stable-diffusion/v1-inference.yaml'
    ckpt = './stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt'
    config = OmegaConf.load(f"{config}")
    torch.cuda.set_device(gpu)
    model = load_model_from_config(config, f"{ckpt}", gpu)

    # 设置超参数
    n_samples = 1
    ddim_steps = 50 #采样步数
    ddim_eta = 0.0
    strength = 0.8
    scale = 5.0
    n_iter = 5
    precision = 'autocast'
    precision_scope = autocast if precision == "autocast" else nullcontext
    batch_size = n_samples
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 图像输出路径
    outdir = f'../../decoded/image-{method}/{subject}/'
    os.makedirs(outdir, exist_ok=True)

    # 采样方法设置
    sample_path = os.path.join(outdir, f"samples")
    os.makedirs(sample_path, exist_ok=True)
    precision = 'autocast'
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    # Load z (Image)
    # 加载 z(早期视觉图像特征)
    imgidx_te = np.where(tr_idx==0)[0][imgidx] # Extract test image index
    idx73k= stims_ave[imgidx_te]
    Image.fromarray(np.squeeze(sdataset[idx73k,:,:,:]).astype(np.uint8)).save(
        os.path.join(sample_path, f"{imgidx:05}_org.png"))    
    
    if method in ['cvpr','text']:
        # 提取早期视觉特征 z
        roi_latent = 'early'
        scores_latent = np.load(f'../../decoded/{subject}/{subject}_{roi_latent}_scores_init_latent.npy')
        imgarr = torch.Tensor(scores_latent[imgidx,:].reshape(4,40,40)).unsqueeze(0).to('cuda')

        # Generate image from Z
        # 从z开始重建图像
        precision_scope = autocast if precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    # 解码z -> 粗解码图像Xz
                    x_samples = model.decode_first_stage(imgarr)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    
                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        # resized copy
        im = Image.fromarray(x_sample.astype(np.uint8)).resize((512,512))
        im = np.array(im)

    elif method == 'gan':
        ganpath = f'{gandir}/recon_image_normalized-VGG19-fc8-{subject}-streams-{imgidx:06}.tiff'
        im = Image.open(ganpath).resize((512,512))
        im = np.array(im)

    # 编码Xz -> 潜在表征
    init_image = load_img_from_arr(im).to('cuda')
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    # Load c (Semantics)
    # 加载c（语义文本特征）
    if method == 'cvpr':
        # 提取腹侧视觉
        roi_c = 'ventral'
        scores_c = np.load(f'../../decoded/{subject}/{subject}_{roi_c}_scores_c.npy')
        carr = scores_c[imgidx,:].reshape(77,768)
        c = torch.Tensor(carr).unsqueeze(0).to('cuda')
    elif method in ['text','gan']:
        captions = pd.read_csv(f'{captdir}/captions_brain.csv', sep='\t',header=None)
        c = model.get_learned_conditioning(captions.iloc[imgidx][0]).to('cuda')

    # Generate image from Z (image) + C (semantics)
    # 从 Z+C 共同重建图像
    base_count = 0
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for n in trange(n_iter, desc="Sampling"):
                    uc = model.get_learned_conditioning(batch_size * [""])

                    # encode (scaled latent)
                    # 编码潜在表征-> zT(扩散过程)
                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                    # decode it
                    #编码后的zT和关联文本c共同进入解码得到Zc（逆向扩散）
                    samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,)

                    # 解码Zc -> 最终图像Xzc
                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    # 保存重建图像
                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(
                        os.path.join(sample_path, f"{imgidx:05}_{base_count:03}.png"))    
                    base_count += 1

'''主程序'''
if __name__ == "__main__":
    main()
