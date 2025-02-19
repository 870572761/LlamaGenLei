# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
from PIL import Image
import glob
import argparse
import os
import sys
import json
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from utils.distributed import init_distributed_mode
from dataset.augmentation import center_crop_arr
from tokenizer.tokenizer_image.vq_model import VQ_models


#################################################################################
#                             Training Helper Functions                         #
#################################################################################
class CustomDataset(Dataset):
    def __init__(self, data_json_file, start, end, transform):
        img_path_list = []
        print(f"data_json_file is {data_json_file}")
        with open(data_json_file, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                if end==idx:
                    break
                if idx>=start:
                    # 解析每一行的 JSON 数据
                    try:
                        data = json.loads(line.strip())
                        # 打印或处理数据
                        # print(f"Name: {data['name']}, Image Path: {data['img_path']}")
                        img_path_list.append((data['img_path'], data['name']))
                    except json.JSONDecodeError as e:
                        print(f"无法解析 JSON 行: {line.strip()}. 错误: {e}")
        self.img_path_list = img_path_list
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path, code_name = self.img_path_list[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, code_name

def save_rank_to_separate_files(output_dir="output", base_filename="rank", content=""):
    # 获取当前进程的 rank 和世界大小
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 构造文件名，包含 rank 信息
    output_file = os.path.join(output_dir, f"{base_filename}_{rank}.txt")

    # 写入 rank 到对应的文件
    with open(output_file, "a") as f:
        f.write(f"{content}")

#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    # dist.init_process_group("nccl")
    init_distributed_mode(args)
    rank = dist.get_rank()
    print(f"rank: {rank}")
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup a feature folder:
    if rank == 0:
        os.makedirs(args.code_path, exist_ok=True)

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint


    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    print(f"Dataset is preparing...")
    dataset = CustomDataset(args.data_path, args.data_start, args.data_end, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=1, # important!
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"Dataset contains {len(dataset):,} images")

    # 每个进程保存 rank 到独立文件
    total = 0
    for img, code_name in loader:
        img = img.to(device)
        code_name = code_name[0]
        with torch.no_grad():
            _, _, [_, _, indices] = vq_model.encode(img)
        codes = indices.reshape(img.shape[0], -1)
        x = codes.detach().cpu().numpy()    # (1, args.image_size//16 * args.image_size//16)
        np.save(os.path.join(args.code_path, '{}.npy'.format(code_name)), x)

        total += dist.get_world_size()
        print(code_name)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--data-start", type=int, required=True)
    parser.add_argument("--data-end", type=int, required=True)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=512)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    args = parser.parse_args()
    main(args)
