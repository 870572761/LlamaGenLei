import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import argparse
import os
import json
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from utils.distributed import init_distributed_mode
from language.t5 import T5Embedder

CAPTION_KEY = {
    'blip': 0,
    'llava': 1,
    'llava_first': 2,
}
#################################################################################
#                             Training Helper Functions                         #
#################################################################################
class CustomDataset(Dataset):
    def __init__(self, data_json_file, start, end):
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
                        img_path_list.append((data['text'], data['name']))
                    except json.JSONDecodeError as e:
                        print(f"无法解析 JSON 行: {line.strip()}. 错误: {e}")
        self.img_path_list = img_path_list

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        caption, code_name = self.img_path_list[index]
        return caption, code_name


        
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
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup a feature folder:
    if rank == 0:
        os.makedirs(args.t5_path, exist_ok=True)

    # Setup data:
    print(f"Dataset is preparing...")
    dataset = CustomDataset(args.data_path, args.data_start, args.data_end)
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

    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    assert os.path.exists(args.t5_model_path)
    t5_xxl = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_model_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision
    )

    count = 0
    for caption, code_name in loader:
        code_name = code_name[0]
        out_file = os.path.join(args.t5_path, '{}.npy'.format(code_name))
        if not os.path.isfile(out_file):
            caption_embs, emb_masks = t5_xxl.get_text_embeddings(caption)
            valid_caption_embs = caption_embs[:, :emb_masks.sum()]
            x = valid_caption_embs.to(torch.float32).detach().cpu().numpy()
            np.save(out_file, x)
            count += 1
            print(code_name)
            with open(f"./rank_{rank}.txt", 'w', encoding='utf-8') as file:
                file.write(code_name + '\n')

    dist.destroy_process_group()
    print(f"{rank}Finish!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--t5-path", type=str, required=True)
    parser.add_argument("--data-start", type=int, required=True)
    parser.add_argument("--data-end", type=int, required=True)
    parser.add_argument("--caption-key", type=str, default='blip', choices=list(CAPTION_KEY.keys()))
    parser.add_argument("--trunc-caption", action='store_true', default=False)
    parser.add_argument("--t5-model-path", type=str, default='./pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    args = parser.parse_args()
    main(args)
