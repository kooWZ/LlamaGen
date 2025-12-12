import os
import json
import numpy as np
import h5py
from tqdm import tqdm
from multiprocessing import Process
import argparse


current_dir = os.path.dirname(os.path.abspath(__file__))
llamagen_path = os.path.abspath(os.path.join(current_dir, "../"))

def convert_path(path, force=False):
    if force:
        return os.path.join(llamagen_path, path)
    if os.path.exists(path) or os.path.isabs(path):
        return path
    else:
        return os.path.join(llamagen_path, path)


def process_rank(rank, files, args):
    """处理单个 rank，输出一个 HDF5 shard"""
    print(f"[Rank {rank}] 开始处理 {len(files)} 个文件")

    # 统计展开后的样本数
    total_samples = 0
    for fname in files:
        arr = np.load(os.path.join(args.code_dir, fname), mmap_mode="r")
        total_samples += arr.shape[0] * arr.shape[1]  # 展开后数量 = 500 * 10

    print(f"[Rank {rank}] 总样本数（展开后）: {total_samples}")
    min_val = 999999999
    max_val = -999999999
    # 创建 HDF5 数据集
    output_h5 = f"{args.output_prefix}{rank}.h5"
    with h5py.File(output_h5, "w") as f:
        code_dset = f.create_dataset(
            "code",
            shape=(total_samples, args.code_len),  # 展开成 [N, 256]
            dtype=np.uint16,
            compression="lzf",
        )
        label_dset = f.create_dataset("label", shape=(total_samples,), dtype=np.uint16)
        path_dset = f.create_dataset(
            "path", shape=(total_samples,), dtype=h5py.string_dtype(encoding="utf-8")
        )

        idx = 0
        for code_fname in tqdm(files, desc=f"Rank {rank}"):
            label_fname = os.path.splitext(code_fname)[0] + ".json"

            code_path = os.path.join(args.code_dir, code_fname)
            label_path = os.path.join(args.label_dir, label_fname)

            code_arr = np.load(code_path).astype(np.uint16)  # [500, 10, 256]
            min_val = min(min_val, code_arr.min())
            max_val = max(max_val, code_arr.max())
            with open(label_path, "r", encoding="utf-8") as jf:
                label_info = json.load(jf)  # len = 500

            assert code_arr.shape[0] == len(
                label_info
            ), f"{code_fname} 与 {label_fname} 样本数不匹配"

            # 展开 augmentation
            n_imgs = code_arr.shape[0]
            n_aug = code_arr.shape[1]
            code_arr_flat = code_arr.reshape(n_imgs * n_aug, args.code_len)  # [5000, 256]

            # 复制 label 和 path
            labels_flat = np.repeat([item["label"] for item in label_info], n_aug)
            paths_flat = np.repeat([item["path"] for item in label_info], n_aug)

            # 写入
            n = code_arr_flat.shape[0]
            code_dset[idx : idx + n] = code_arr_flat
            label_dset[idx : idx + n] = labels_flat
            path_dset[idx : idx + n] = paths_flat
            idx += n

    print(f"[Rank {rank}] 完成写入 {output_h5}")
    print(f"[Rank {rank}] Code 值范围: [{min_val}, {max_val}]")

if __name__ == "__main__":
    code_dir = "/root/kongly/AR/LlamaGen/dataset/ImageNet-1k/flextok_codes/random_crop/imagenet256_codes"
    label_dir = "/root/kongly/AR/LlamaGen/dataset/ImageNet-1k/flextok_codes/random_crop/imagenet256_labels"
    output_prefix = "/root/kongly/AR/LlamaGen/dataset/ImageNet-1k/flextok_codes/random_crop/h5_dataset/dataset_rank"  # 文件名前缀

    parser = argparse.ArgumentParser()
    parser.add_argument("--code_dir", type=str, default=code_dir)
    parser.add_argument("--label_dir", type=str, default=label_dir)
    parser.add_argument("--output_prefix", type=str, default=output_prefix)
    parser.add_argument("--code-len", type=int, default=256)
    args = parser.parse_args()

    args.code_dir = convert_path(args.code_dir)
    args.label_dir = convert_path(args.label_dir)
    args.output_prefix = convert_path(args.output_prefix)
    assert os.path.exists(args.code_dir), f"code dir {args.code_dir} does not exist!"
    assert os.path.exists(args.label_dir), f"label dir {args.label_dir} does not exist!"
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    # 获取所有 .npy 文件
    code_files = sorted([f for f in os.listdir(args.code_dir) if f.endswith(".npy")])

    # 按 rank 分组文件
    rank_files = {r: [] for r in range(8)}
    for fname in code_files:
        rank = int(fname.split("_")[0])
        rank_files[rank].append(fname)

    processes = []
    for rank, files in rank_files.items():
        p = Process(target=process_rank, args=(rank, files, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("所有 shard 转换完成 ✅")
