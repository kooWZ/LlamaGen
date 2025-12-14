# extract codes
cd autoregressive/train
WANDB_MODE=offline torchrun --nproc_per_node=8 extract_codes_llamagen.py \
    --ckpt-path /inspire/hdd/project/autoregressive-video-generation/pengwujian-240108120095/workspace/zijie/projects/postTok/weights/LlamaGen/vq_ds16_c2i.pt \
    --bsz 32


# convert codes
cd ..
cd .. #at LLamaGen
cd tools
python convert_npy_to_hdf5.py \
    --code_dir dataset/ImageNet-1k/llamagen_codes_384/imagenet384_codes \
    --label_dir dataset/ImageNet-1k/llamagen_codes_384/imagenet384_labels \
    --output_prefix dataset/ImageNet-1k/llamagen_codes_384/h5_dataset/dataset_rank \
    --code-len 576

cd ..
# train
cd autoregressive/train
WANDB_MODE=offline torchrun --nproc_per_node=8 train_c2i.py --config configs/llamagen.yaml