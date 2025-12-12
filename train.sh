# extract codes
cd autoregressive/train
WANDB_MODE=offline torchrun --nproc_per_node=8 extract_codes_llamagen.py \
    --ckpt-path xxx/vq_ds16_c2i.pt \
    --bsz 1


# convert codes
cd convert
python convert_npy_to_hdf5.py \
    --code_dir dataset/ImageNet-1k/llamagen_codes_384/imagenet384_codes \
    --label_dir dataset/ImageNet-1k/llamagen_codes_384/imagenet384_labels \
    --output_prefix dataset/ImageNet-1k/llamagen_codes_384/h5_dataset/dataset_rank \
    --code-len 576


# train
cd autoregressive/train
WANDB_MODE=offline torchrun --nproc_per_node=8 train_c2i.py --config configs/llamagen.yaml