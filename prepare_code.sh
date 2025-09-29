cd autoregressive/train
torchrun --nproc_per_node=8 extract_codes_titok.py --data-path "$IMGNET_PATH"

cd ../../tools
python convert_npy_to_hdf5.py \
    --code_dir dataset/ImageNet-1k/titok_codes/imagenet256_codes \
    --label_dir dataset/ImageNet-1k/titok_codes/imagenet256_labels \
    --output_prefix dataset/ImageNet-1k/titok_codes/h5_dataset/dataset_rank \
    --code-len 128
