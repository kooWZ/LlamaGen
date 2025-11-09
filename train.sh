# cd autoregressive/train
# torchrun --nproc_per_node=8 extract_codes_ours.py --data-path "$IMGNET_PATH"

# cd ../../

# cd tools
# python convert_npy_to_hdf5.py \
#     --code_dir dataset/ImageNet-1k/ours_0250000_codes/imagenet256_codes \
#     --label_dir dataset/ImageNet-1k/ours_0250000_codes/imagenet256_labels \
#     --output_prefix dataset/ImageNet-1k/ours_0250000_codes/h5_dataset/dataset_rank \
#     --code-len 128

# cd ..


cd autoregressive/train
torchrun --nproc_per_node=8 train_c2i.py --config configs/ours_0250000_l.yaml