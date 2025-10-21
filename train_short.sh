cd autoregressive/train
torchrun --nproc_per_node=8 train_c2i.py --config configs/titok_l.yaml