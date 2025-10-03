cd autoregressive/train

torchrun --nproc_per_node=8 eval.py --config configs/evals/eval_flextok_cfg1_32.yaml &>> eval.log
torchrun --nproc_per_node=8 eval.py --config configs/evals/eval_flextok_cfg175_32.yaml &>> eval.log
torchrun --nproc_per_node=8 eval.py --config configs/evals/eval_flextok_cfg2_32.yaml &>> eval.log
torchrun --nproc_per_node=8 eval.py --config configs/evals/eval_titok_cfg1.yaml &>> eval.log
torchrun --nproc_per_node=8 eval.py --config configs/evals/eval_titok_cfg175.yaml &>> eval.log
torchrun --nproc_per_node=8 eval.py --config configs/evals/eval_titok_cfg2.yaml &>> eval.log