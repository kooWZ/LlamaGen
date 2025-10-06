cd autoregressive/train

torchrun --nproc_per_node=8 eval.py --config configs/evals/eval_titok_cfg2.yaml --override cfg_scale=2.25 &> eval.log
torchrun --nproc_per_node=8 eval.py --config configs/evals/eval_titok_cfg2.yaml --override cfg_scale=2.5 &>> eval.log
torchrun --nproc_per_node=8 eval.py --config configs/evals/eval_titok_cfg2.yaml --override cfg_scale=2.75 &>> eval.log
torchrun --nproc_per_node=8 eval.py --config configs/evals/eval_titok_cfg2.yaml --override cfg_scale=3 &>> eval.log
torchrun --nproc_per_node=8 eval.py --config configs/evals/eval_titok_cfg2.yaml --override cfg_scale=3.5 &>> eval.log
torchrun --nproc_per_node=8 eval.py --config configs/evals/eval_titok_cfg2.yaml --override cfg_scale=4 &>> eval.log
torchrun --nproc_per_node=8 eval.py --config configs/evals/eval_titok_cfg2.yaml --override cfg_scale=5 &>> eval.log
torchrun --nproc_per_node=8 eval.py --config configs/evals/eval_titok_cfg2.yaml --override cfg_scale=6 &>> eval.log