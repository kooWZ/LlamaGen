# torchrun --nproc_per_node=8 do_eval.py --latent-size 32
# torchrun --nproc_per_node=8 do_eval.py
torchrun --nproc_per_node=8 do_eval.py --gpt-ckpt /root/kongly/AR/LlamaGen/outputs/test/023-GPT-Mini/checkpoints/60_0763171.pt
torchrun --nproc_per_node=8 do_eval.py --gpt-ckpt /root/kongly/AR/LlamaGen/outputs/test/023-GPT-Mini/checkpoints/60_0763171.pt --latent-size 32
torchrun --nproc_per_node=8 do_eval.py --gpt-ckpt /root/kongly/AR/LlamaGen/outputs/test/023-GPT-Mini/checkpoints/30_0387841.pt
torchrun --nproc_per_node=8 do_eval.py --gpt-ckpt /root/kongly/AR/LlamaGen/outputs/test/023-GPT-Mini/checkpoints/30_0387841.pt --latent-size 32
python ~/kongly/end.py --vimd kongly2-0