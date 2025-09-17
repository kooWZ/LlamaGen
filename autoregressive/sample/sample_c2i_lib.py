import torch
import torch.distributed as dist
from tqdm import tqdm
import os
import time
import numpy as np
import math
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
llamagen_path = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(llamagen_path)
from autoregressive.models.generate import generate

flextok_path = os.path.abspath(os.path.join(llamagen_path, "external_tokenizers/flextok"))
sys.path.append(flextok_path)
from flextok.flextok_wrapper import FlexTokFromHub
from flextok.utils.demo import denormalize
from flextok.utils.misc import get_bf16_context, get_generator


def do_sample(gpt_model, args, rank, device, npz_path):
    vq_model = FlexTokFromHub.from_pretrained(args.vq_ckpt)
    vq_model.to(device)
    vq_model.eval()
    dist.barrier()

    n = args.eval_per_gpu_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(
        math.ceil(args.eval_num_fid_samples / global_batch_size) * global_batch_size
    )
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert (
        total_samples % dist.get_world_size() == 0
    ), "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert (
        samples_needed_this_gpu % n == 0
    ), "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    # Use CPU storage and periodic gathering to manage memory efficiently
    batch_samples = []  # Temporary storage for current batch
    batch_labels = []  # Temporary storage for corresponding class indices

    # rank0 maintains all collected samples
    if rank == 0:
        final_samples_list = []
        final_labels_list = []
        print(
            f"Will gather samples every {args.eval_gather_freq} iterations to manage memory efficiently"
        )

    # Store image shape once we know it
    img_shape = None

    for iteration in pbar:
        # Sample inputs:
        c_indices = torch.randint(0, args.num_classes, (n,), device=device)

        index_sample = generate(
            gpt_model,
            c_indices,
            args.latent_size,
            cfg_scale=args.cfg_scale,
            cfg_interval=args.cfg_interval,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_logits=True,
        )
        with get_bf16_context(True):
            reconst = vq_model.detokenize(
                [index_sample[t].unsqueeze(0) for t in range(index_sample.shape[0])],
                timesteps=args.decoder_timesteps,  # Number of denoising steps
                guidance_scale=args.decoder_guidance_scale,  # Classifier-free guidance scale
                perform_norm_guidance=True,  # APG, see https://arxiv.org/abs/2410.02416
                # Optionally control initial noise. Note that while the initial noise is deterministic, the rest of the model isn't.
                generator=get_generator(seed=0, device=device),
                verbose=False,  # Enable to show denoising progress bar with tqdm
            )

        # Convert tensors directly to CPU numpy arrays to save GPU memory
        batch_data = []
        batch_class_indices = []
        for i in range(n):
            # Convert tensor to numpy in the same format as the original PNG->numpy pipeline
            # Immediately move to CPU to save GPU memory
            img_tensor = denormalize(reconst[i].cpu()).clamp(0, 1)
            # Convert from tensor (C, H, W) to numpy (H, W, C) and to uint8
            img_numpy = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            batch_data.append(img_numpy)
            # Store corresponding class index
            batch_class_indices.append(c_indices[i].cpu().numpy())

        # Store in CPU memory and record image shape
        if batch_data and img_shape is None:
            img_shape = batch_data[0].shape
        batch_samples.extend(batch_data)
        batch_labels.extend(batch_class_indices)
        total += global_batch_size

        # Periodic gathering to rank0 to manage memory
        if (iteration + 1) % args.eval_gather_freq == 0 or (
            iteration + 1
        ) == iterations:
            start_gather_time = time.time()

            # Convert current batch to numpy arrays for gathering
            if batch_samples:
                rank_batch_images = np.stack(batch_samples)
                rank_batch_labels = np.array(batch_labels)
                batch_samples = []  # Clear after gathering
                batch_labels = []  # Clear after gathering
            else:
                # Create empty arrays with correct shape
                if img_shape is None:
                    img_shape = (256, 256, 3)  # Default fallback
                rank_batch_images = np.empty((0, *img_shape), dtype=np.uint8)
                rank_batch_labels = np.empty((0,), dtype=np.int64)

            if rank == 0:
                print(f"Gathering samples from iteration {iteration + 1}...")

            # Gather this batch from all ranks (both images and labels)
            gathered_images = [None for _ in range(dist.get_world_size())]
            gathered_labels = [None for _ in range(dist.get_world_size())]
            torch.distributed.all_gather_object(gathered_images, rank_batch_images)
            torch.distributed.all_gather_object(gathered_labels, rank_batch_labels)

            # rank0 processes and stores the gathered data
            if rank == 0:
                for rank_images, rank_labels in zip(gathered_images, gathered_labels):
                    if rank_images.shape[0] > 0:
                        final_samples_list.append(rank_images)
                        final_labels_list.append(rank_labels)

                total_collected = sum(arr.shape[0] for arr in final_samples_list)
                print(
                    f"Rank0: Gather took {time.time() - start_gather_time} seconds. Collected {total_collected} total samples and labels so far"
                )

            # Synchronize all processes
            dist.barrier()

    # Final processing and NPZ creation (data already collected via periodic gathering)
    del vq_model
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        if final_samples_list and final_labels_list:
            print("Concatenating all collected samples and labels...")
            final_samples = np.concatenate(final_samples_list, axis=0)
            final_labels = np.concatenate(final_labels_list, axis=0)

            # Truncate to exact number of samples needed (maintain correspondence)
            final_samples = final_samples[: args.eval_num_fid_samples]
            final_labels = final_labels[: args.eval_num_fid_samples]

            # Save directly to NPZ with both images and labels
            np.savez(npz_path, arr_0=final_samples, labels=final_labels)
            print(f"Saved .npz file to {npz_path}")
            print(f"  Images shape: {final_samples.shape}")
            print(f"  Labels shape: {final_labels.shape}")
            print("Memory efficient generation with labels completed!")
        else:
            print("!!!WARNING: No samples collected!")
