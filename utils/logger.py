import logging
import torch.distributed as dist
import base64
import os
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        handlers = [logging.StreamHandler()]
        if os.path.exists(logging_dir):
            handlers.append(logging.FileHandler(f"{logging_dir}/log.txt"))
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=handlers
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def setup_wandb(args, experiment_dir, logger, rank, extra_config={}):
    """Initialize wandb logging if available and requested."""
    if not WANDB_AVAILABLE:
        logger.info("wandb not available, skipping wandb setup")
        return False

    if args.wandb_project is None or rank != 0:
        return False

    # Create wandb config from args
    wandb_config = vars(args)
    wandb_config.update(extra_config)

    # Initialize wandb
    run_name = f"{args.gpt_model}_{args.gpt_type}_{args.latent_size}"
    if args.wandb_run_name:
        run_name = args.wandb_run_name

    wandb.login(
        key=base64.b64decode(
            "ZmU2N2E1NjJkOGM5NjhjMjE1ZmU3Zjc1NDM2Zjc4YzljYTVkZWVjNg=="
        ).decode("utf-8")
    )
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=wandb_config,
        dir=experiment_dir,
        entity=args.wandb_entity,
        tags=(
            args.wandb_tags if hasattr(args, "wandb_tags") and args.wandb_tags else None
        ),
        notes=(
            args.wandb_notes
            if hasattr(args, "wandb_notes") and args.wandb_notes
            else None
        ),
    )

    logger.info(f"wandb initialized: project={args.wandb_project}, run_name={run_name}")
    return True
