
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
# import socket

from models import ComboStoc_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
import wandb_utils
from torchvision.utils import save_image



@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])



def main(args):
    """
    Trains a new ComboStoc model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())


    
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True) 
        
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-") 
    experiment_name = f"{model_string_name}-{args.exp}" 

    experiment_dir = f"{args.results_dir}/{experiment_name}"   
    
    

    if rank == 0:


        checkpoint_dir = f"{experiment_dir}/checkpoints" 
        os.makedirs(checkpoint_dir, exist_ok=True)
        sample_dir = f"{experiment_dir}/samples"
        os.makedirs(sample_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        

        os.makedirs(f"{experiment_dir}/code", exist_ok=True)
        os.system(f"cp -r ./transport {experiment_dir}/code")
        os.system(f"cp -r ./train.py {experiment_dir}/code")
        os.system(f"cp -r ./sample.py {experiment_dir}/code")
        os.system(f"cp -r ./sample_ddp.py {experiment_dir}/code")
        os.system(f"cp -r ./train_utils.py {experiment_dir}/code")
        os.system(f"cp -r ./models.py {experiment_dir}/code")
        os.system(f"cp -r ./wandb_utils.py {experiment_dir}/code")
        

        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = ComboStoc_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    step_continue=0
    ckpt_state_dict = None
    autoRetrain = True  # if True, will load the last checkpoint to continue training
    if autoRetrain:
        ckpt_path = f'{experiment_dir}/ckpt.pt'
        # check exist
        if os.path.isfile(ckpt_path):
            ckpt_state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt_state_dict["model"])
            ema.load_state_dict(ckpt_state_dict["ema"])
            # if ckpt_state_dict.get("step") is not None:
            step_continue = ckpt_state_dict["step"]
            print("Loading ckpt to continue...")
        else:
            print(f"Could not find ComboStoc Continue checkpoint at {ckpt_path}")



    
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        # state_dict = find_model(ckpt_path)
        assert os.path.isfile(ckpt_path), f'Could not find ComboStoc checkpoint at {ckpt_path}'
        ckpt_state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt_state_dict["model"])
        ema.load_state_dict(ckpt_state_dict["ema"])
        if ckpt_state_dict.get("step") is not None:
            step_continue = ckpt_state_dict["step"]

    

    requires_grad(ema, False)
    
    model = DDP(model.to(device), device_ids=[rank])
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        combostoc_type = args.combostoc_type
    )  # default: velocity; 
    logger.info(f"Transporting using ComboStoc mode: {args.combostoc_type}.")
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"ComboStoc Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    
    if args.ckpt is not None:
        opt.load_state_dict(ckpt_state_dict["opt"])

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode


    train_steps = step_continue
    log_steps = 0
    running_loss = 0
    start_time = time()

    ys = torch.randint(1000, size=(8,), device=device)
    use_cfg = args.cfg_scale > 1.0
    # Create sampling noise:
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            model_kwargs = dict(y=y)
            
            debug_train = False
            if debug_train:
                # for t = 0.0 to 1.0 with 10 times
                for t in range(11):
                    x1_pred, x1 = transport.training_eval_debug(model, x, model_kwargs,t/10)

                
                    samples = vae.decode(x1_pred / 0.18215).sample
                    out_samples = torch.zeros((8*dist.get_world_size(), 3, args.image_size, args.image_size), device=device)
                    dist.all_gather_into_tensor(out_samples, samples)
                    save_image(out_samples, f"{experiment_dir}/samples/sample_pred_{t:07d}.png", nrow=4, normalize=True, value_range=(-1, 1))

                    samples = vae.decode(x1/ 0.18215).sample
                    out_samples = torch.zeros((8*dist.get_world_size(), 3, args.image_size, args.image_size), device=device)
                    dist.all_gather_into_tensor(out_samples, samples)
                    save_image(out_samples, f"{experiment_dir}/samples/sample_GT_{t:07d}.png", nrow=4, normalize=True, value_range=(-1, 1))
            
                continue
            
            
            
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        { "train loss": avg_loss, "train steps/sec": steps_per_sec },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "step": train_steps,
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            if train_steps % args.reckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "step": train_steps,
                        "args": args
                    }
                    checkpoint_path = f'{experiment_dir}/ckpt.pt'
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            
            if train_steps % args.sample_every == 1 and train_steps > 0:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    sample_fn = transport_sampler.sample_ode() # default to ode sampling
                    samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                    dist.barrier()

                    if use_cfg: #remove null samples
                        samples, _ = samples.chunk(2, dim=0)
                    samples = vae.decode(samples / 0.18215).sample
                    out_samples = torch.zeros((8*dist.get_world_size(), 3, args.image_size, args.image_size), device=device)
                    dist.all_gather_into_tensor(out_samples, samples)
                    if args.wandb:
                        wandb_utils.log_image(out_samples, train_steps)
                    else:
                        # Save images
                        if rank == 0:
                            save_image(out_samples, f"{experiment_dir}/samples/sample_{train_steps:07d}.png", nrow=4, normalize=True, value_range=(-1, 1))
                    logging.info("Generating EMA samples done.")

    model.eval()

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="imagenet/train") # Path to ImageNet training data
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(ComboStoc_models.keys()), default="ComboStoc-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--exp", type=str, default="exp_tmp")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--reckpt-every", type=int, default=5000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom ComboStoc checkpoint")
    parser.add_argument("--combostoc-type", type=str, default="UNSYNC_ALL", 
                        help="ComboStoch type, one of [UNSYNC_NONE, UNSYNC_VEC, UNSYNC_PATCH, UNSYNC_ALL]") # select the type of ComboStoc

    parse_transport_args(parser)
    args = parser.parse_args()
    

    main(args)

