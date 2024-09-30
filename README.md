## ComboStoc: Combinatorial Stochasticity for Diffusion Generative Models<br><sub>Official Code</sub>



![ComboStoc samples](visuals/teaser.png)

This repo contains PyTorch model definitions and training/sampling code for our paper exploring 
the Combinatorial Stochasticity for Diffusion Generative Models.



In this paper, we study an under-explored but important factor of diffusion generative models, i.e., the combinatorial complexity. 
Data samples are generally high-dimensional, and for various structured generation tasks, additional attributes are combined to associate with data samples.
We show that the space spanned by the combination of dimensions and attributes is insufficiently sampled by existing training scheme of diffusion generative models, causing degraded test time performance.
We present a simple fix to this problem by constructing stochastic processes that fully exploit the combinatorial structures, hence the name ComboStoc.
Using this simple strategy, we show that network training is significantly accelerated across diverse data modalities, including images and 3D structured shapes.
Moreover, ComboStoc enables a new way of test time generation which uses asynchronous time steps for different dimensions and attributes, thus allowing for varying degrees of control over them.


It is worth noting that our project utilized the SiT code framework, with key modifications made to the 'sample' and 'compensate_offdiagonal_ut' functions within the 'Transport' class in the 'transport.py' file. For varying combination complexities, one can choose from the options provided by the --combostoc-type flag during the training phase. The choices available are: [UNSYNC_NONE, UNSYNC_VEC, UNSYNC_PATCH, UNSYNC_ALL]. 
We extend our gratitude to the SiT team for their exceptional contribution and for making their work available as open source.




This repository contains:

* 🪐 A simple PyTorch [implementation](models.py) of ComboStoc following SiT
<!-- * ⚡️ Pre-trained class-conditional ComboStoc models trained on ImageNet 256x256 -->
* 🛸 A ComboStoc [training script](train.py) using PyTorch DDP

## Setup



We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate ComboStoc
```


## Sampling 
![More ComboStoc samples](visuals/yingwu.drawio.png)

You can sample from ComboStoc checkpoints with [`sample.py`](sample.py). The script has various arguments to adjust sampler configurations (ODE & SDE), sampling steps, change the classifier-free guidance scale, etc. For example, to sample from
our 256x256 ComboStoc-XL model with default ODE setting, you can use:

```bash
python sample.py SDE --model ComboStoc-XL/2 --image-size 256 --ckpt /path/to/model.pt
```

### Advanced sampler settings (Following SiT)

|     |          |          |                         |
|-----|----------|----------|--------------------------|
| ODE | `--atol` | `float` |  Absolute error tolerance |
|     | `--rtol` | `float` | Relative error tolenrace |   
|     | `--sampling-method` | `str` | Sampling methods (refer to [`torchdiffeq`] ) |

|     |          |          |                         |
|-----|----------|----------|--------------------------|
| SDE | `--diffusion-form` | `str` | Form of SDE's diffusion coefficient (refer to Tab. 2 in [paper]()) |
|     | `--diffusion-norm` | `float` | Magnitude of SDE's diffusion coefficient |
|     | `--last-step` | `str` | Form of SDE's last step |
|     |               |       | None - Single SDE integration step |
|     |               |       | "Mean" - SDE integration step without diffusion coefficient |
|     |               |       | "Tweedie" - [Tweedie's denoising] step | 
|     |               |       | "Euler" - Single ODE integration step
|     | `--sampling-method` | `str` | Sampling methods |
|     |               |       | "Euler" - First order integration | 
|     |               |       | "Heun" - Second order integration | 

There are some more options; refer to [`train_utils.py`](train_utils.py) for details.

## Training ComboStoc

We provide a training script for ComboStoc in [`train.py`](train.py). To launch ComboStoc-XL/2 (256x256) training with `N` GPUs on 
one node:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model ComboStoc-XL/2 --data-path /path/to/imagenet/train --combostoc-type UNSYNC_ALL
```

**Logging.** To enable `wandb`, firstly set `WANDB_KEY`, `ENTITY`, and `PROJECT` as environment variables:

```bash
export WANDB_KEY="key"
export ENTITY="entity name"
export PROJECT="project name"
```

Then in training command add the `--wandb` flag:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model ComboStoc-XL/2 --data-path /path/to/imagenet/train --combostoc-type UNSYNC_ALL --wandb 
```

**Interpolant settings.** We also support different choices of interpolant and model predictions. For example, to launch ComboStoc-XL/2 (256x256) with `Linear` interpolant and `noise` prediction: 

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model ComboStoc-XL/2 --data-path /path/to/imagenet/train --combostoc-type UNSYNC_ALL --path-type Linear --prediction noise
```

**Resume training.** To resume training from custom checkpoint:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model ComboStoc-L/2 --data-path /path/to/imagenet/train --combostoc-type UNSYNC_ALL --ckpt /path/to/model.pt
```

**Caution.** Resuming training will automatically restore both model, EMA, and optimizer states and training configs to be the same as in the checkpoint.

## Evaluation (FID, Inception Score, etc.)

We include a [`sample_ddp.py`](sample_ddp.py) script which samples a large number of images from a ComboStoc model in parallel. This script 
generates a folder of samples as well as a `.npz` file which can be directly used with [PyTorch-FID] to compute FID, Inception Score and
other metrics. For example, to sample 50K images from our pre-trained ComboStoc-XL/2 model over `N` GPUs under default ODE sampler settings, run:

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py SDE --model ComboStoc-XL/2 --num-fid-samples 50000
```





