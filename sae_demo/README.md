Demonstration of how to train, evaluate, and plot evaluation results using the `dictionary_learning` repo. We use `dictionary_learning` as a git submodule.

# Setup

For ease of development, we use SSH for our git submodule url. If you don't have an ssh key setup, run `git config submodule.dictionary_learning.url https://github.com/saprmarks/dictionary_learning.git` before updating the submodule.

```
git submodule update --init --recursive
pip install -e .
```
If using wandb logging, run `wandb login {my token}`. If downloading a gated model (like Gemma-2-2B) or uploading SAEs to huggingface, run `huggingface-cli login --token {my token}`.

# Usage

In `demo_config.py`, we set various hyperparameters like number of training tokens and expansion factors. We also set various SAE specific hyperparameters, such as the bandwidth for JumpReLU.

The bottom of `demo.py` contains a variety of example commands for training a variety of SAEs on different models. As an example, the following command will traing 24 different SAEs (4 sparsities per architecture) on 50 million tokens on Pythia-70M in ~1 hour on a RTX 3090. It will also evaluate all SAEs on metrics like L0 and loss recovered.

`python demo.py --save_dir ./saes --model_name EleutherAI/pythia-70m-deduped --layers 3 --architectures standard jump_relu batch_top_k top_k gated`

As an example command to train on dataset with chat / pretrain mix, log with wandb, and upload to huggingface at the end, run the following. This easily fits on to an 80GB H100 (peak memory usage of 62 GB) and trains on 500M tokens in ~24 hours.

`python demo.py --save_dir ./saes --model_name Qwen/Qwen2.5-Coder-32B-Instruct --layers 32 --architectures batch_top_k --use_wandb --hf_repo_id adamkarvonen/qwen_coder_32b_saes --mixed_dataset`

We currently support the following SAEs:

- ReLU (Towards Monosemanticity, aka `standard`)
- ReLU (Anthropic April Update, aka `standard_new`)
- JumpReLU
- TopK
- BatchTopK
- Gated
- P-anneal ReLU
- Matryoshka BatchTopK

Confused about so many variants? BatchTopK is a reasonable default.

NOTE: For TopK and BatchTopK, we record the average minimum activation value during training, enabling the use of the SAEs using a simple JumpReLU threshold during inference. We use this as the default approach during inference, as it tends to get a better loss recovered, eliminates interaction between features, and enables encoding only a subset of latents.

We can then graph the results using `graphing.ipynb` if we specify the names of the output folders.

There's also various command line arguments available. Notable ones include `hf_repo_id` to automatically push trained SAEs to HuggingFace after training and `save_checkpoints` to save checkpoints during training.

# How does this differ from dictionary_learning?

The default settings are often pretty conservative in `dictionary_learning`. For example, the autocast dtype defaults to float32. We use bfloat16, which has been fine on all SAE variants tested and provides a ~1.5x speedup. However, if experimenting with new variants, you may want to initially test with float32 - this can bite you with underflow issues! We also use HuggingFace transformers instead of nnsight LanguageModels, which lets us easily truncate the model for significant memory savings. We also normalize activations and backup every 1000 steps by default.

# SAE Bench Replication

SAE Bench provides a suite of baseline SAEs. You can deterministically reproduce the training of SAE Bench SAEs using this repository.

As an example of how to reproduce the Gemma-2-2B 16K width TopK SAEs, set [num_tokens](https://github.com/adamkarvonen/dictionary_learning_demo/blob/main/demo_config.py#L54) to 500M and run the following command:

`python demo.py --save_dir topk --model_name google/gemma-2-2b --layers 12 --architectures top_k`

The SAE Bench suite was trained with the following widths: 2^12, 2^14, and 2^16. The width is set [here](https://github.com/adamkarvonen/dictionary_learning_demo/blob/main/demo_config.py#L57).