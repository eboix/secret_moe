# Organization of directory

`sae_demo` contains code to train sparse autoencoders, adapted from https://github.com/adamkarvonen/dictionary_learning_demo

* python train_mlp_sae.py --save_dir ./saes --model_name EleutherAI/pythia-70m-deduped --io in --layers 3 --architectures top_k --use_wandb --num_tokens 50000000
* python train_mlp_sae.py --save_dir ./saes --model_name EleutherAI/pythia-70m-deduped --io out --layers 3 --architectures top_k --use_wandb --num_tokens 50000000

<!-- * python train_mlp_sae.py --save_dir ./saes --model_name EleutherAI/pythia-2.8b-deduped --io in --layers 16 --archite
ctures top_k --use_wandb --num_tokens 50000000 -->

`my_distilling_code` contains code to distill MLP layers to other kinds of networks -- this was written by me

