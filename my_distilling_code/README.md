## Code accompanying the paper _Your LLM is secretly a mixture of experts_

See `moe_test.ipynb` for the experiments


### Step 1: generate and save activations

Saves activations from pythia 410m model into temporary, fast storage.

```
python save_activations.py \
    --model_name "EleutherAI/pythia-410m" \
    --dataset_name "wikitext" \
    --dataset_config_name "wikitext-2-raw-v1" \
    --layer_idx 12 \
    --activation_type "input" \
    --output_dir "/work/nvme/bbjr/eboix/saved_activations" \
    --device "cuda" \
    --dtype "float16" \
    --activation_dl_batch_size 1024 \
    --text_dl_batch_size 8 \
    --max_length 512 \
    --num_val_act_batches 200 \
    --max_val_text_samples 2000
```

### Step 2: train student models on the teacher model
```

```