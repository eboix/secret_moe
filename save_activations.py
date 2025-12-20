# save_activations.py
import torch
import os
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import re # For sanitizing model name for filename
import warnings
from activation_loader import ActivationDataLoader
import numpy as np


def sanitize_filename(name):
    """Sanitizes a string to be used as a filename."""
    if name is None: return "None"
    name = str(name).replace('/', '_') 
    name = re.sub(r'[^\w_.-]', '', name) 
    return name if name else "sanitized_name"


def save_activations_for_split(model, tokenizer, random_init_from_seed, layer_idx, dataset_name, dataset_config, split_name,
                               dataset_obj, # This is the actual dataset[split_name] object
                               activation_dl_batch_size, text_dl_batch_size,
                               max_length, dtype, device, output_dir, 
                               num_act_batches_to_save=None, activation_type='input',
                               max_text_samples=None, min_token_length=20,
                               activs_per_file=1000000): # MODIFIED: Added min_token_length
    """
    Saves activations for a given dataset split, with token length filtering.
    """
    print(f"\nProcessing split: {split_name} from {dataset_name}/{dataset_config if dataset_config else 'default'}")
    current_split_data = dataset_obj
    original_num_rows = len(current_split_data)

    # --- MODIFIED SECTION: Token Length Filtering ---
    if min_token_length > 0:
        print(f"Filtering dataset split '{split_name}' to keep lines with at least {min_token_length} tokens...")

        def token_length_filter_fn(examples):
            # examples is a dict like {'text': ['line1', 'line2', ...]}
            # Ensure all elements in examples['text'] are strings for tokenizer
            texts_for_tokenizer = [str(t) if t is not None else "" for t in examples['text']]
            
            # Tokenize without truncation to get actual token length
            tokenized_texts = tokenizer(texts_for_tokenizer, truncation=False, padding=False)
            return [len(ids) >= min_token_length for ids in tokenized_texts['input_ids']]

        filtered_dataset = current_split_data.filter(
            token_length_filter_fn,
            batched=True,
            batch_size=1000  # Process 1000 examples at a time for filtering efficiency
        )
        new_num_rows = len(filtered_dataset)
        print(f"Original number of lines in '{split_name}': {original_num_rows}")
        print(f"Number of lines after filtering (>= {min_token_length} tokens): {new_num_rows} (removed {original_num_rows - new_num_rows} lines)")
        
        if new_num_rows == 0:
            print(f"No lines remaining in '{split_name}' after filtering. Skipping this split.")
            return
        current_split_data = filtered_dataset
    else:
        print(f"Skipping token length filtering as min_token_length is {min_token_length} or less.")
    # --- END OF MODIFIED SECTION ---

    # Optionally limit text samples from the (potentially) filtered dataset
    if max_text_samples is not None and len(current_split_data) > max_text_samples:
        print(f"Limiting text samples for '{split_name}' split to {max_text_samples} (from processed data).")
        current_split_data = current_split_data.select(range(max_text_samples))

    if len(current_split_data) == 0:
        print(f"No data remaining in '{split_name}' after all processing steps. Skipping.")
        return

    text_dataloader = DataLoader(
        current_split_data, 
        batch_size=text_dl_batch_size, 
        shuffle=True, 
        collate_fn=lambda batch_list: [item['text'] for item in batch_list if item.get('text') and item['text'].strip()]
    )

    # Remainder of the function (ActivationDataLoader creation, saving logic) is the same as before...
    activation_loader = ActivationDataLoader(
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        text_dataloader=text_dataloader,
        max_length=max_length, # This max_length is for the ActivationDataLoader processing
        batch_size=activation_dl_batch_size, 
        activation_type=activation_type,
        max_buffer_size=1000000, 
        dtype=dtype,
        device=device
    )

    all_activations_list = []
    
    s_dataset = sanitize_filename(dataset_name)
    s_config = sanitize_filename(dataset_config if dataset_config else "default_config")
    s_model = sanitize_filename(model.config._name_or_path) 
    
    modelinitstr = f"_randominit{random_init_from_seed}" if random_init_from_seed is not None else ""

    output_sub_directory = f"{s_dataset}_{s_config}_{s_model}{modelinitstr}_layer{layer_idx}_act{activation_type}_{split_name}_mintok{min_token_length}/"
    output_path = os.path.join(output_dir, output_sub_directory)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    if os.path.exists(os.path.join(output_path, '.completed')):
        print('COMPLETED... SKIPPING')
        return
    elif os.path.exists(os.path.join(output_path, '.lock')):
        print('LOCKED... SKIPPING')
        return

    # create lock
    os.makedirs(os.path.join(output_path, '.lock'))

    print(f"Collecting activations from layer {layer_idx} ({activation_type}). Target directory: {output_path}")
    
    progress_bar = tqdm(unit=" act_batch")
    

    file_number = 0
    act_batches_collected_count = 0
    try:
        for act_batch_tensor in activation_loader: 
            if act_batch_tensor is not None and act_batch_tensor.numel() > 0:
                all_activations_list.append(act_batch_tensor.clone().cpu()) 
                act_batches_collected_count += 1
                progress_bar.update(1)
                # progress_bar.set_postfix(collected=f"{act_batches_collected_count * activation_dl_batch_size} vecs")

            if len(all_activations_list) >= activs_per_file:
                final_activations_tensor = torch.cat(all_activations_list, dim=0)
                print(f"Collected activations shape for {split_name}: {final_activations_tensor.shape}")
                output_filename = os.path.join(output_path, f"activations_{file_number}.pt")
                torch.save(final_activations_tensor.cpu(), output_filename)
                print(f"Saved {len(all_activations_list)} activation batches to {output_filename}")
                file_number += 1
                all_activations_list.clear()

            if num_act_batches_to_save is not None and act_batches_collected_count >= num_act_batches_to_save:
                print(f"\nReached specified {num_act_batches_to_save} activation batches for saving.")
                break
    except Exception as e:
        warnings.warn(f"Error during activation collection for {split_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        progress_bar.close()
        if 'activation_loader' in locals() and hasattr(activation_loader, '_cleanup_hook'):
            activation_loader._cleanup_hook() 
        del activation_loader 
        del text_dataloader

    if all_activations_list:
        final_activations_tensor = torch.cat(all_activations_list, dim=0)
        print(f"Collected activations shape for {split_name}: {final_activations_tensor.shape}")
        output_filename = os.path.join(output_path, f"activations_{file_number}.pt")
        torch.save(final_activations_tensor.cpu(), output_filename)
        print(f"Successfully saved {split_name} activations to {output_filename}")
    else:
        print('No excess activations collected.')

    del all_activations_list 
    if torch.cuda.is_available() and device == 'cuda':
        torch.cuda.empty_cache()

    # Create completion signal
    os.makedirs(os.path.join(output_path, '.completed'), exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Save model activations from a dataset.")
    # ... (all previous arguments are the same) ...
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m", help="Hugging Face model name.")
    parser.add_argument('--model_revision', type=str, default=None, help='Model revision (branch, tag, commit).')
    parser.add_argument('--random_init_from_seed', type=int, default=None,
                        help='If set, will randomly initialize the model from this seed instead of loading pretrained weights')
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Hugging Face dataset name.")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1", help="Configuration for the dataset. Use 'None' if no specific config.")
    parser.add_argument("--layer_idx", type=int, required=True, help="Layer index for activation extraction.")
    parser.add_argument("--activation_type", type=str, default="input", choices=['input', 'output'], help="Type of activation.")
    parser.add_argument("--output_dir", type=str, default="/nobackups/eboix/activation_data", help="Directory to save activation files.")
    parser.add_argument("--activation_dl_batch_size", type=int, default=1024, help="Batch size for ActivationDataLoader.")
    parser.add_argument("--text_dl_batch_size", type=int, default=8, help="Batch size for text DataLoader.")
    parser.add_argument("--max_length", type=int, default=None, help="Max sequence length for tokenization during activation generation.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Datatype for model and activations.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu'). Autodetects.")
    parser.add_argument("--num_train_act_batches", type=int, default=None, help="Number of TRAIN activation batches to save. All if None.")
    parser.add_argument("--num_val_act_batches", type=int, default=100, help="Number of VALIDATION activation batches to save. All if None.")
    parser.add_argument("--max_train_text_samples", type=int, default=None, help="Max number of text samples from TRAIN split to process. All if None.")
    parser.add_argument("--max_val_text_samples", type=int, default=1000, help="Max number of text samples from VALIDATION split to process. All if None.")
    parser.add_argument("--no_train", action="store_true", help="Skip saving training activations.")
    parser.add_argument("--no_val", action="store_true", help="Skip saving validation activations.")
    parser.add_argument("--val_split_name", type=str, default="validation", help="Name of the validation split.")
    parser.add_argument("--train_split_name", type=str, default="train", help="Name of the training split.")
    parser.add_argument("--activs_per_file", type=int, default=1000, help="Number of activation batches per file. Default is 1000.")

    # MODIFIED: Added new argument
    parser.add_argument("--min_token_length", type=int, default=20,
                        help="Minimum token length for a line to be processed. Lines shorter than this will be filtered out. Set to 0 to disable.")

    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch_dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = torch_dtype_map.get(args.dtype)
    if torch_dtype is None: raise ValueError(f"Unsupported dtype: {args.dtype}")
    print(f"Using dtype: {torch_dtype}")

    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}")
    
    print(f"Loading model {args.model_name}...")
    model_kwargs = {'trust_remote_code': True}
    if torch_dtype in [torch.float16, torch.bfloat16]: model_kwargs['torch_dtype'] = torch_dtype
    
    if args.model_revision is not None:
        print('Loading model revision', args.model_revision)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, revision=args.model_revision, **model_kwargs).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs).to(device)
        print(model)

    if args.random_init_from_seed is not None:
        print(f"Randomly initializing model weights from seed {args.random_init_from_seed}...")
        torch.manual_seed(args.random_init_from_seed)
        model = AutoModelForCausalLM.from_config(model.config).to(device)
        # print(model.gpt_neox.layers[args.layer_idx].mlp.dense_h_to_4h.weight)
    model.eval()
    print(f"Loaded {args.model_name} ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params) to {device} with dtype {torch_dtype}.")

    dataset_config_arg = args.dataset_config_name if args.dataset_config_name != 'None' else None
    print(f"Loading dataset: {args.dataset_name}, Config: {dataset_config_arg if dataset_config_arg else 'Default'}")
    try:
        full_dataset_dict = load_dataset(args.dataset_name, dataset_config_arg)
    except Exception as e:
        print(f"Failed to load dataset {args.dataset_name} with config {dataset_config_arg}: {e}")
        return
    print(f"Dataset loaded. Available splits: {list(full_dataset_dict.keys())}")

    # Pass tokenizer and min_token_length to save_activations_for_split
    common_save_args = {
        "model": model, "tokenizer": tokenizer, "layer_idx": args.layer_idx,
        "random_init_from_seed": args.random_init_from_seed,
        "dataset_name": args.dataset_name, "dataset_config": dataset_config_arg,
        "activation_dl_batch_size": args.activation_dl_batch_size,
        "text_dl_batch_size": args.text_dl_batch_size, "max_length": args.max_length,
        "dtype": torch_dtype, "device": device, "output_dir": args.output_dir,
        "activation_type": args.activation_type,
        "min_token_length": args.min_token_length, # MODIFIED: Pass new arg
        "activs_per_file" : args.activs_per_file  # Default chunk size for DataLoader
    }

    if not args.no_train and args.train_split_name in full_dataset_dict:
        save_activations_for_split(
            split_name=args.train_split_name,
            dataset_obj=full_dataset_dict[args.train_split_name],
            num_act_batches_to_save=args.num_train_act_batches,
            max_text_samples=args.max_train_text_samples,
            **common_save_args
        )
    # ... (similar modification for validation split call) ...
    elif not args.no_train:
        print(f"Train split '{args.train_split_name}' not found. Skipping.")

    if not args.no_val and args.val_split_name in full_dataset_dict:
        save_activations_for_split(
            split_name=args.val_split_name,
            dataset_obj=full_dataset_dict[args.val_split_name],
            num_act_batches_to_save=args.num_val_act_batches,
            max_text_samples=args.max_val_text_samples,
            **common_save_args
        )
    elif not args.no_val: # Validation not found, try to create from train
        if args.train_split_name in full_dataset_dict and \
           args.max_val_text_samples is not None and args.max_val_text_samples > 0:
            print(f"Validation split '{args.val_split_name}' not found. Attempting to create from '{args.train_split_name}'.")
            train_data_for_split = full_dataset_dict[args.train_split_name]
            val_sample_count = min(args.max_val_text_samples, len(train_data_for_split) // 10)
            if val_sample_count == 0 and len(train_data_for_split) > 0: val_sample_count = 1
            
            if len(train_data_for_split) - val_sample_count <= 0 or val_sample_count == 0:
                print(f"Train split too small or val_sample_count is 0. Skipping validation creation.")
            else:
                try:
                    split_result = train_data_for_split.train_test_split(test_size=val_sample_count, shuffle=True, seed=42)
                    print(f"Created temp validation set. Processing as '{args.val_split_name}_from_train'.")
                    save_activations_for_split(
                        split_name=f"{args.val_split_name}_from_train",
                        dataset_obj=split_result['test'],
                        num_act_batches_to_save=args.num_val_act_batches,
                        max_text_samples=val_sample_count, # Use the actual count
                        **common_save_args
                    )
                except Exception as e:
                    print(f"Could not create validation split from train: {e}")
        else:
             print(f"Validation split '{args.val_split_name}' not found and cannot create from train. Skipping.")


    print("\nActivation saving process complete.")

if __name__ == "__main__":
    main()