import torch
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from tqdm import tqdm
from torch.utils.data import DataLoader
import random
from transformers import AutoModelForCausalLM
import transformers
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

# os.chdir("/home/eboix/projects/secret_moe/massive_distillation")

from utils import MultiFileActivationDataLoader, extract_layer_mlp
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Distill teacher MLP to student MLP')
    parser.add_argument('--activ_dir', type=str, required=True,
                        help='Directory containing saved activations')
    parser.add_argument('--valid_activ_dir', type=str, default=None,
                        help='Directory containing validation activations (if different from activ_dir)')
    parser.add_argument('--layer_idx', type=int, required=True,
                        help='Layer index to extract from teacher model')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the teacher model')
    parser.add_argument('--model_revision', type=str, default=None,
                        help='Revision of the teacher model')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Base learning rate for optimizer with cosine decay')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Weights and Biases project name for logging (optional)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Weights and Biases run name for logging (optional)')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='Hidden dimension for student MLP (default: 6 * input_dim)')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Folder to save the trained student model (optional)')
    return parser.parse_args()

# An example command to run this script:
# python distill_teacher_to_mlp_student.py \
# --activ_dir /scratch/tmp/eboix/saved_activations/wikitext_wikitext-2-raw-v1_EleutherAI_pythia-410m_layer12_actinput_train_mintok20
# --layer_idx 12 --model_name EleutherAI/pythia-410m \
# --batch_size 2048 --num_epochs 10 --learning_rate 3e-4 \
# --hidden_dim 24576 \
# --wandb_project secret_moe_distillation --wandb_run_name pythia-410m-layer12-mlp-student \
# --output_folder /scratch/tmp/eboix/saved_logs/mlp_distillation/

args = parse_args()

if args.output_folder is not None:
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
if 'output_logs.pkl' in os.listdir(args.output_folder):
    print(f"Output logs already exist in {args.output_folder}, exiting to avoid overwriting.")
    exit(0)

activ_dir = args.activ_dir
valid_activ_dir = args.valid_activ_dir if args.valid_activ_dir else activ_dir.replace("train", "validation")
model_name = args.model_name
layer_idx = args.layer_idx

# Load the model's MLP layer
torch_dtype = torch.float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device', device)

model_kwargs = {'trust_remote_code': True}
if torch_dtype in [torch.float16, torch.bfloat16]: model_kwargs['torch_dtype'] = torch_dtype

if args.model_revision is not None:
    model_kwargs['revision'] = args.model_revision

model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)

# Load model's MLP layer at specified index
mlp_layer = extract_layer_mlp(model, layer_idx)

mlp_layer = copy.deepcopy(mlp_layer)

# clone the MLP layer and delete the rest of the model
mlp_layer = mlp_layer.to(device)
del model
torch.cuda.empty_cache()

print(f"Loaded MLP layer {layer_idx} from model {model_name}")


def get_train_dataloader():
    file_dataloader = MultiFileActivationDataLoader(activ_dir, batch_size=1023, shuffle=True, shuffle_files=False)
    return file_dataloader
def get_val_dataloader():
    file_dataloader = MultiFileActivationDataLoader(valid_activ_dir, batch_size=1023, shuffle=False, shuffle_files=False)
    return file_dataloader


# Estimate mean of activations
def estimate_activation_mean(dataloader, module):
    module.eval()
    # total_count = 0
    input_sum = None
    output_sum = None
    with torch.no_grad():
        num_batches = 0
        print('Estimating activation means...')
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            outputs = module(batch)

            if input_sum is None:
                # input_mean = torch.zeros_like(batch)
                # output_mean = torch.zeros_like(batch)
                input_sum = torch.zeros_like(batch.mean(dim=0))
                output_sum = torch.zeros_like(outputs.mean(dim=0))

            input_sum += batch.mean(dim=0)
            output_sum += outputs.mean(dim=0)
            num_batches += 1
    input_sum /= num_batches
    output_sum /= num_batches
    return {'input' : input_sum, 'output': output_sum}
    # return {'input' : input_mean, 'output' : output_mean, 'alternative_input' : input_sum, 'alternative_output' : output_sum}

def estimate_activation_variance(dataloader, module):
    means = estimate_activation_mean(dataloader, module)
    dataloader.reset()
    input_mean = means['input']
    output_mean = means['output']
    input_var_sum = 0.0
    output_var_sum = 0.0
    with torch.no_grad():
        num_batches = 0
        print('Estimating activation variance...')
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            outputs = module(batch)
            input_var_sum += torch.mean(((batch - input_mean)**2).mean(dim=0))
            output_var_sum += torch.mean(((outputs - output_mean)**2).mean(dim=0))
            num_batches += 1
    input_var_sum /= num_batches
    output_var_sum /= num_batches
    return {'input' : input_var_sum, 'output': output_var_sum}

validation_dataloader = MultiFileActivationDataLoader(valid_activ_dir, batch_size=1024, shuffle=False, shuffle_files=False)
variances = estimate_activation_variance(validation_dataloader, mlp_layer)
output_variance = variances['output']
input_variance = variances['input']


# Distill the model to a student
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, dtype=torch.float32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, dtype=dtype)
        )

    def forward(self, x):
        return self.fc(x)

import transformers

# d_in is this if dense_h_to_4h exists
if hasattr(mlp_layer, "dense_h_to_4h"):
    d_in = mlp_layer.dense_h_to_4h.in_features
elif hasattr(mlp_layer, "up_proj"):
    d_in = mlp_layer.up_proj.in_features
elif hasattr(mlp_layer, "c_fc"):
    d_in = mlp_layer.c_fc.nx
else:
    raise AttributeError("MLP layer has neither 'dense_h_to_4h' nor 'up_proj'")

d_out = d_in
student_model = MLP(input_dim=d_in, output_dim=d_out, hidden_dim=args.hidden_dim, dtype=torch_dtype).to(device)
# student_model = transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXMLP(d_in, 4 * d_in, d_in, dtype=torch_dtype).to(device)

base_lr = args.learning_rate
num_epochs = args.num_epochs
batch_size = args.batch_size

optimizer = torch.optim.Adam(student_model.parameters(), lr=base_lr)
criterion = nn.MSELoss()
# Set up gradient scaler for mixed precision training
scaler = torch.amp.GradScaler(device=device)


file_dataloader = MultiFileActivationDataLoader(activ_dir, batch_size=batch_size, shuffle=True, shuffle_files=False)
total_iterations = len(file_dataloader) * num_epochs

optimizer = torch.optim.Adam(student_model.parameters(), lr=base_lr)
scheduler = CosineAnnealingLR(optimizer, T_max=total_iterations)

validation_losses = []
train_losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    batch_count = 0

    student_model.train()
    file_dataloader = MultiFileActivationDataLoader(activ_dir, batch_size=batch_size, shuffle=True, shuffle_files=False)
    pbar = tqdm(file_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in pbar:
        # Move data to device
        activations = batch.to(device)

        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.amp.autocast(device.type, dtype=torch.float16):
            output = mlp_layer(activations)
            student_output = student_model(activations)
            loss = criterion(student_output, output)
        
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        running_loss += loss.item()
        batch_count += 1
        
        # Update progress bar with current loss
        pbar.set_postfix(loss=f"{loss.item() / output_variance:.6f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")


    avg_loss = running_loss / batch_count
    train_losses.append((epoch,avg_loss))
    print(f"Epoch {epoch+1}, Average Train Loss: {avg_loss / output_variance:.6f}")

    # Validation step
    validation_dataloader = MultiFileActivationDataLoader(valid_activ_dir, batch_size=batch_size, shuffle=False, shuffle_files=False)
    with torch.no_grad():
        student_model.eval()
        val_loss = 0.0
        val_batch_count = 0
        
        for val_batch in validation_dataloader:
            val_activations = val_batch.to(device)
            with torch.amp.autocast(device.type, dtype=torch.float16):
                val_output = mlp_layer(val_activations)
                student_val_output = student_model(val_activations)
                val_loss += criterion(student_val_output, val_output).item()
            val_batch_count += 1

        print('Val batch count', val_batch_count)
        
        avg_val_loss = val_loss / val_batch_count
        validation_losses.append((epoch,avg_val_loss))
        print(f"Validation Loss after Epoch {epoch+1}: {avg_val_loss / output_variance:.6f}")

# dump validation_losses and train_losses, and output_variance to 'output_logs.pkl' in args.output_folder
import pickle
output_logs = {
    'validation_losses': validation_losses,
    'train_losses': train_losses,
    'output_variance': output_variance.item(),
    'input_variance': input_variance.item()
}
if args.output_folder is not None:
    with open(os.path.join(args.output_folder, 'output_logs.pkl'), 'wb') as f:
        pickle.dump(output_logs, f)
    print(f"Saved training logs to {os.path.join(args.output_folder, 'output_logs.pkl')}")


    
