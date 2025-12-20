import os
import torch
os.chdir('/home/eboix/projects/secret_moe/massive_distillation')
import pickle
from utils import MultiFileActivationDataLoader
import argparse

parser = argparse.ArgumentParser(description="Compute mean and covariance from activations.")
parser.add_argument("--activ_dir", type=str, help="Directory containing activation files")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for processing activations")
parser.add_argument('--out_file', type=str, default='mean_and_cov.pt', help='Output file for mean and covariance')
args = parser.parse_args()

activ_dir = args.activ_dir
batch_size = args.batch_size
out_file = args.out_file

os.makedirs(os.path.dirname(out_file), exist_ok=True)
if os.path.exists(out_file):
    print(f"Output file {out_file} already exists. Exiting to avoid overwriting.")
    exit(0)

batch_size = 1024
file_dataloader = MultiFileActivationDataLoader(activ_dir, batch_size=batch_size, shuffle=True, shuffle_files=False)

# Compute mean of all activations from file_dataloader
# load batches and computing running mean
activ_mean = None
batch_num = 0
for batch in file_dataloader:
    batch_mean = batch.mean(dim=0)
    if activ_mean is None:
        activ_mean = batch_mean
    else:
        activ_mean = activ_mean * (batch_num / (batch_num + 1)) + batch_mean / (batch_num + 1)
    
    batch_num += 1
print(f"Mean computed over {batch_num} batches.")

batch_size = 1024
file_dataloader = MultiFileActivationDataLoader(activ_dir, batch_size=batch_size, shuffle=True, shuffle_files=False)

# Compute covariance of all activations from file_dataloader
activ_cov = None
batch_num = 0
for batch in file_dataloader:
    batch_centered = batch - activ_mean
    batch_cov = torch.mm(batch_centered.t(), batch_centered) / (batch.size(0) - 1)

    if activ_cov is None:
        activ_cov = batch_cov
    else:
        activ_cov = activ_cov * (batch_num / (batch_num + 1)) + batch_cov / (batch_num + 1)

    batch_num += 1

print(f'Cov computed over {batch_num} batches.')

pickle.dump({'mean' : activ_mean, 'cov' : activ_cov}, open(out_file, 'wb'))

print('Saved')