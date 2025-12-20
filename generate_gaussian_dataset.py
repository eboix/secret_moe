import os
import torch
os.chdir('/home/eboix/projects/secret_moe/massive_distillation')
import pickle
from utils import MultiFileActivationDataLoader
import argparse

parser = argparse.ArgumentParser(description="Generate Gaussian data dataset.")
parser.add_argument("--out_dir", type=str, help="Directory to save generated dataset")
parser.add_argument('--mean_and_cov_file', type=str, help='File containing mean and covariance')
parser.add_argument('--num_samples', type=int, default=3746736, help='Number of samples to generate')
parser.add_argument('--samples_per_file', type=int, default=256000, help='Number of samples per output file')
args = parser.parse_args()

out_dir = args.out_dir
mean_and_cov_file = args.mean_and_cov_file
num_samples = args.num_samples
os.makedirs(out_dir, exist_ok=True)

mean_cov_dict = pickle.load(open(mean_and_cov_file, 'rb'))
activ_mean = mean_cov_dict['mean']
activ_cov = mean_cov_dict['cov']

activ_cov = (activ_cov + activ_cov.T) / 2  # Ensure covariance matrix is symmetric
eigenvalues = torch.linalg.eigh(activ_cov).eigenvalues
assert(eigenvalues[-1] > 1) # make sure that the covariance is not too small, so when we add the diagonal it doesn't change too much
# OLD: 1e-6
# activ_cov = activ_cov + 1e-6 * torch.eye(activ_cov.size(0))  # Add small value to diagonal for numerical stability
activ_cov = activ_cov + 1e-11 * torch.eye(activ_cov.size(0))  # Add small value to diagonal for numerical stability
print(eigenvalues)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
activ_cov = activ_cov.to(device)
activ_mean = activ_mean.to(device)

with torch.no_grad():
    num_generated = 0
    while num_generated < num_samples:
        samples_to_generate = min(args.samples_per_file, num_samples - num_generated)
        samples = torch.distributions.MultivariateNormal(activ_mean, activ_cov).sample((samples_to_generate,)).cpu()
        out_file = os.path.join(out_dir, f'activations_{num_generated // args.samples_per_file}.pt')
        torch.save(samples, out_file)
        print(f'Saved {samples_to_generate} samples to {out_file}')
        num_generated += samples_to_generate
    completed_dir = os.path.join(out_dir, '.completed')
    os.makedirs(completed_dir, exist_ok=True)
    print('Generation complete.')
