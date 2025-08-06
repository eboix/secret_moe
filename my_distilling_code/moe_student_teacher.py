import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse
os.chdir('/u/eboix/moe_experiment')

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define MLP (used for both standalone model and MoE experts)
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Define MoE with Top-K (where each expert is an MLP)
class MoE_TopK(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, k, hidden_dim=32,bias=False):
        super().__init__()
        self.num_experts = num_experts
        self.k = k  # Number of selected experts

        # Gating network
        self.gate = nn.Linear(input_dim, num_experts,bias=bias)

        self.expert_tally = nn.Parameter(torch.zeros(self.num_experts))
        self.expert_tally.requires_grad = False

        # Expert networks (each expert is an MLP)
        self.experts = nn.ModuleList([MLP(input_dim, output_dim, hidden_dim) for _ in range(num_experts)])

    def forward(self, x):
        gate_scores = self.gate(x)  # (batch_size, num_experts) 
        topk_vals, topk_idxs = torch.topk(gate_scores, self.k, dim=-1)  # Get top-k expert indices
        # print(topk_idxs.shape)
        # Count the number of occurrences of each index in topk_idxs
        # unique_idxs, counts = torch.unique(topk_idxs, return_counts=True)
        # print(unique_idxs)
        # print(counts)
        # print(self.expert_tally)
        # self.expert_tally[unique_idxs] += counts
        # print('before',self.expert_tally)
        # print(topk_idxs.shape)
        # for i in range(topk_idxs.shape[0]):
        #     for j in range(topk_idxs.shape[1]):
        #         print(i,j)
        #         self.expert_tally[topk_idxs[j]] += 1
        # print('after',self.expert_tally)
        topk_weights = torch.softmax(topk_vals, dim=-1)  # Normalize weights over top-k

        batch_size, _ = x.shape
        expert_outputs = torch.stack([self.experts[i](x) for i in range(self.num_experts)], dim=1)  # (batch_size, num_experts, output_dim)
        selected_expert_outputs = torch.gather(expert_outputs, 1, topk_idxs.unsqueeze(-1).expand(-1, -1, expert_outputs.shape[-1]))

        output = torch.sum(selected_expert_outputs * topk_weights.unsqueeze(-1), dim=1)
        return output

# Generate data from a teacher model
def generate_teacher_data(teacher_model, n_samples=1000, input_dim=10):
    X = torch.randn(n_samples, input_dim).to(device)
    Y = teacher_model(X).detach()
    return X, Y

# Training function
def train(model, train_loader, criterion, optimizer, num_epochs=20):
    model.to(device)
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move data to GPU
            
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch) * output.shape[1]
            loss.backward()
            optimizer.step()

        # if epoch % 1 == 0:
        #     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    return loss.item()

# Evaluation function
def evaluate(model, test_loader):
    model.to(device)
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move data to GPU
            output = model(x_batch)
            total_loss += criterion(output, y_batch).item() * output.shape[1]

    # print(f"Test Loss: {total_loss / len(test_loader):.4f}")
    model.train()
    return total_loss / len(test_loader)

def evaluate_zero(test_loader):
    total_loss = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move data to GPU
            output = torch.zeros_like(y_batch)
            total_loss += criterion(output, y_batch).item() * output.shape[1]

    # print(f"Test Loss: {total_loss / len(test_loader):.4f}")
    return total_loss / len(test_loader)

import pickle
from tqdm import tqdm
import math
def generate_datapoint(d,
                       k1,num_experts1,active_neurons1,
                       k2,num_experts2,active_neurons2,
                       num_epochs,lr,batch_size,
                       trial,
                       bias=False):
    input_dim = d
    output_dim = d
    hidden_dim1 = active_neurons1 // k1
    hidden_dim2 = active_neurons2 // k2
    active_neurons1 = hidden_dim1 * k1
    active_neurons2 = hidden_dim2 * k2
    assert(hidden_dim1 * k1 == active_neurons1)
    assert(hidden_dim2 * k2 == active_neurons2)
    if bias:
        bias_str = '_bias'
    else:
        bias_str = ''
    filename = f'd{d}_{num_experts1}e{k1}a{active_neurons1}s_{num_experts2}e{k2}a{active_neurons2}s{bias_str}_lr{lr}_b{batch_size}_e{num_epochs}_trial{trial}.pkl'
    experiment_filename = 'experiment_data/' + filename
    if not os.path.exists('experiment_data'):
        os.makedirs('experiment_data')
    if os.path.exists(experiment_filename):
        print(f'Experiment file {experiment_filename} already exists. Loading data.')
        data = pickle.load(open(experiment_filename, 'rb'))
        return data
    else:
        # return {'train_losses': [-1], 'test_losses': [math.inf], 'base_losses': [1]}
        print(f'Generating teacher model with {k1} / {num_experts1} experts and {active_neurons1} active neurons')
        # Generate dataset using the teacher model
        teacher_model = MoE_TopK(input_dim, output_dim, num_experts1, k1, hidden_dim1).to(device)


        # Train MoE_TopK with cosine decay learning rate
        print(f'Training student model with {k2} / {num_experts2} experts and {active_neurons2} active neurons')
        student_model = MoE_TopK(input_dim, output_dim, num_experts2, k2, hidden_dim2,bias=bias).to(device)
        base_optimizer = optim.AdamW([
            {'params': student_model.gate.parameters(), 'lr' : lr},  # Gate parameters
            {'params': student_model.experts.parameters(), 'lr' : lr}  # Expert parameters
        ])

        # Define a cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=num_epochs)

        train_losses = []
        test_losses = []
        base_losses = []

        for epoch in tqdm(range(num_epochs)):
            print(f'Generating train and test for epoch {epoch}')
            X_train, Y_train = generate_teacher_data(teacher_model, n_samples=2**18, input_dim=input_dim)
            X_test, Y_test = generate_teacher_data(teacher_model, n_samples=2**12, input_dim=input_dim)
            train_dataset = TensorDataset(X_train, Y_train)
            test_dataset = TensorDataset(X_test, Y_test)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            train_loss = train(student_model, train_loader, nn.MSELoss(), base_optimizer, 1)
            test_loss = evaluate(student_model, test_loader)
            base_loss = evaluate_zero(test_loader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            base_losses.append(base_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Base Loss: {base_loss:.4f}")
            scheduler.step()  # Update learning rate
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'base_losses': base_losses,
        }
        pickle.dump(data, open(experiment_filename, 'wb'))
        return data



def main():
    parser = argparse.ArgumentParser(description="MoE and Teacher Experiment")
    parser.add_argument('--d', type=int, default=256, help='Input dimension')
    parser.add_argument('--k1', type=int, default=2, help='Number of experts for teacher model')
    parser.add_argument('--num_experts1', type=int, default=4, help='Total number of experts for teacher model')
    parser.add_argument('--active_neurons1', type=int, default=256, help='Number of active neurons for teacher model')
    parser.add_argument('--k2', type=int, default=4, help='Number of experts for student model')
    parser.add_argument('--num_experts2', type=int, default=8, help='Total number of experts for student model')
    parser.add_argument('--active_neurons2', type=int, default=320, help='Number of active neurons for student model')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--trial', type=int, default=0, help='Trial number')
    parser.add_argument('--bias', action='store_true', help='Use bias in the gating network')
    args = parser.parse_args()
    d = args.d
    k1 = args.k1
    num_experts1 = args.num_experts1
    active_neurons1 = args.active_neurons1
    k2 = args.k2
    num_experts2 = args.num_experts2
    active_neurons2 = args.active_neurons2
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    trial = args.trial
    bias = args.bias
    print(f'Input dimension: {d}')
    print(f'Number of experts for teacher model: {num_experts1}')
    print(f'Number of active neurons for teacher model: {active_neurons1}')
    print(f'Number of experts for student model: {num_experts2}')
    print(f'Number of active neurons for student model: {active_neurons2}')
    print(f'Number of epochs: {num_epochs}')
    print(f'Learning rate: {lr}')
    print(f'Batch size: {batch_size}')

    datum = generate_datapoint(d,
                            k1,num_experts1,active_neurons1,
                            k2,num_experts2,active_neurons2,
                            num_epochs,lr,batch_size,
                            trial=trial,bias=bias)
    print(datum['test_losses'][-1])
    print(datum['base_losses'][-1])


if __name__ == '__main__':
    main()