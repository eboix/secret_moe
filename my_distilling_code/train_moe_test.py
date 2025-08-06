import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define MLP
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

# Define MoE with Top-K
class MoE_TopK(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, k):
        super().__init__()
        self.num_experts = num_experts
        self.k = k  # Number of selected experts
        
        # Gating network
        self.gate = nn.Linear(input_dim, num_experts)

        # Expert networks
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])

    def forward(self, x):
        gate_scores = self.gate(x)
        topk_vals, topk_idxs = torch.topk(gate_scores, self.k, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)

        batch_size, _ = x.shape
        expert_outputs = torch.stack([self.experts[i](x) for i in range(self.num_experts)], dim=1)  # (batch_size, num_experts, output_dim)
        selected_expert_outputs = torch.gather(expert_outputs, 1, topk_idxs.unsqueeze(-1).expand(-1, -1, expert_outputs.shape[-1]))

        output = torch.sum(selected_expert_outputs * topk_weights.unsqueeze(-1), dim=1)
        return output

# Generate data from a teacher model
def generate_teacher_data(teacher_model, n_samples=1000, input_dim=10):
    X = torch.randn(n_samples, input_dim)
    Y = teacher_model(X).detach()  # Get teacher's outputs
    return X, Y

# Training function
def train(model, train_loader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            output = model(x_batch)
            total_loss += criterion(output, y_batch).item()
    print(f"Test Loss: {total_loss / len(test_loader):.4f}")
    model.train()

# Hyperparameters
input_dim = 10
output_dim = 5
hidden_dim = 32
num_experts = 4
k = 2
batch_size = 32
num_epochs = 20

# Choose teacher model (MLP or MoE_TopK)
use_mlp_teacher = True  # Set False to use MoE_TopK as teacher

if use_mlp_teacher:
    print("\nUsing MLP as teacher...")
    teacher_model = MLP(input_dim, output_dim, hidden_dim)
else:
    print("\nUsing MoE_TopK as teacher...")
    teacher_model = MoE_TopK(input_dim, output_dim, num_experts, k)

# Generate dataset using the teacher model
X_train, Y_train = generate_teacher_data(teacher_model, n_samples=800, input_dim=input_dim)
X_test, Y_test = generate_teacher_data(teacher_model, n_samples=200, input_dim=input_dim)

# DataLoader
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train MLP
print("\nTraining MLP on teacher-generated data...")
mlp_model = MLP(input_dim, output_dim, hidden_dim)
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.01)
train(mlp_model, train_loader, nn.MSELoss(), mlp_optimizer, num_epochs)
evaluate(mlp_model, test_loader)

# Train MoE_TopK
print("\nTraining MoE_TopK on teacher-generated data...")
moe_model = MoE_TopK(input_dim, output_dim, num_experts, k)
moe_optimizer = optim.Adam(moe_model.parameters(), lr=0.01)
train(moe_model, train_loader, nn.MSELoss(), moe_optimizer, num_epochs)
evaluate(moe_model, test_loader)
