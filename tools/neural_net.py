import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import numpy as np

# C0_05 IS IMG_4014
# C0_06 IS IMG_4014
# C0_07 is IMG_4014
# C0_09 IS IMG_4015
# C0_10 is IMG_4015
# C0_13 is IMG_4013 TWICE
# C0_13 is IMG_4010 TWICE


n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)


class Move(Dataset):
    def __init__(self) -> None:
        super().__init__()
        with open("saves/neural", 'rb') as f:
            neural_data = np.load(f)
        self.x = torch.from_numpy(neural_data[:, :266])
        self.y = torch.from_numpy(neural_data[:, 266:])
        self.n_samples = len(neural_data[0])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.sigmoid(self.l1(x))
        x = F.sigmoid(self.l2(x))
        return F.sigmoid(self.l3(x))


dataset = Move()
train_data, test_data = torch.utils.data.random_split(dataset, [(int)(len(dataset)*0.8), len(
    dataset)-(int)(len(dataset)*0.8)], generator=torch.Generator().manual_seed(random_seed))
train_loader = DataLoader(
    dataset=train_data, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(
    dataset=test_data, batch_size=batch_size_test, shuffle=False)

mse = nn.MSELoss
model = Model(266, 300, 6)
optim = torch.optim.SGD(
    lr=learning_rate, params=model.parameters(), momentum=momentum)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
total_steps = len(train_loader)
for epoch in range(n_epochs):
    for i, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = mse(output, label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % log_interval == 0:
            print(
                f"EPOCH: {epoch}/{n_epochs}, step {i}/{total_steps}, loss = {loss}")


total_tests = len(test_loader)
sum_err = torch.zeros(6)
with torch.no_grad():
    for (data, label) in enumerate(test_loader):
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = mse(output, label)
        sum_err += loss

print(sum_err/total_tests)
