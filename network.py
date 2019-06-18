import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.linear2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.linear3 = nn.Linear(hidden_size, output_size).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))

        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        action_size = 4
        self.linear1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.linear2 = nn.Linear(hidden_size+action_size, hidden_size).to(self.device)
        self.linear3 = nn.Linear(hidden_size, output_size).to(self.device)

    def forward(self, state, action):
        x = F.relu(self.linear1(state))
        x = torch.cat((x, action.float()), dim=1)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

