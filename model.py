import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(x)
        out = self.relu(out)
        out = self.l3(out)
        return out

    import torch
    import torch.nn as nn

    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(NeuralNet, self).__init__()
            self.l1 = nn.Linear(input_size, hidden_size)
            self.l2 = nn.Linear(hidden_size, hidden_size)
            self.l3 = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.l1(x))
            x = self.relu(self.l2(x))
            x = self.l3(x)
            return x

    def forward(self, x):
        hidden_output = self.hidden_layer(x)
        hidden_output = self.relu(hidden_output)
        output = self.output_layer(hidden_output)
        return output
