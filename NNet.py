from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class NNet(nn.Module):
    def __init__(self,
                 train,
                 test,
                 epochs=10000,
                 lr=0.5,
                 batch_size=100,
                 hidden_nodes=100
                 ):
        super().__init__()

        self.epochs = epochs
        self.features = train.__features__()
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test, batch_size=1, shuffle=False)
        self.model = nn.Sequential(nn.Linear(self.features, hidden_nodes),
                                   nn.SELU(),
                                   nn.Linear(hidden_nodes, hidden_nodes),
                                   nn.SELU(),
                                   nn.Linear(hidden_nodes, 1),
                                   nn.Sigmoid()).double()
        # Define the loss
        self.criterion = nn.NLLLoss()
        # Optimizers require the parameters to optimize and a learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_model(self):
        for e in range(self.epochs):
            running_loss = 0
            for (data, label) in self.train_loader:
                data = data.double()
                label = label.long()
                output = self.model(data)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                running_loss = loss.item()
            print("Epoch: {}/{}\t Loss: {:.4f}".format(e + 1, self.epochs, running_loss))

    def predict(self):
        res = []
        for data in self.test_loader:
            data = data.double()
            out = str(self.model(data).tolist())
            res.append(out[2:-2])
        return res

