from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader


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
                                   nn.Linear(hidden_nodes, int(hidden_nodes/2)),
                                   nn.SELU(),
                                   nn.Linear(int(hidden_nodes/2), 1),
                                   nn.Sigmoid()).double()
        # Define the loss
        self.criterion = nn.MSELoss()
        # Optimizers require the parameters to optimize and a learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_model(self):
        losses = []
        for e in range(self.epochs):
            running_loss = 0
            for (data, label) in self.train_loader:
                data = data.double()
                label = label.double()
                output = self.model(data)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                running_loss = loss.item()
            losses.append(running_loss)
            if len(losses) == 10:
                if sum(losses)/10 <= 0.001:
                    break
                else:
                    losses = []
            print("Epoch: {}/{}\t Loss: {:.4f}".format(e + 1, self.epochs, running_loss))

    def predict(self):
        res = []
        for data in self.test_loader:
            data = data.double()
            out = str(self.model(data).tolist())
            res.append(out[2:-2])
        return res

    def lr_schedule(self, epoch):
        """returns a custom learning rate
           that decreases as epochs progress.
        """
        epochs = self.epochs
        learning_rate = 0.7
        if epoch > epochs * 0.5:
            learning_rate = 0.5
        if epoch > epochs * 0.75:
            learning_rate = 0.01
        if epoch > epochs * 0.9:
            learning_rate = 0.007
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
