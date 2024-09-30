from tqdm import tqdm
import numpy
import torch
import torchmetrics
from torch.utils.data import Dataset, DataLoader


class SignalsDataset(Dataset):
    def __init__(self, X_l:torch.Tensor, X_r:torch.Tensor, y_l:torch.Tensor, y_r:torch.Tensor)->None:
        self.X:torch.Tensor = torch.cat((X_l, X_r), dim=0)
        self.y:torch.Tensor = torch.cat((y_l, y_r), dim=0)
        self.n_samples:int = self.X.shape[0]

    def __getitem__(self, index:int):
        return (self.X[index], self.y[index])
    
    def __len__(self)->int:
        return self.n_samples


class MLP(torch.nn.Module):
    def __init__(self)->None:
        super(MLP, self).__init__()

        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(200*19, 250, dtype=torch.float32)
        self.linear2 = torch.nn.Linear(250, 2, dtype=torch.float32)
    
    def forward(self, X:torch.Tensor)->torch.Tensor:
        out:torch.Tensor = torch.nn.functional.relu(self.linear1(self.flatten(X)))
        out:torch.Tensor = torch.nn.functional.log_softmax(self.linear2(out), dim=1)
        return out

class CNN(torch.nn.Module):
    def __init__(self)->None:
        super(CNN, self).__init__()
        
        self.conv1:torch.nn.Conv2d = torch.nn.Conv2d(in_channels=61, out_channels=122, kernel_size=9) # (61, 100, 19) -> (122, ?, ?)
        self.batch_norm:torch.nn.BatchNorm2d = torch.nn.BatchNorm2d(122)

    def forward(self, X:torch.Tensor)->torch.Tensor:
        pass



def train(model, device:str, train_loader:DataLoader, optimizer):
    model.train()
    for batch_index, (data, labels) in enumerate(tqdm(train_loader)):
        data, labels = data.to(device), labels.reshape(-1,).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, labels.type(torch.long)) # negative log likelihood
        loss.backward()
        optimizer.step()


def eval(model, device:str, test_loader:DataLoader)->None:
    model.eval()
    test_loss:float = 0
    correct:int = 0
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            data, label = data.to(device), label.reshape(-1,).to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, label.type(torch.long), reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy:float = (100.0 * correct) / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
