import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import flwr as fl
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict

from .models import create_model


class FedAvgClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        config: Dict
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer()

    def _create_optimizer(self) -> optim.Optimizer:
        optimizer_type = self.config['training'].get('optimizer', 'sgd')
        lr = self.config['training'].get('learning_rate', 0.01)
        momentum = self.config['training'].get('momentum', 0.9)
        weight_decay = self.config['training'].get('weight_decay', 1e-4)

        if optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.train()

        num_epochs = config.get('epochs', self.config['training'].get('epochs', 5))
        lr = config.get('lr', self.config['training'].get('learning_rate', 0.01))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct / total

            print(f"Client {self.client_id} | Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()

        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

        test_loss /= total
        test_acc = correct / total

        print(f"Client {self.client_id} | Evaluation | Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")

        return test_loss, total, {"accuracy": test_acc}


class FedProxClient(FedAvgClient):
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        config: Dict
    ):
        super().__init__(client_id, model, train_loader, device, config)
        self.mu = config['training'].get('mu', 0.01)
        self.global_params = None

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        self.global_params = [torch.Tensor(p).to(self.device) for p in parameters]
        return super().fit(parameters, config)

    def _compute_prox_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.criterion(output, target)

        prox_loss = 0.0
        for param, global_param in zip(self.model.parameters(), self.global_params):
            prox_loss += torch.norm(param - global_param) ** 2

        return ce_loss + (self.mu / 2) * prox_loss


def create_client(
    client_id: int,
    train_loader: DataLoader,
    config: Dict
) -> fl.client.NumPyClient:
    device = torch.device(config['experiment'].get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    model = create_model(config)

    algorithm = config['experiment'].get('algorithm', 'fedavg')

    if algorithm == 'fedavg':
        return FedAvgClient(client_id, model, train_loader, device, config)
    elif algorithm == 'fedprox':
        return FedProxClient(client_id, model, train_loader, device, config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def start_client(client_fn, server_address: str = "127.0.0.1:8080") -> None:
    fl.client.start_numpy_client(server_address=server_address, client_fn=client_fn)