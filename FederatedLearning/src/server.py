import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import flwr as fl
from flwr.server.strategy import FedAvg, Strategy
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

from .models import create_model


class FedAvgServer:
    def __init__(self, config: Dict, test_loader: DataLoader = None):
        self.config = config
        self.test_loader = test_loader
        self.device = torch.device(
            config['experiment'].get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
        )
        self.model = create_model(config)
        self.model.to(self.device)
        self.results_history = []

    def _evaluate_fn(self, server_round: int, parameters: List[np.ndarray], config: Dict) -> Optional[Tuple[float, Dict]]:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v).to(self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        if self.test_loader is None:
            return None

        test_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

        test_loss /= total
        test_acc = correct / total

        print(f"Server Round {server_round} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        self.results_history.append({
            'round': server_round,
            'test_loss': test_loss,
            'test_acc': test_acc
        })

        return test_loss, {"accuracy": test_acc}

    def _get_initial_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def create_strategy(self) -> Strategy:
        num_clients = self.config['experiment'].get('num_clients', 3)
        fraction_fit = self.config['experiment'].get('fraction_fit', 1.0)
        min_fit_clients = self.config['experiment'].get('min_fit_clients', num_clients)
        min_available_clients = self.config['experiment'].get('min_available_clients', num_clients)

        strategy = FedAvg(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=self._evaluate_fn,
            initial_parameters=fl.common.ndarrays_to_parameters(self._get_initial_parameters()),
        )

        return strategy

    def start_server(self, server_address: str = "127.0.0.1:8080") -> fl.server.Server:
        strategy = self.create_strategy()

        num_rounds = self.config['experiment'].get('num_rounds', 50)

        history = fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )

        return history


class FedProxServer(FedAvgServer):
    def __init__(self, config: Dict, test_loader: DataLoader = None):
        super().__init__(config, test_loader)
        self.mu = config['training'].get('mu', 0.01)


def create_server(config: Dict, test_loader: DataLoader = None) -> FedAvgServer:
    algorithm = config['experiment'].get('algorithm', 'fedavg')

    if algorithm == 'fedavg':
        return FedAvgServer(config, test_loader)
    elif algorithm == 'fedprox':
        return FedProxServer(config, test_loader)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")