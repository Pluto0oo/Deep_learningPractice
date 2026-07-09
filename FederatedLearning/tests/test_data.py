import pytest
import numpy as np
from unittest.mock import MagicMock
from src.data import partition_data_iid, partition_data_dirichlet


def test_partition_data_iid():
    mock_dataset = MagicMock()
    mock_dataset.targets = np.random.randint(0, 10, 1000)
    mock_dataset.__len__.return_value = len(mock_dataset.targets)
    num_clients = 3
    client_indices = partition_data_iid(mock_dataset, num_clients)

    assert len(client_indices) == num_clients
    total_indices = sum(len(indices) for indices in client_indices.values())
    assert total_indices == len(mock_dataset.targets)


def test_partition_data_dirichlet():
    mock_dataset = MagicMock()
    mock_dataset.targets = np.random.randint(0, 10, 1000)
    num_clients = 3
    client_indices = partition_data_dirichlet(mock_dataset, num_clients, alpha=0.5)

    assert len(client_indices) == num_clients
    total_indices = sum(len(indices) for indices in client_indices.values())
    assert total_indices == len(mock_dataset.targets)