import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, generate_exp_id
from src.data import prepare_client_data
from src.server import create_server
from src.client import create_client
from src.results import save_experiment_results
from src.utils import set_seed


def run_experiment(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(config['experiment'].get('seed', 42))

    exp_id = generate_exp_id(config)
    exp_dir = os.path.join(config['results'].get('save_dir', './results'), exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Starting experiment: {exp_id}")
    print(f"Algorithm: {config['experiment']['algorithm']}")
    print(f"Model: {config['model']['type']}")
    print(f"Number of clients: {config['experiment']['num_clients']}")
    print(f"Number of rounds: {config['experiment']['num_rounds']}")

    client_loaders, test_loader, client_sizes = prepare_client_data(config)

    print("\nClient data distribution:")
    for client_id, size in client_sizes.items():
        print(f"  Client {client_id}: {size} samples")

    server = create_server(config, test_loader)
    strategy = server.create_strategy()

    def client_fn(cid: str):
        client_id = int(cid)
        client = create_client(client_id, client_loaders[client_id], config)
        return client.to_client()

    import flwr as fl
    num_clients = config['experiment'].get('num_clients', 3)
    num_rounds = config['experiment'].get('num_rounds', 50)

    print("\nStarting federated simulation...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.3} if config['experiment']['device'] == 'cuda' else {"num_cpus": 1},
    )

    save_experiment_results(config, server.results_history, exp_dir)

    print(f"\nExperiment completed! Results saved to {exp_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_federated.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    run_experiment(config_path)