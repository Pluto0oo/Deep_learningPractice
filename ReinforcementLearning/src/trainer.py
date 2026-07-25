import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class BehavioralCloningTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = None
        self.loss_history = []
        self.accuracy_history = []

    def train(self, states, actions, epochs=100, batch_size=32, lr=0.001):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(states, actions)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in tqdm(range(epochs), desc="Training BC model"):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_states, batch_actions in dataloader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_states)
                loss = F.cross_entropy(outputs, batch_actions)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * batch_states.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_actions.size(0)
                correct += (predicted == batch_actions).sum().item()

            avg_loss = total_loss / len(dataset)
            accuracy = correct / total
            
            self.loss_history.append(avg_loss)
            self.accuracy_history.append(accuracy)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        return self.loss_history, self.accuracy_history

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"BC model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"BC model loaded from {path}")
