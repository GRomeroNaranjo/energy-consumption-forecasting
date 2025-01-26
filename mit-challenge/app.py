import torch
from torch import nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

data = pd.read_csv("PJM(AEP)_energy_dataset.csv")
data_series = data["PJME_MW"].tolist()
num = 100

data_x, data_y = [], []

for i in range(len(data_series) - num):
    data_x.append(data_series[i : i + num])
    data_y.append(data_series[i + num])

@dataclass
class Config():
    sequence_length = 100
    mlp1_output_size = 120
    mlp2_output_size = 60

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.sequence_length)
        self.layer_norm2 = nn.LayerNorm(config.mlp1_output_size)
        self.layer_norm3 = nn.LayerNorm(config.mlp1_output_size)
        self.layer_norm4 = nn.LayerNorm(config.mlp1_output_size)
        self.layer_norm5 = nn.LayerNorm(config.mlp1_output_size)

        self.attention_mechanism1 = AttentionMechanism1(config)
        self.attention_mechanism2 = AttentionMechanism2(config)

        self.mlp1 = MLP1(config)
        self.mlp3 = MLP3(config)
        self.mlp2 = MLP2(config)
        self.mlp4 = MLP4(config)

        self.linear = nn.Linear(config.mlp2_output_size, 1)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.mlp1(x)

        x = self.layer_norm2(x)
        x = self.attention_mechanism1(x)

        x = self.layer_norm3(x)
        x = self.mlp3(x)

        x = self.layer_norm4(x)
        x = self.mlp4(x)

        x = self.layer_norm5(x)
        x = self.attention_mechanism2(x)

        x = self.mlp2(x)
        x = self.linear(x)
        return x
   

class MLP1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.f1 = nn.Linear(config.sequence_length, 60)
        self.r1 = nn.ReLU()
        self.f2 = nn.Linear(60, config.mlp1_output_size)

    def forward(self, x):
        x = self.f1(x)
        x = self.r1(x)
        x = self.f2(x)
        return x
    

class MLP2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.f1 = nn.Linear(config.mlp1_output_size, 80)
        self.r1 = nn.ReLU()
        self.f2 = nn.Linear(80, config.mlp2_output_size)

    def forward(self, x):
        x = self.f1(x)
        x = self.r1(x)
        x = self.f2(x)
        return x  
    
class MLP3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.f1 = nn.Linear(config.mlp1_output_size, 240)
        self.r1 = nn.ReLU()
        self.f2 = nn.Linear(240, config.mlp1_output_size)

    def forward(self, x):
        x = self.f1(x)
        x = self.r1(x)
        x = self.f2(x)
        return x
    
class MLP4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.f1 = nn.Linear(config.mlp1_output_size, 240)
        self.r1 = nn.ReLU()
        self.f2 = nn.Linear(240, config.mlp1_output_size)

    def forward(self, x):
        x = self.f1(x)
        x = self.r1(x)
        x = self.f2(x)
        return x

class AttentionMechanism1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.queries = nn.Linear(config.mlp1_output_size, config.mlp1_output_size)
        self.keys = nn.Linear(config.mlp1_output_size, config.mlp1_output_size)
        self.values = nn.Linear(config.mlp1_output_size, config.mlp1_output_size)
        self.final_linear = nn.Linear(config.mlp1_output_size, config.mlp1_output_size)

    def forward(self, x):
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)
        scores = torch.matmul(queries, keys.T) / (self.config.mlp1_output_size ** 0.5)
        scores = torch.softmax(scores, dim=1)
        output = torch.matmul(scores, values)
        output = self.final_linear(output)
        return output
    
class AttentionMechanism2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.queries = nn.Linear(config.mlp1_output_size, config.mlp1_output_size)
        self.keys = nn.Linear(config.mlp1_output_size, config.mlp1_output_size)
        self.values = nn.Linear(config.mlp1_output_size, config.mlp1_output_size)
        self.final_linear = nn.Linear(config.mlp1_output_size, config.mlp1_output_size)

    def forward(self, x):
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)
        scores = torch.matmul(queries, keys.T) / (self.config.mlp1_output_size ** 0.5)
        scores = torch.softmax(scores, dim=1)
        output = torch.matmul(scores, values)
        output = self.final_linear(output)
        return output

class System():
    def __init__(self, config):
        self.model = Model(config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.criterion = nn.MSELoss()
        self.X_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        
    def fit_scalers(self, X_train, y_train):
        self.X_scaler.fit(np.array(X_train).reshape(-1, 1))
        self.y_scaler.fit(np.array(y_train).reshape(-1, 1))
        
    def scale_data(self, data, scaler, func):
        data = np.array(data).reshape(-1, 1)
        if func == 1:
            return scaler.transform(data).flatten()
        elif func == 0:
            return scaler.inverse_transform(data).flatten()
        else:
            raise ValueError("Invalid scaling function code (must be 0 or 1).")
            
    def train(self, epochs, X_train, y_train, batch_size=64):
        self.fit_scalers(X_train, y_train)
        
        X_train = self.scale_data(X_train, self.X_scaler, 1)
        y_train = self.scale_data(y_train, self.y_scaler, 1)
        
        X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 100)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                
            print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

class Tools():
    def __init__(self, config):
        self.system = System(config)

    def train(self, epochs, X_train, y_train):
        self.system.train(epochs, X_train, y_train)

    def predict(self, x):
        x = self.system.scale_data(x, self.system.X_scaler, 1)
        x = torch.tensor(x, dtype=torch.float32).view(-1, 100)
        y = self.system.model(x)
        y = y.detach().numpy()
        y = self.system.scale_data(y, self.system.y_scaler, 0)
        return y

    def train_accuracy(self, X_train, y_train):
        predictions = self.predict(X_train)
        plt.plot(y_train, label="True")
        plt.plot(predictions, label="Predicted")
        plt.title("Train Accuracy")
        plt.legend()
        plt.show()

    def predict_long(self, x, num):
        prediction_list = []
        for _ in range(num):
            y = self.predict(x)
            prediction_list.append(y[0])
            x = np.roll(x, -1)
            x[-1] = y[0]
        return np.array(prediction_list)

def test_accuracy_individual(X_train, y_train, X_test, y_test, config, epochs):
    tools = Tools(config)
    tools.train(epochs, X_train, y_train)
    predictions = tools.predict(X_test)
    plt.plot(y_test, label="True")
    plt.plot(predictions, label="Predicted")
    plt.title("Test Accuracy")
    plt.legend()
    plt.show()

def test_accuracy_whole_predictions(X_train, y_train, X_test, y_test, config, epochs, num):
    tools = Tools(config)
    tools.train(epochs, X_train, y_train)
    predicted_values = tools.predict_long(X_test[:100], num)
    plt.plot(y_test[:num], label="True")
    plt.plot(predicted_values, label="Predicted")
    plt.title("Test Accuracy")
    plt.legend()
    plt.show()
"""
data_x = data_x[10000:]
data_y = data_y[10000:]
"""
"""
tools = Tools(config)
tools.train(300, data_x, data_y)
tools.train_accuracy(data_x, data_y)
"""

config = Config()
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.01, random_state=42)

"test_accuracy_individual(X_train, y_train, X_test, y_test, config, 500)"
test_accuracy_whole_predictions(X_train, y_train, X_test, y_test, config, 10, len(y_test))
