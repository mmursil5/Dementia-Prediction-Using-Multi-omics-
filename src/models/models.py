import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
import pandas as pd


class RegularizedDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(RegularizedDNN, self).__init__()

        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, num_classes)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(16)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)
        x = self.bn4(self.layer4(x))
        x = self.output_layer(x)
        return x


class PyTorchModelWrapper(BaseEstimator):
    """
    Wrapper for using the trained PyTorch model with sklearn tools
    like permutation_importance.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        # Not used in our pipeline; model is trained elsewhere.
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, pd.Series):
            y_tensor = torch.tensor(y.values, dtype=torch.long)
        else:
            y_tensor = torch.tensor(y, dtype=torch.long)

        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5)

        num_epochs = 10  # small stub training (unused in practice)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            _, y_pred = torch.max(probs, 1)
        return y_pred.numpy()

    def predict_proba(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.numpy()
