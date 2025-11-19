import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from imblearn.under_sampling import RandomUnderSampler   
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from src.models.models import RegularizedDNN


def train_model_with_randomundersampler(
    X_train_sfs,
    y_train,
    num_classes=2,
    lr=0.0001,
    weight_decay=1e-5,
    num_epochs=240,
    batch_size=64,
    patience_early=160,
):
    """
    Apply RandomUnderSampler on the training set, then train RegularizedDNN.
    """

    # ---- RandomUnderSampler on the (SFS-selected) training set ----
    rus = RandomUnderSampler(random_state=44)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_sfs, y_train)

    X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_resampled.values, dtype=torch.long)

    input_dim = X_train_tensor.shape[1]
    model = RegularizedDNN(input_dim, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=160, factor=0.5)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    best_loss = np.inf
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step(running_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

        if running_loss < best_loss:
            best_loss = running_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience_early:
            print(f'Early stopping after {epoch+1} epochs.')
            break

    return model


def evaluate_model(model, X_test_sfs, y_test):
    """
    Evaluate the trained model on test data and return metrics dict.
    """
    X_test_tensor = torch.tensor(X_test_sfs, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.softmax(logits, dim=1)
        _, y_pred = torch.max(probs, 1)

    accuracy = accuracy_score(y_test_tensor, y_pred)
    precision = precision_score(y_test_tensor, y_pred, average='weighted')
    recall = recall_score(y_test_tensor, y_pred, average='weighted')
    f1 = f1_score(y_test_tensor, y_pred, average='weighted')
    auc = roc_auc_score(y_test_tensor, probs[:, 1])
    cm = confusion_matrix(y_test_tensor, y_pred)

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc,
        'Confusion Matrix': cm
    }

    return metrics, probs[:, 1].numpy(), y_pred.numpy()
