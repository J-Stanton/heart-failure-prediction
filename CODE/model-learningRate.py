import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


data = pd.read_csv('../heart.csv')

cleanData = data.dropna()
cleanData = cleanData.drop_duplicates()
encodedData = pd.get_dummies(cleanData, columns=[
    'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'
])
scaler = StandardScaler()

numericalColumns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
encodedData[numericalColumns] = scaler.fit_transform(encodedData[numericalColumns])


X = encodedData.drop('HeartDisease', axis=1)
y = encodedData['HeartDisease']

tempX, testX, tempY, testY = train_test_split(X, y, test_size=0.2)
trainX, valX, trainY, valY = train_test_split(tempX, tempY, test_size=0.25)

trainX = trainX.astype('float32')
testX = testX.astype('float32')

X_train_tensor = torch.tensor(trainX.values, dtype=torch.float32)
y_train_tensor = torch.tensor(trainY.values, dtype=torch.float32).unsqueeze(1)

X_val_tensor = torch.tensor(valX.astype('float32').values)
y_val_tensor = torch.tensor(valY.values, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(testX.values, dtype=torch.float32)
y_test_tensor = torch.tensor(testY.values, dtype=torch.float32).unsqueeze(1)

trainDataset = TensorDataset(X_train_tensor, y_train_tensor)
trainLoader = DataLoader(trainDataset, batch_size=16, shuffle=True)

class HeartNet(nn.Module):
    def __init__(self, input_size):
        super(HeartNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

model = HeartNet(trainX.shape[1])

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
learningRates = [0.0001, 0.001, 0.01, 0.05]
results = {}

for lr in learningRates:
    print(f"\nTraining with learning rate: {lr}")
    
    model = HeartNet(trainX.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    trainLosses = []
    valLosses = []

    for epoch in range(epochs):
        model.train()
        batchLosses = []
        for batchX, batchY in trainLoader:
            optimizer.zero_grad()
            outputs = model(batchX)
            loss = criterion(outputs, batchY)
            loss.backward()
            optimizer.step()
            batchLosses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            valOutput = model(X_val_tensor)
            valLoss = criterion(valOutput, y_val_tensor)
        
        trainLosses.append(sum(batchLosses) / len(batchLosses))
        valLosses.append(valLoss.item())
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {trainLosses[-1]:.4f} | Val Loss: {valLosses[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).round()
        testAccuracy = accuracy_score(y_test_tensor, predictions)

    results[lr] = {
        'trainLosses': trainLosses,
        'valLosses': valLosses,
        'testAccuracy': testAccuracy
    }

    print(f"Test Accuracy with LR={lr}: {testAccuracy:.4f}")

plt.figure(figsize=(12, 6))
for lr, metrics in results.items():
    plt.plot(metrics['trainLosses'], label=f'Train LR={lr}')
    plt.plot(metrics['valLosses'], linestyle='--', label=f'Val LR={lr}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Val Loss for Different Learning Rates')
plt.legend()
plt.tight_layout()
plt.savefig('loss_comparison.png')

print("\nSummary of Learning Rates:")
for lr, metrics in results.items():
    print(f"LR={lr}: Final Val Loss={metrics['valLosses'][-1]:.4f}, Test Accuracy={metrics['testAccuracy']:.4f}")

