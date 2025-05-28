import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('../heart.csv')


cleanData = data.dropna()
cleanData = cleanData.drop_duplicates()
dataEncoded = pd.get_dummies(cleanData, columns=[
    'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'
])
scaler = StandardScaler()
numericalColumns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
dataEncoded[numericalColumns] = scaler.fit_transform(dataEncoded[numericalColumns])


X = dataEncoded.drop('HeartDisease', axis=1)
y = dataEncoded['HeartDisease']


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

trainLosses = []
valLosses = []

epochs = 50
for epoch in range(epochs):
    model.train()
    epochLoss = 0.0
    for batchX, batchY in trainLoader:
        optimizer.zero_grad()
        outputs = model(batchX)
        loss = criterion(outputs, batchY)
        loss.backward()
        optimizer.step()
        epochLoss += loss.item()

    avgTrainLoss = epochLoss / len(trainLoader)
    trainLosses.append(avgTrainLoss)

    model.eval()
    with torch.no_grad():
        valOutputs = model(X_val_tensor)
        valLoss = criterion(valOutputs, y_val_tensor)
        valLosses.append(valLoss.item())

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train_Loss: {avgTrainLoss:.4f}, Val Loss: {valLoss.item():.4f}")

plt.figure(figsize=(10, 5))
plt.plot(trainLosses, label='Train Loss')
plt.plot(valLosses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.close()

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predictions = predictions.round()



testAccuracy = accuracy_score(y_test_tensor, predictions)
precision = precision_score(y_test_tensor, predictions)
recall = recall_score(y_test_tensor, predictions)
f1 = f1_score(y_test_tensor, predictions)
conMatrix = confusion_matrix(y_test_tensor, predictions)

print(f"\nTest Accuracy: {testAccuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_tensor, predictions))


plt.figure(figsize=(6, 5))
sns.heatmap(conMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
plt.close()