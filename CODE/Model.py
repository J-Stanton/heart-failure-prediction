import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv('../heart.csv')

# Preprocessing
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

trainX_tensor = torch.tensor(trainX.values, dtype=torch.float32)
trainY_tensor = torch.tensor(trainY.values, dtype=torch.float32).unsqueeze(1)

valX_tensor = torch.tensor(valX.astype('float32').values)
valY_tensor = torch.tensor(valY.values, dtype=torch.float32).unsqueeze(1)

testX_tensor = torch.tensor(testX.values, dtype=torch.float32)
testY_tensor = torch.tensor(testY.values, dtype=torch.float32).unsqueeze(1)

trainDataset = TensorDataset(trainX_tensor, trainY_tensor)
trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)

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

#Training the model
epochs = 50
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in trainLoader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        valOutput = model(valX_tensor)
        valLoss = criterion(valOutput, valY_tensor)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train_Loss: {loss.item():.4f}, Val Loss: {valLoss.item():.4f}")

model.eval()
with torch.no_grad():
    predictions = model(testX_tensor)
    predictions = predictions.round()


# Results
testAccuracy = accuracy_score(testY_tensor, predictions)
conMatrix = confusion_matrix(testY_tensor, predictions)

print(f"Test Accuracy: {testAccuracy:.4f}")
print("Confusion Matrix:")
print(conMatrix)