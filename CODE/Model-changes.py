import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('../heart.csv')

# Check for nulls
#print("Missing values per column:")
#print(data.isnull().sum())

# Drop rows with any nulls (only if there's a small amount)
data_clean = data.dropna()

# Drop duplicates
data_clean = data_clean.drop_duplicates()

# encoding columns 
data_encoded = pd.get_dummies(data_clean, columns=[
    'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'
])

# Normalize numeric columns
scaler = StandardScaler()
numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
data_encoded[numerical_columns] = scaler.fit_transform(data_encoded[numerical_columns])

# Separate input and output
X = data_encoded.drop('HeartDisease', axis=1)
y = data_encoded['HeartDisease']

# Check resulting shape
#print("Original shape:", data.shape)
#print("Cleaned shape:", data_clean.shape)

#print(X.dtypes)

# Train-test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)

# all boolean values are converted
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

X_val_tensor = torch.tensor(X_val.astype('float32').values)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Wrap in DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Feedforward steps using relu and sigmoid at the output for binary
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

model = HeartNet(X_train.shape[1])

criterion = nn.BCELoss()  # Binary cross-entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
val_losses = []

# Train the model
epochs = 60
for epoch in range(epochs):
    epoch_train_loss = 0.0
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)  # Log last batch loss
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses.append(val_loss.item())
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train_Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# predictions
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predictions = predictions.round()  # Convert probs to 0/1

test_accuracy = accuracy_score(y_test_tensor, predictions)
con_matrix = confusion_matrix(y_test_tensor, predictions)

print(f"Test Accuracy: {test_accuracy:.4f}")
print("Confusion Matrix:")
print(con_matrix)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(f"loss_batch{train_loader.batch_size}_epoch{epochs}.png")
plt.show()
