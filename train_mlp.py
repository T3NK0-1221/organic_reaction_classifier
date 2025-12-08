import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import joblib

# ==========================================
# 1. Settings (Configuration)
# ==========================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
FILE_PATH = 'organic_reaction_data_B_final.csv'

# Hyperparameters
INPUT_DIM = 6       # Number of input features
HIDDEN_DIM = 64     # Number of hidden units
OUTPUT_DIM = 4      # Number of output classes (Sn1, Sn2, E1, E2)
LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 32

# ==========================================
# 2. Model and Dataset Definitions
# ==========================================
class ReactionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ReactionPredictor, self).__init__()
        # 3-layer MLP architecture (Input -> 64 -> 32 -> Output)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Stabilize training
            nn.ReLU(),                   # Add non-linearity
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class ReactionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 3. Main Execution Function
# ==========================================
def main():
    print("[Deep Learning] PyTorch MLP Training Start...")
    
    # 1. Load data
    df = pd.read_csv(FILE_PATH)
    
    # Select input features to use for training
    feature_cols = ['Substrate_Degree', 'Base_pKa', 'Steric_Hindrance', 
                    'Temperature', 'Solvent_Dielectric', 'Solvent_Type_Num']
    X = df[feature_cols].values
    
    # Encode target labels (string -> integer)
    le = LabelEncoder()
    y = le.fit_transform(df['Reaction_Label'])
    
    # **Important**: Save the encoder (used later in test_demo.py to decode predictions)
    joblib.dump(le, 'encoder.pkl')
    print(f"Classes: {le.classes_}")

    # 2. Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 3. Scaling (preprocessing)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # **Important**: Save the scaler (used later in test_demo.py to transform new inputs)
    joblib.dump(scaler, 'scaler.pkl')

    # 4. Create DataLoaders
    train_ds = ReactionDataset(X_train_scaled, y_train)
    test_ds = ReactionDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Initialize model, loss, and optimizer
    model = ReactionPredictor(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training loop
    print("-" * 50)
    for epoch in range(EPOCHS):
        model.train()  # Training mode
        running_loss = 0.0
        
        for xb, yb in train_loader:
            optimizer.zero_grad()           # Reset gradients
            outputs = model(xb)             # Forward pass
            loss = criterion(outputs, yb)   # Compute loss
            loss.backward()                 # Backpropagation
            optimizer.step()                # Update weights
            running_loss += loss.item()
        
        # Print log every 10 epochs
        if (epoch+1) % 10 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # 7. Final evaluation
    model.eval()  # Evaluation mode
    all_preds = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            outputs = model(xb)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
    
    # Compute metrics
    acc = accuracy_score(y_test, all_preds)
    f1 = f1_score(y_test, all_preds, average='macro')
    
    print("-" * 50)
    print(f"MLP Final Accuracy: {acc:.4f}")
    print(f"MLP Final F1-Score: {f1:.4f}")
    
    # 8. Save model
    torch.save(model.state_dict(), 'reaction_model.pth')
    print("-" * 50)
    print("Training Complete!")
    print("Files Saved: 'reaction_model.pth', 'scaler.pkl', 'encoder.pkl'")

if __name__ == '__main__':
    main()
