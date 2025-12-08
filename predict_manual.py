import torch
import torch.nn as nn
import numpy as np
import joblib
import sys

# ==========================================
# 1. Configuration & Model Definition
# ==========================================
# The model architecture must be identical to the one used during training.
class ReactionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ReactionPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    def forward(self, x): 
        return self.net(x)

# User-friendly solvent database (name -> dielectric constant, type encoded)
SOLVENT_DB = {
    'water': (78.5, 0), 'methanol': (32.6, 0), 'ethanol': (24.6, 0), 
    'acetone': (21.0, 1), 'dmso': (47.0, 1), 'dmf': (38.3, 1),
    'acetonitrile': (36.6, 1), 'thf': (7.5, 1),
    # Hexane and others were not in the training set, but we allow manual input anyway
    'hexane': (1.9, 1), 
    'ether': (4.3, 1), 'mc': (9.1, 1), 'ea': (6.0, 1)
}

def get_user_input():
    print("\n" + "="*50)
    print("🧪 Enter Chemical Conditions for Prediction")
    print("="*50)
    
    try:
        # 1. Substrate degree
        sub = int(input("1. Substrate Degree (1, 2, 3): "))
        if sub not in [1, 2, 3]:
            raise ValueError("Must be 1, 2, or 3")
        
        # 2. Base/Nucleophile pKa
        pka = float(input("2. Base/Nucleophile pKa (e.g., -9 for I-, 15.7 for OH-): "))
        
        # 3. Steric hindrance (bulky base?)
        steric_input = input("3. Is the Base Bulky? (y/n): ").lower()
        steric = 1 if steric_input == 'y' else 0
        
        # 4. Temperature
        temp = float(input("4. Temperature (Kelvin) (e.g., 298 for RT, 350 for Heat): "))
        
        # 5. Solvent
        print(f"   (Supported Solvents: {', '.join(list(SOLVENT_DB.keys())[:5])}...)")
        solv_name = input("5. Solvent Name (e.g., Acetone, Water): ").lower()
        
        if solv_name in SOLVENT_DB:
            dielectric, solv_type = SOLVENT_DB[solv_name]
            print(f"   -> Detected: {solv_name.capitalize()} (Dielectric: {dielectric}, Type: {'Aprotic' if solv_type==1 else 'Protic'})")
        else:
            print("   -> Unknown solvent. Enter manual values.")
            dielectric = float(input("   -> Manual Dielectric Constant: "))
            solv_type = int(input("   -> Solvent Type (0: Protic, 1: Aprotic): "))

        return [sub, pka, steric, temp, dielectric, solv_type]

    except ValueError as e:
        print(f"❌ Invalid Input: {e}")
        return None

def main():
    # ---------------------------------------------------------
    # 1. Load models & preprocessing tools
    # ---------------------------------------------------------
    print("[System] Loading Models...")
    try:
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        classes = encoder.classes_
        
        # Load Random Forest model
        rf_model = joblib.load('rf_model.pkl')
        
        # Load MLP model
        mlp_model = ReactionPredictor(6, 64, 4)
        mlp_model.load_state_dict(torch.load('reaction_model.pth'))
        mlp_model.eval()
        
    except FileNotFoundError:
        print("❌ Error: Model files not found. Run training first.")
        return

    # ---------------------------------------------------------
    # 2. Interactive prediction loop
    # ---------------------------------------------------------
    while True:
        features = get_user_input()
        # If input failed, ask again
        if features is None:
            continue

        # Preprocessing (Scaling)
        features_arr = np.array([features])
        features_scaled = scaler.transform(features_arr)
        
        # --- Prediction ---
        # 1. Random Forest prediction
        rf_pred_idx = rf_model.predict(features_scaled)[0]
        rf_pred = classes[rf_pred_idx]
        
        # 2. MLP prediction
        inputs = torch.tensor(features_scaled, dtype=torch.float32)
        with torch.no_grad():
            outputs = mlp_model(inputs)
            # Also compute probability distribution with softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            mlp_pred_idx = torch.argmax(probs).item()
            mlp_pred = classes[mlp_pred_idx]
            confidence = probs[mlp_pred_idx].item() * 100

        # --- Output ---
        print("\n" + "-"*30)
        print("🤖 AI Prediction Result")
        print("-"*30)
        print(f"🌲 Random Forest says:  [{rf_pred}]")
        print(f"🧠 PyTorch MLP says:    [{mlp_pred}] (Confidence: {confidence:.1f}%)")
        print("-"*30)
        
        if rf_pred != mlp_pred:
            print("⚠️ The models disagree! This is an ambiguous case.")
        
        # Ask whether to continue
        cont = input("\nTry another? (y/n): ").lower()
        if cont != 'y':
            print("Bye! 👋")
            break

if __name__ == '__main__':
    main()
