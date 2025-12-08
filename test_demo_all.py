import torch
import torch.nn as nn
import numpy as np
import joblib

# ==========================================
# 1. MLP 모델 구조 정의
# ==========================================
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
    def forward(self, x): return self.net(x)

def main():
    print("[Final Test] Comparing Random Forest vs PyTorch MLP on REAL Textbook Examples...")
    print("-" * 105)

    # 1. 로드
    try:
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        classes = encoder.classes_ 
        rf_model = joblib.load('rf_model.pkl')
        mlp_model = ReactionPredictor(6, 64, 4)
        mlp_model.load_state_dict(torch.load('reaction_model.pth'))
        mlp_model.eval()
        print("✅ Models loaded. Starting evaluation on real-world reactions...")
    except FileNotFoundError:
        print("❌ Error: Missing model files. Run training scripts first.")
        return

    # 2. 실제 교과서 등재 반응 예제 (Real-world Examples)
    # 아래 예제들은 McMurry 및 Wade 유기화학 교과서의 Chapter 11(Alkyl Halide)에 나오는 실제 반응입니다.
    
    examples = [
        {
            # Case 1: 1차 기질의 전형적인 치환
            # 반응식: CH3CH2CH2CH2Br + NaI -> CH3CH2CH2CH2I + NaBr (in Acetone)
            "desc": "1. [Sn2] 1-Bromobutane + NaI (Finkelstein Reaction)",
            "feat": [1, -9.0, 0, 298, 21.0, 1], # 1차, I-(pKa -9), 입체X, 상온, Acetone(Aprotic)
            "ans": "Sn2"
        },
        {
            # Case 2: 3차 기질의 가수분해 (Solvolysis)
            # 반응식: (CH3)3CCl + H2O --(heat)--> (CH3)2C=CH2 + HCl
            # 설명: 물은 약염기이자 Protic 용매. 가열하면(Heat) 제거반응(E1)이 우세해짐.
            "desc": "2. [E1] t-Butyl Chloride + H2O + Heat (Solvolysis)",
            "feat": [3, -1.74, 0, 350, 78.5, 0], # 3차, H2O(pKa -1.74), 입체X, 350K, Water(Protic)
            "ans": "E1"
        },
        {
            # Case 3: 2차 기질과 강염기의 제거 반응 (Zaitsev Rule)
            # 반응식: (CH3)2CHBr + NaOEt -> CH3CH=CH2 + EtOH
            # 설명: 2차 기질에 강염기(Ethoxide)를 쓰면 입체장애로 E2가 우세함.
            "desc": "3. [E2] 2-Bromopropane + NaOEt (Williamson Ether Fail)",
            "feat": [2, 16.0, 0, 320, 24.6, 0], # 2차, EtO-(pKa 16), 입체X, 상온, Ethanol(Protic)
            "ans": "E2"
        },
        {
            # Case 4: 3차 기질과 강염기
            # 설명: 3차는 Sn2 불가. 강염기(OH-)가 오면 Sn1 기다릴 시간 없이 바로 수소 뜯음(E2).
            "desc": "4. [E2] t-Butyl Bromide + NaOH (Strong Base E2)",
            "feat": [3, 15.7, 0, 298, 78.5, 0], # 3차, OH-(pKa 15.7), 입체X, 상온, Water
            "ans": "E2"
        },
        {
            # Case 5: 2차 기질에서의 핀켈슈타인 반응 (핵심 예제!)
            # 설명: 2차 기질이라 경쟁이 치열하지만, Aprotic 용매(Acetone)가 I-의 친핵성을 극대화시켜 Sn2로 유도함.
            "desc": "5. [Sn2] 2-Chlorobutane + NaI + Acetone (Secondary Finkelstein)",
            "feat": [2, -9.0, 0, 298, 21.0, 1], # 2차, I-(pKa -9), 입체X, 상온, Acetone(Aprotic)
            "ans": "Sn2"
        }
    ]

    # 3. 비교 출력
    print("\n" + "="*110)
    print(f"{'Real-world Scenario':<55} | {'Ans':<5} | {'RF Pred':<8} | {'MLP Pred':<8} | {'Result'}")
    print("="*110)

    rf_score = 0
    mlp_score = 0

    for ex in examples:
        raw_features = np.array([ex['feat']])
        scaled_features = scaler.transform(raw_features)
        
        # RF Prediction
        rf_pred = classes[rf_model.predict(scaled_features)[0]]
        
        # MLP Prediction
        inputs = torch.tensor(scaled_features, dtype=torch.float32)
        with torch.no_grad():
            output = mlp_model(inputs)
            mlp_pred = classes[torch.max(output, 1)[1].item()]
        
        # Check
        rf_ok = (rf_pred == ex['ans'])
        mlp_ok = (mlp_pred == ex['ans'])
        if rf_ok: rf_score += 1
        if mlp_ok: mlp_score += 1
        
        mark = "✅ Both Correct" if (rf_ok and mlp_ok) else "❌ Mismatch"
        print(f"{ex['desc']:<55} | {ex['ans']:<5} | {rf_pred:<8} | {mlp_pred:<8} | {mark}")

    print("-" * 110)
    print(f"Final Score -> Random Forest: {rf_score}/5  |  PyTorch MLP: {mlp_score}/5")
    print("="*110)

if __name__ == '__main__':
    main()