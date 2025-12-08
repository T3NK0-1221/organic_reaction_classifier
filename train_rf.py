import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

# preset
SEED = 42
np.random.seed(SEED)
FILE_PATH = 'organic_reaction_data_B_final.csv'

def main():
    print("[Machine Learning] Random Forest Experiment Start...")
    
    # 1. data_lodaing
    df = pd.read_csv(FILE_PATH)
    feature_cols = ['Substrate_Degree', 'Base_pKa', 'Steric_Hindrance', 
                    'Temperature', 'Solvent_Dielectric', 'Solvent_Type_Num']
    X = df[feature_cols].values
    y = LabelEncoder().fit_transform(df['Reaction_Label'])

    # 2. spliting and scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Hyperparameter Experiments
    n_estimators_list = [10, 50, 100]
    
    print("-" * 50)
    print(f"{'n_estimators':<15} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-" * 50)

    best_acc = 0
    best_model = None

    for n in n_estimators_list:
        rf = RandomForestClassifier(n_estimators=n, random_state=SEED)
        rf.fit(X_train_scaled, y_train)
        
        preds = rf.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        
        print(f"{n:<15} | {acc:.4f}     | {f1:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = rf

    print("-" * 50)
    print(f"Best Random Forest Accuracy: {best_acc:.4f}")
    
    # 4. Feature Importance
    print("\n[Analysis] Feature Importance:")
    for name, score in zip(feature_cols, best_model.feature_importances_):
        print(f" - {name}: {score:.4f}")

    # model file save
    joblib.dump(best_model, 'rf_model.pkl')
    print("\nSaved best RF model to 'rf_model.pkl'")

if __name__ == '__main__':
    main()