# AI-Based Organic Reaction Classifier ($S_N1, S_N2, E1, E2$)
### EF2039 Term Project 02

---

## 1. Project Overview

This project aims to develop an AI model that predicts the major reaction pathway (**$S_N1, S_N2, E1, E2$**) of alkyl halides.  
Instead of relying on image-based molecular structures, this project utilizes **physicochemical feature vectors** (e.g., $pK_a$, Temperature, Steric Hindrance, Solvent Properties) to train machine learning models.

The goal is to demonstrate a **"Sim-to-Real"** AI approach:

- Train the model on a large-scale **synthetic dataset based on chemical logic**
- Verify it against **real-world textbook examples**

### Key Features

- **Rule-Based Synthetic Data**  
  Generates 10,000 samples based on McMurry/Wade textbook logic (simulating chemical rules).

- **Realistic Noise Injection**  
  Adds 5% random noise to simulate experimental errors and prevent overfitting.

- **Solvent Database Integration**  
  Uses real dielectric constants from standard chemical data  
  (e.g., Water = 78.5, Acetone = 21.0).

- **Dual Model Comparison**  
  Compares a **Random Forest (ML)** baseline and a **PyTorch MLP (DL)** model.

- **Advanced Logic Handling**  
  Correctly handles complex scenarios such as the **Finkelstein Reaction**  
  (Secondary substrate + Weak base + Aprotic solvent → $S_N2$).

---

## 2. Requirements

To run this project, you need the following libraries:

- Python >= 3.8
- `pandas`
- `torch` (PyTorch)
- `scikit-learn`
- `numpy`
- `joblib`

Install dependencies using:

```bash
pip install -r requirements.txt
3. How to Run (Step-by-Step)
3.1 Step 1: Generate Synthetic Data
Generates 10,000 synthetic reaction samples based on physicochemical rules.

bash
Copy code
python generate_data.py
Output:

organic_reaction_data_B_final.csv

Note:
This script includes logic for solvent effects (Protic vs Aprotic) and temperature dependence.

3.2 Step 2: Train Models (Experiment)
Train two different models to compare performance.

3.2.1 Train Random Forest (Machine Learning Baseline)
Performs hyperparameter tuning (e.g., n_estimators) and analyzes feature importance.

bash
Copy code
python train_rf.py
Output:

rf_model.pkl (saved Random Forest model)

3.2.2 Train MLP (Deep Learning Model)
Trains a 3-layer neural network using PyTorch.

bash
Copy code
python train_mlp.py
Output:

reaction_model.pth (model weights)

scaler.pkl (feature scaler)

encoder.pkl (label encoder)

3.3 Step 3: Validation (Textbook Examples)
Tests both models against 5 real-world scenarios from organic chemistry textbooks
(e.g., Finkelstein reaction, solvolysis of t-BuCl).

bash
Copy code
python test_demo_all.py

3.4 Step 4: Manual Prediction (Demo)
An interactive tool where users can input chemical conditions to get real-time predictions.

bash
Copy code
python predict_manual.py
4. Methodology & Models
4.1 Data Construction Strategy
Instead of crawling raw experimental data, a synthetic dataset was constructed based on chemical logic.

Input Features (6 dimensions):

Substrate Degree: 1, 2, 3

Base $pK_a$: −5.0 to 25.0 (continuous)

Steric Hindrance: 0 (normal) / 1 (bulky)

Temperature: 273 K to 373 K

Solvent Dielectric Constant: Real values from class PDF handouts

Solvent Type: 0 (Protic) / 1 (Aprotic)

Logic Source:

Organic Chemistry (McMurry, 9th Ed), Chapter 11

The label ($S_N1, S_N2, E1, E2$) is determined by a rule-based function that encodes:

Substrate degree

Base strength ($pK_a$)

Steric hindrance

Solvent type (Protic vs Aprotic)

Temperature effects

A 5% label noise is injected to simulate experimental errors and prevent overfitting to the rule set.

4.2 Model Architectures
Model	Type	Configuration	Purpose
Random Forest	ML (Ensemble)	n_estimators tuning (e.g., 10, 50, 100)	Establish a strong baseline and analyze feature importance on tabular data.
PyTorch MLP	Deep Learning	Input(6) → FC(64) → BN → ReLU → FC(32) → BN → ReLU → Output(4)	Capture non-linear relationships (e.g., solvent–base interaction, mixed effects).

5. Experimental Results
5.1 Performance Summary
Model	Accuracy (Validation)	F1-Score (Macro)	Note
Random Forest	~97.2%	~0.96	High accuracy on structured tabular data.
PyTorch MLP	~95.7%	~0.94	Robust performance; no overfitting observed.

Analysis:

Even with 5% noise injection, both models successfully learned the underlying chemical rules.

The Random Forest model showed slightly higher accuracy due to the tabular nature of the data.

The MLP demonstrated strong generalization on unseen, real-world textbook examples.

5.2 Feature Importance (Random Forest)
Random Forest feature importance analysis revealed that:

Substrate Degree and Base $pK_a$ are the most critical factors in determining the reaction pathway.

This aligns well with standard organic chemistry theory for substitution and elimination reactions.

6. File Structure
text
Copy code
📂 Root
 ├── 📄 requirements.txt         # Library dependencies
 ├── 🐍 generate_data.py         # Data generator (rule-based logic + noise)
 ├── 🐍 train_rf.py              # Random Forest training script
 ├── 🐍 train_mlp.py             # PyTorch MLP training script
 ├── 🐍 test_demo_all.py         # Final verification using textbook examples
 ├── 🐍 predict_manual.py        # Interactive CLI demo for manual prediction
 └── 📄 README.md                # Project documentation

7. References

McMurry, J. E. (2015). Organic Chemistry (9th ed.). Cengage Learning.

Bordwell $pK_a$ Table.

Class Handout: Common Organic Solvents and Their Dielectric Constants.