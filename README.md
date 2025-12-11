# Organic Reaction Pathway Prediction Project

## 1. Project Overview

This project aims to develop an AI model that predicts the major reaction pathway (SN1, SN2, E1, E2) of alkyl halides.  
Instead of relying on image based molecular structures, this project utilizes physicochemical feature vectors (for example, pKa, temperature, steric hindrance, solvent properties) to train machine learning models.

The goal is to demonstrate a **Sim-to-Real** AI approach:

- Train the model on a large scale synthetic dataset based on chemical logic  
- Verify it against real-world textbook examples  

### Key Features

#### Rule Based Synthetic Data
- Generates 10,000 samples based on McMurry and Wade textbook logic (simulating chemical rules).

#### Realistic Noise Injection
- Adds 5 percent random noise to simulate experimental errors and prevent overfitting.

#### Solvent Database Integration
- Uses real dielectric constants from standard chemical data  
  (for example, Water = 78.5, Acetone = 21.0).

#### Dual Model Comparison
- Compares a Random Forest (ML) baseline and a PyTorch MLP (DL) model.

#### Advanced Logic Handling
- Correctly handles complex scenarios such as the Finkelstein reaction  
  (Secondary substrate + weak base + aprotic solvent → SN2).

---

## 2. Requirements

This project requires a dedicated Conda environment for reproducibility.

### Setup Instructions
1.  **Create Environment:** Create the environment using the provided YAML file.
    ```bash
    conda env create -f environment.yaml
    ```
2.  **Activate Environment:**
    ```bash
    conda activate classifier_env
    ```
    *Note: This command handles the installation of Python, PyTorch, and all dependencies automatically.*

## 3. How to Run (Step by Step)
### 3.1 Step 1: Generate Synthetic Data

Generates 10,000 synthetic reaction samples based on physicochemical rules.

```bash
python generate_data.py
```

Output

organic_reaction_data_B_final.csv

Note

This script includes logic for solvent effects (protic vs aprotic) and temperature dependence.

### 3.2 Step 2: Train Models (Experiment)

Train two different models to compare performance.

#### 3.2.1 Train Random Forest (Machine Learning Baseline)

Performs hyperparameter tuning (for example, n_estimators) and analyzes feature importance.

```bash
python train_rf.py
```

Output

rf_model.pkl (saved Random Forest model)

#### 3.2.2 Train MLP (Deep Learning Model)

Trains a 3 layer neural network using PyTorch.

```bash
python train_mlp.py
```

Output

reaction_model.pth (model weights)

scaler.pkl (feature scaler)

encoder.pkl (label encoder)

### 3.3 Step 3: Validation (Textbook Examples)

Tests both models against 5 real-world scenarios from organic chemistry textbooks
(for example, Finkelstein reaction, solvolysis of t BuCl).

```bash
python test_demo_all.py
```

### 3.4 Step 4: Manual Prediction (Demo)

Interactive tool where users can input chemical conditions to get real-time predictions.

```bash
python predict_manual.py
```

## 4. Methodology and Models
### 4.1 Data Construction Strategy

Instead of crawling raw experimental data, a synthetic dataset was constructed based on chemical logic.

**Input Features (6 dimensions)**

- Substrate degree: 1, 2, 3  
- Base pKa: -5.0 to 25.0 (continuous)  
- Steric hindrance: 0 (normal) or 1 (bulky)  
- Temperature: 273 K to 373 K  
- Solvent dielectric constant: real values from class PDF handouts  
- Solvent type: 0 (protic) or 1 (aprotic)

A 5 percent label noise is injected to simulate experimental errors and prevent overfitting to the rule set.

### 4.2 Model Architectures

| Model        | Type          | Configuration                                                   | Purpose                                                                 |
|-------------|---------------|-----------------------------------------------------------------|-------------------------------------------------------------------------|
| Random Forest | ML (Ensemble) | n_estimators tuning (for example, 10, 50, 100)                 | Establish a strong baseline and analyze feature importance on tabular data. |
| PyTorch MLP | Deep Learning | Input(6) → FC(64) → BN → ReLU → FC(32) → BN → ReLU → Output(4) | Capture non linear relationships (for example, solvent base interaction, mixed effects). |


## 5. Experimental Results
### 5.1 Performance Summary

| Model         | Accuracy (Validation) | F1 Score (Macro) | Note                                      |
|--------------|-----------------------|------------------|-------------------------------------------|
| Random Forest | ~97.2%               | ~0.96            | High accuracy on structured tabular data. |
| PyTorch MLP  | ~95.7%               | ~0.94            | Robust performance, no overfitting observed. |


**Analysis**

Even with 5 percent noise injection, both models successfully learned the underlying chemical rules.

The Random Forest model showed slightly higher accuracy due to the tabular nature of the data.

The MLP demonstrated strong generalization on unseen, real-world textbook examples.

### 5.2 Feature Importance (Random Forest)

Random Forest feature importance analysis revealed that:

Substrate degree and base pKa are the most critical factors in determining the reaction pathway.

This aligns well with standard organic chemistry theory for substitution and elimination reactions.

## 6. References

- McMurry, J. E. (2015). *Organic Chemistry* (9th ed.). Cengage Learning.  
- Bordwell pKa Table.  
- Class handout: Common organic solvents and their dielectric constants.

## 7. File Structure

- generate_data.py  – synthetic dataset generator (rule-based logic + noise)  
- train_rf.py       – Random Forest training script  
- train_mlp.py      – MLP training script (PyTorch)  
- test_demo_all.py  – textbook-based validation script  
- predict_manual.py – CLI demo for manual prediction  