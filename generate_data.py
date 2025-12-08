import pandas as pd
import random

# ==========================================
# 1. Solvent Database (Source: PDF Handout)
# ==========================================
# Note: Non-polar solvents are EXCLUDED.
# Only Protic and Polar Aprotic solvents are included.
SOLVENT_DATABASE = {
    # --- Protic Solvents (Favor Sn1/E1) ---
    'Water': (78.5, 'Protic'), 'Heavy water': (77.9, 'Protic'),
    'Ethylene glycol': (37.7, 'Protic'), 'Methanol': (32.6, 'Protic'),
    'Diethylene glycol': (31.8, 'Protic'), 'Ethanol': (24.6, 'Protic'),
    '1-Propanol': (20.1, 'Protic'), '2-Propanol (IPA)': (18.3, 'Protic'),
    '1-Butanol': (17.8, 'Protic'), '2-Butanol': (17.3, 'Protic'),
    't-Butanol': (12.5, 'Protic'), 'Acetic acid': (6.2, 'Protic'),

    # --- Polar Aprotic Solvents (Favor Sn2) ---
    'DMSO': (47.0, 'Aprotic'), 'DMF': (38.3, 'Aprotic'),
    'Acetonitrile': (36.6, 'Aprotic'), 'NMP': (32.0, 'Aprotic'),
    'Acetone': (21.0, 'Aprotic'), '2-Butanone (MEK)': (18.6, 'Aprotic'),
    'Pyridine': (12.3, 'Aprotic'), '1,2-Dichloroethane': (10.4, 'Aprotic'),
    'Methylene chloride': (9.1, 'Aprotic'), 'THF': (7.5, 'Aprotic'),
    'Diglyme': (7.2, 'Aprotic'), 'Ethyl acetate': (6.0, 'Aprotic'),
    'Chlorobenzene': (5.7, 'Aprotic'), 'Chloroform': (4.8, 'Aprotic'),
    'Diethyl ether': (4.3, 'Aprotic')
}

# ==========================================
# 2. Simulation Configuration
# ==========================================
NUM_SAMPLES = 10000             # Number of synthetic samples
SUBSTRATES = [1, 2, 3]          # 1°, 2°, 3° substrates
TEMP_MIN, TEMP_MAX = 273.0, 373.0   # Temperature range (K)
PKA_MIN, PKA_MAX = -5.0, 25.0       # Base pKa range
NOISE_RATE = 0.05  # 5% of data will be noisy (simulating experimental error)

# ==========================================
# 3. Chemical Logic Function
# ==========================================
def determine_reaction(sub_degree, pka, temp, solv_dielectric, solv_type, steric_hindrance):
    # [Case 1] Tertiary substrate
    if sub_degree == 3:
        if pka > 11: return 'E2' 
        else: 
            if temp > 333: return 'E1'
            return 'Sn1'

    # [Case 2] Primary substrate
    if sub_degree == 1:
        if steric_hindrance == 1: return 'E2' 
        return 'Sn2'

    # [Case 3] Secondary substrate
    if sub_degree == 2:
        if pka > 11: return 'E2' 
        else:
            if solv_type == 'Aprotic':
                if temp > 330: return 'E2' 
                return 'Sn2'
            else:
                if temp > 340: return 'E1'
                return 'Sn1'
    return 'Sn1'

# ==========================================
# 4. Data Generation Loop (With Noise)
# ==========================================
data = []
print(f"Generating Synthetic Data with {NOISE_RATE*100}% Noise...")

for _ in range(NUM_SAMPLES):
    sub = random.choice(SUBSTRATES)
    pka = round(random.uniform(PKA_MIN, PKA_MAX), 1)
    temp = round(random.uniform(TEMP_MIN, TEMP_MAX), 1)
    steric = random.choice([0, 1])  # 0: low steric hindrance, 1: high steric hindrance
    
    solv_name = random.choice(list(SOLVENT_DATABASE.keys()))
    solv_dielectric, solv_type = SOLVENT_DATABASE[solv_name]
    solv_type_num = 0 if solv_type == 'Protic' else 1
    
    # 1. Apply logical rule to determine the reaction type
    reaction = determine_reaction(sub, pka, temp, solv_dielectric, solv_type, steric)
    
    # 2. Inject noise (simulating real-world mislabeling / experimental error)
    # With 5% probability, replace the correct label with a random reaction
    if random.random() < NOISE_RATE:
        reaction = random.choice(['Sn1', 'Sn2', 'E1', 'E2'])
    
    data.append([sub, pka, steric, temp, solv_dielectric, solv_type_num, reaction, solv_name])

# ==========================================
# 5. Export
# ==========================================
df = pd.DataFrame(
    data,
    columns=[
        'Substrate_Degree', 'Base_pKa', 'Steric_Hindrance', 
        'Temperature', 'Solvent_Dielectric', 'Solvent_Type_Num',
        'Reaction_Label', 'Solvent_Name'
    ]
)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('organic_reaction_data_B_final.csv', index=False)

print(f"Done! Created 'organic_reaction_data_B_final.csv' with {len(df)} samples.")
print("-" * 30)
print("Class Distribution:")
print(df['Reaction_Label'].value_counts())
