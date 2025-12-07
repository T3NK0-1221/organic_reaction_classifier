import pandas as pd
import random

# ==========================================
# 1. Solvent Database (Source: PDF Handout)
# ==========================================
# Note: Non-polar solvents are EXCLUDED.
# Only Protic and Polar Aprotic solvents are included to ensure reaction feasibility.
SOLVENT_DATABASE = {
    # --- Protic Solvents (Favor Sn1/E1 via H-bonding) ---
    'Water': (78.5, 'Protic'), 
    'Heavy water': (77.9, 'Protic'),
    'Ethylene glycol': (37.7, 'Protic'), 
    'Methanol': (32.6, 'Protic'),
    'Diethylene glycol': (31.8, 'Protic'), 
    'Ethanol': (24.6, 'Protic'),
    '1-Propanol': (20.1, 'Protic'), 
    '2-Propanol (IPA)': (18.3, 'Protic'),
    '1-Butanol': (17.8, 'Protic'), 
    '2-Butanol': (17.3, 'Protic'),
    't-Butanol': (12.5, 'Protic'), 
    'Acetic acid': (6.2, 'Protic'),

    # --- Polar Aprotic Solvents (Favor Sn2 via Naked Anion effect) ---
    'DMSO': (47.0, 'Aprotic'), 
    'DMF': (38.3, 'Aprotic'),
    'Acetonitrile': (36.6, 'Aprotic'), 
    'NMP': (32.0, 'Aprotic'),
    'Acetone': (21.0, 'Aprotic'), 
    '2-Butanone (MEK)': (18.6, 'Aprotic'),
    'Pyridine': (12.3, 'Aprotic'), 
    '1,2-Dichloroethane': (10.4, 'Aprotic'),
    'Methylene chloride': (9.1, 'Aprotic'), 
    'THF': (7.5, 'Aprotic'),
    'Diglyme': (7.2, 'Aprotic'), 
    'Ethyl acetate': (6.0, 'Aprotic'),
    'Chlorobenzene': (5.7, 'Aprotic'), 
    'Chloroform': (4.8, 'Aprotic'),
    'Diethyl ether': (4.3, 'Aprotic')
}

# ==========================================
# 2. Simulation Configuration
# ==========================================
NUM_SAMPLES = 10000            # Total number of synthetic data points
SUBSTRATES = [1, 2, 3]         # Substrate degrees: Primary(1), Secondary(2), Tertiary(3)
TEMP_MIN, TEMP_MAX = 273.0, 373.0  # Temperature range in Kelvin (0°C ~ 100°C)
PKA_MIN, PKA_MAX = -5.0, 25.0      # pKa range: Strong Acid Conjugate ~ Weak Acid Conjugate

# ==========================================
# 3. Chemical Logic Function (Rule-based)
# ==========================================
def determine_reaction(sub_degree, pka, temp, solv_dielectric, solv_type, steric_hindrance):
    """
    Determines the major reaction pathway based on McMurry/Wade organic chemistry rules.
    Inputs:
        - sub_degree: Degree of substrate (1, 2, 3)
        - pka: Basicity of the reagent
        - temp: Temperature in Kelvin
        - solv_type: 'Protic' or 'Aprotic'
        - steric_hindrance: 0 (Normal) or 1 (Bulky Base)
    Returns:
        - Reaction Type: 'Sn1', 'Sn2', 'E1', 'E2'
    """

    # [Case 1] Tertiary Substrate (3°)
    if sub_degree == 3:
        # Sn2 is impossible due to steric hindrance at the reaction center.
        if pka > 11: 
            return 'E2' # Strong Base -> E2 dominates
        else: 
            # Weak Base -> Solvolysis (Competition between Sn1 and E1)
            # High temperature (> 60°C/333K) favors Elimination due to entropy (ΔS > 0).
            if temp > 333: return 'E1'
            return 'Sn1' # Usually Sn1 at lower temperatures

    # [Case 2] Primary Substrate (1°)
    if sub_degree == 1:
        # If the base is bulky (Steric=1), it cannot access the carbon, favoring Elimination.
        if steric_hindrance == 1: return 'E2' 
        # Otherwise, Sn2 is dominant due to low steric hindrance.
        return 'Sn2'

    # [Case 3] Secondary Substrate (2°) - Refined Logic
    if sub_degree == 2:
        # 3-1. Strong Base (pKa > 11)
        # E2 is dominant over Sn2 due to steric hindrance in secondary substrates.
        if pka > 11: 
            return 'E2' 
        else:
            # 3-2. Weak Base / Good Nucleophile (pKa <= 11)
            # Aprotic Solvents: "Naked Anion" effect increases nucleophilicity -> Favors Sn2.
            # (e.g., Finkelstein reaction using NaI in Acetone)
            if solv_type == 'Aprotic':
                # High temp might induce E2, but Sn2 is generally preferred in Aprotic conditions.
                if temp > 330: return 'E2' 
                return 'Sn2'
            
            # Protic Solvents: Solvation shell reduces nucleophilicity -> Favors Sn1/E1.
            else:
                if temp > 340: return 'E1'
                return 'Sn1'

    return 'Sn1' # Fallback safety

# ==========================================
# 4. Data Generation Loop
# ==========================================
data = []
print("Generating Synthetic Data...")

for _ in range(NUM_SAMPLES):
    # --- Random Input Generation ---
    sub = random.choice(SUBSTRATES)
    pka = round(random.uniform(PKA_MIN, PKA_MAX), 1)
    temp = round(random.uniform(TEMP_MIN, TEMP_MAX), 1)
    steric = random.choice([0, 1]) # 0: Normal, 1: Bulky Base
    
    # --- Solvent Selection ---
    solv_name = random.choice(list(SOLVENT_DATABASE.keys()))
    solv_dielectric, solv_type = SOLVENT_DATABASE[solv_name]
    
    # Encode Solvent Type: 0 for Protic, 1 for Aprotic
    solv_type_num = 0 if solv_type == 'Protic' else 1
    
    # --- Logic Application (Labeling) ---
    reaction = determine_reaction(sub, pka, temp, solv_dielectric, solv_type, steric)
    
    # Append to dataset
    data.append([sub, pka, steric, temp, solv_dielectric, solv_type_num, reaction, solv_name])

# ==========================================
# 5. Export to CSV
# ==========================================
df = pd.DataFrame(data, columns=['Substrate_Degree', 'Base_pKa', 'Steric_Hindrance', 
                                 'Temperature', 'Solvent_Dielectric', 'Solvent_Type_Num',
                                 'Reaction_Label', 'Solvent_Name'])

# Shuffle data to prevent any order bias during training
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV file
file_name = 'organic_reaction_data_B_final.csv'
df.to_csv(file_name, index=False)

print(f"Done! Created '{file_name}' with {len(df)} samples.")
print("-" * 30)
print("Class Distribution:")
print(df['Reaction_Label'].value_counts())