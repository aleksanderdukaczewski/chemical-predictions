import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors, MolFromSmiles

# Generate descriptors such as LogP, Molecular Weight, Rotatable bonds, Aromatic Proportion
def generate_descriptors(smilesSeries) -> pd.DataFrame:
    # Define functions to count halogen atoms
    countChlorine = lambda s: s.count("Cl")
    countFluorine = lambda s: s.count("F")
    countBromine = lambda s: s.count("Br")
    countIodine = lambda s: s.count("I")
    countHalogens = lambda s: (countChlorine(s) + countFluorine(s) + countBromine(s) + countIodine(s))

    # Calculate aromatic proportion
    aromatic_atoms = lambda m: sum([1 for aromatic in [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())] if aromatic])
    aromatic_proportion = lambda m: aromatic_atoms(m) / Descriptors.HeavyAtomCount(m)

    molecule_list = []
    for s in smilesSeries:
        mol = MolFromSmiles(s)
        molecule_list.append((s, mol))

    base_data = np.arange(1,1)
    i = 0
    for s, mol in molecule_list:
        desc_LogP = Descriptors.MolLogP(mol)
        desc_AromaticProportion = aromatic_proportion(mol)
        # desc_MolWt = Descriptors.MolWt(mol)
        # desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_FluorineAtoms = countFluorine(s)
        desc_ChlorineAtoms = countChlorine(s)
        desc_BromineAtoms = countBromine(s)
        desc_IodineAtoms = countIodine(s)
        desc_TotalHalogenAtoms = countHalogens(s)

        row = np.array([
            desc_LogP,
            desc_AromaticProportion,
            # desc_MolWt,
            # desc_NumRotatableBonds,
            desc_FluorineAtoms,
            desc_ChlorineAtoms,
            desc_BromineAtoms,
            desc_IodineAtoms,
            desc_TotalHalogenAtoms
        ])

        if i == 0:
            base_data = row
        else:
            base_data = np.vstack([base_data, row])
        i += 1

    column_names = [
        "LogP",
        "AromaticProportion",
        # "MolecularWeight",
        # "RotatableBonds",
        "FluorineAtoms",
        "ChlorineAtoms",
        "BromineAtoms",
        "IodineAtoms",
        "TotalHalogenAtoms"
    ]
    descriptors = pd.DataFrame(data=base_data, columns=column_names)
    return descriptors

# Read the original dataset
df = pd.read_csv('./datasets/esol2.csv', delimiter=",")

# Add descriptors to the DataFrame
desc = generate_descriptors(df['smiles'])
df = df.join(desc)

# Export results
df.to_csv("./datasets/processed_esol.csv", index=False)
