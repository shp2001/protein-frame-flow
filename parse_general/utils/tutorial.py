from BioMol.BioMol import BioMol

if __name__ == "__main__":
    # Automatically generate all biological assemblies
    biomol = BioMol(
        pdb_ID="9dw6",       # lowercase
        mol_types=["protein"],   # only protein
    )
    # Alternative initialization:
    # biomol = BioMol(
    #     cif="9dw6.cif",    # must be downloaded from PDB
    #     mol_types=["protein","nucleic_acid","ligand"],
    #     remove_signal_peptide=True,
    #     use_lmdb=False,      # required for NA or ligand loading
    # )

    # Select assembly, model, and alt_id
    biomol.choose("1", "1", ".")

    # Save loaded structure to mmCIF
    biomol.structure.to_pdb("loaded_9dw6.pdb")

    # Crop and load MSA
    biomol.crop_and_load_msa(
        chain_bias=('A_1',),
        interaction_bias=('A_1','C_1'),
        params={
            "method_prob": [0.0, 0.0, 1.0],  # contiguous, spatial, interface
            "crop_size": 384,
        }
    )