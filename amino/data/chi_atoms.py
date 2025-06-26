"""
Dict containing atoms which define the sidechain torsion angles for each amino acid
"""

chi_atoms = dict(
    chi1=dict(
        ARG=["N", "CA", "CB", "CG", [3]],
        ASN=["N", "CA", "CB", "CG", [3]],
        ASP=["N", "CA", "CB", "CG", [3]],
        CYS=["N", "CA", "CB", "SG", [3]],
        GLN=["N", "CA", "CB", "CG", [3]],
        GLU=["N", "CA", "CB", "CG", [3]],
        HIS=["N", "CA", "CB", "CG", [3]],
        ILE=["N", "CA", "CB", "CG1", [3]],
        LEU=["N", "CA", "CB", "CG", [3]],
        LYS=["N", "CA", "CB", "CG", [3]],
        MET=["N", "CA", "CB", "CG", [3]],
        PHE=["N", "CA", "CB", "CG", [3]],
        PRO=["N", "CA", "CB", "CG", []],
        SER=["N", "CA", "CB", "OG", [3]],
        THR=["N", "CA", "CB", "OG1", [3]],
        TRP=["N", "CA", "CB", "CG", [3]],
        TYR=["N", "CA", "CB", "CG", [3]],
        VAL=["N", "CA", "CB", "CG1", [3]],
    ),
    altchi1=dict(
        VAL=["N", "CA", "CB", "CG2", [3]],
    ),
    chi2=dict(
        ARG=["CA", "CB", "CG", "CD", [1, 2, 3]],
        ASN=["CA", "CB", "CG", "OD1", [1, 2, 3]],
        ASP=["CA", "CB", "CG", "OD1", [1, 2, 3]],
        GLN=["CA", "CB", "CG", "CD", [1, 2, 3]],
        GLU=["CA", "CB", "CG", "CD", [2, 3]],
        HIS=["CA", "CB", "CG", "ND1", [1, 2, 3]],
        ILE=["CA", "CB", "CG1", "CD1", [1, 2, 3]],
        LEU=["CA", "CB", "CG", "CD1", [1, 2, 3]],
        LYS=["CA", "CB", "CG", "CD", [1, 2, 3]],
        MET=["CA", "CB", "CG", "SD", [1, 2, 3]],
        PHE=["CA", "CB", "CG", "CD1", [1, 2, 3]],
        PRO=["CA", "CB", "CG", "CD", []],
        TRP=["CA", "CB", "CG", "CD1", [1, 2, 3]],
        TYR=["CA", "CB", "CG", "CD1", [1, 2, 3]],
    ),
    altchi2=dict(
        ASP=["CA", "CB", "CG", "OD2", [1, 2, 3]],
        LEU=["CA", "CB", "CG", "CD2", [1, 2, 3]],
        PHE=["CA", "CB", "CG", "CD2", [1, 2, 3]],
        TYR=["CA", "CB", "CG", "CD2", [1, 2, 3]],
    ),
    chi3=dict(
        ARG=["CB", "CG", "CD", "NE", [3]],
        GLN=["CB", "CG", "CD", "OE1", [6]],
        GLU=["CB", "CG", "CD", "OE1", [6]],
        LYS=["CB", "CG", "CD", "CE", [2, 4, 6]],
        MET=["CB", "CG", "SD", "CE", [1, 3]],
    ),
    chi4=dict(
        ARG=["CG", "CD", "NE", "CZ", [0]],
        LYS=["CG", "CD", "CE", "NZ", [3]],
    ),
    chi5=dict(
        # ARG=["CD", "NE", "CZ", "NH1", [2]],
    ),
)
