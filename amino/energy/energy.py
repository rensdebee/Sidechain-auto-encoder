import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import openmm
from openmm import (
    CustomNonbondedForce,
    HarmonicAngleForce,
    HarmonicBondForce,
    NonbondedForce,
    PeriodicTorsionForce,
    Vec3,
    unit,
)
from openmm.app import ForceField, Modeller, PDBFile, Simulation
from pdbfixer import PDBFixer
from tqdm import tqdm

from amino.utils.utils import create_clean_path


def remove_oxt_influence(force, oxt_index):
    if isinstance(force, NonbondedForce):
        charge, sigma, epsilon = force.getParticleParameters(oxt_index)
        force.setParticleParameters(oxt_index, 0, sigma, 0)  # Neutralize charge and LJ
    elif isinstance(force, CustomNonbondedForce):
        acoef_func = force.getTabulatedFunction(0)
        bcoef_func = force.getTabulatedFunction(1)
        aX, aY, old_acoef = acoef_func.getFunctionParameters()
        bX, bY, old_bcoef = bcoef_func.getFunctionParameters()
        # print(f"LJForce table {aX}, {aY}, {bX}, {bY}")

        assert aX == bY, f"Uneven LJForce table {aX}, {aY}, {bX}, {bY}"

        numLjTypes = aX
        new_numLjTypes = numLjTypes + 1
        new_acoef = [0.0] * (new_numLjTypes * new_numLjTypes)
        new_bcoef = [0.0] * (new_numLjTypes * new_numLjTypes)

        # Copy existing data into the new tables
        for i in range(numLjTypes):
            for j in range(numLjTypes):
                idx = i * numLjTypes + j
                new_idx = i * new_numLjTypes + j
                new_acoef[new_idx] = old_acoef[idx]
                new_bcoef[new_idx] = old_bcoef[idx]

        # Create new tabulated functions with extended tables
        acoef_func.setFunctionParameters(new_numLjTypes, new_numLjTypes, new_acoef)
        bcoef_func.setFunctionParameters(new_numLjTypes, new_numLjTypes, new_bcoef)

        # Assign the new type to the OXT atom
        force.setParticleParameters(oxt_index, [numLjTypes])
        acoef_func = force.getTabulatedFunction(0)
        bcoef_func = force.getTabulatedFunction(1)
        aX, aY, old_acoef = acoef_func.getFunctionParameters()
        bX, bY, old_bcoef = bcoef_func.getFunctionParameters()
        # print(f"LJForce table {aX}, {aY}, {bX}, {bY}")

    elif isinstance(force, HarmonicBondForce):
        for i in range(force.getNumBonds()):
            p1, p2, length, k = force.getBondParameters(i)
            if oxt_index in (p1, p2):
                force.setBondParameters(i, p1, p2, length, 0)
    elif isinstance(force, HarmonicAngleForce):
        for i in range(force.getNumAngles()):
            p1, p2, p3, angle, k = force.getAngleParameters(i)
            if oxt_index in (p1, p2, p3):
                force.setAngleParameters(i, p1, p2, p3, angle, 0)
    elif isinstance(force, PeriodicTorsionForce):
        for i in range(force.getNumTorsions()):
            p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
            if oxt_index in (p1, p2, p3, p4):
                force.setTorsionParameters(i, p1, p2, p3, p4, periodicity, phase, 0)


def pdb_to_energy(
    pdbfile,
    forcefield,
    minimize_steps=0,
    platform="CPU",
    write=False,
):
    platform = openmm.Platform.getPlatformByName(platform)
    if type(pdbfile) is str:
        fixer = PDBFixer(filename=pdbfile)
    else:
        fixer = PDBFixer(pdbfile=pdbfile)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=42)
    modeller = Modeller(fixer.topology, fixer.positions)
    oxt_index = next(
        atom.index for atom in modeller.topology.atoms() if atom.name == "OXT"
    )
    # Place OXT 10 nm away from the C-terminal carbon
    oxt_pos = modeller.positions[oxt_index]
    modeller.positions[oxt_index] = Vec3(10, 0, 0) * unit.nanometers
    modeller.addHydrogens(forcefield, platform=platform)
    oxt_index = next(
        atom.index for atom in modeller.topology.atoms() if atom.name == "OXT"
    )

    modeller.positions[oxt_index] = oxt_pos

    integrator = openmm.VerletIntegrator(1.0)
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=openmm.app.NoCutoff,
        ignoreExternalBonds=True,
    )

    forces = {}
    for i, force in enumerate(system.getForces()):
        force_group = i + 1
        name = type(force).__name__
        remove_oxt_influence(force, oxt_index)
        if isinstance(force, NonbondedForce):
            name = "CoulombForce"
        elif isinstance(force, CustomNonbondedForce):
            name = "LJForce"
        forces[force_group] = name
        force.setForceGroup(force_group)

    constraints_to_remove = []
    for i in range(system.getNumConstraints()):
        p1, p2, _ = system.getConstraintParameters(i)
        if oxt_index in (p1, p2):
            constraints_to_remove.append(i)
    for i in reversed(constraints_to_remove):
        system.removeConstraint(i)

    simulation = Simulation(
        modeller.topology,
        system,
        integrator,
        platform,
    )

    simulation.context.setPositions(modeller.positions)
    if minimize_steps > 0 or minimize_steps == -1:
        if minimize_steps == -1:
            minimize_steps = 0
        simulation.minimizeEnergy(maxIterations=minimize_steps, tolerance=0.00001)

    energys = {}
    for force_group, force_name in forces.items():
        state = simulation.context.getState(
            getEnergy=True,
            groups={force_group},
        )
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        energys[force_name] = energy

    state = simulation.context.getState(
        getEnergy=True,
    )
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    energys["TotalEnergy"] = energy

    if type(write) == list:
        modeller.delete([a for a in modeller.topology.atoms() if a.name == "OXT"])
        PDBFile.writeFile(
            modeller.topology,
            modeller.positions,
            f"{write[0]}/{write[1]}_{energys['CoulombForce']:.3f}.pdb",
        )

    del simulation
    del state
    return energys


if __name__ == "__main__":
    path = "pdbs/rotated_ARG_with_H"
    create_clean_path(path)
    files = glob.glob("pdbs/rotated_ARG/*.pdb")
    files.sort(
        key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)]
    )
    forcefield = ForceField("charmm36.xml")
    energy_data = [
        pdb_to_energy(
            f,
            forcefield=forcefield,
            minimize_steps=0,
            platform="CPU",
            # write=[path, i],
        )
        for i, f in enumerate(tqdm(files))
    ]

    energy_types = list(energy_data[0].keys())
    y = len(energy_types)
    fig, axes = plt.subplots(2, int(y // 2), figsize=(16, 8))
    axes = axes.flatten()

    idx = 0
    for energy_type in energy_types:
        if energy_type == "CMMotionRemover":
            continue
        if energy_type == "CMAPTorsionForce":
            continue
        energies = [e[energy_type] for e in energy_data]
        axes[idx].plot(
            np.arange(len((energies))),
            energies,
            marker="o",
            linestyle="-",
            markersize=4,
            linewidth=1,
        )
        axes[idx].set_title(energy_type, fontsize=14, fontweight="bold")
        axes[idx].set_xlabel("Dihedral Angle (degrees)", fontsize=12)
        axes[idx].set_ylabel("Energy (kJ/mol)", fontsize=12)
        axes[idx].grid(True, linestyle="--", linewidth=0.5)
        idx += 1

    idx = -2
    energies = [
        sum(v for k, v in d.items() if k == "CoulombForce" or k == "LJForce")
        for d in energy_data
    ]
    axes[idx].plot(
        np.arange(len((energies))),
        energies,
        marker="o",
        linestyle="-",
        markersize=4,
        linewidth=1,
    )
    axes[idx].set_title("LJ + Coulomb", fontsize=14, fontweight="bold")
    axes[idx].set_xlabel("Dihedral Angle (degrees)", fontsize=12)
    axes[idx].set_ylabel("Energy (kJ/mol)", fontsize=12)
    axes[idx].grid(True, linestyle="--", linewidth=0.5)

    idx = -1
    energies = [sum(v for k, v in d.items() if k != "TotalEnergy") for d in energy_data]
    axes[idx].plot(
        np.arange(len((energies))),
        energies,
        marker="o",
        linestyle="-",
        markersize=4,
        linewidth=1,
    )
    axes[idx].set_title("Summed Total Energy", fontsize=14, fontweight="bold")
    axes[idx].set_xlabel("Dihedral Angle (degrees)", fontsize=12)
    axes[idx].set_ylabel("Energy (kJ/mol)", fontsize=12)
    axes[idx].grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("figures/energy_components_charm_backbone_neutralized.png")
