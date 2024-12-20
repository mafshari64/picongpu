#!/usr/bin/env python

"""
This file is part of PIConGPU.

Copyright 2018-2024 PIConGPU contributors
Authors: Marco Garten, Pawel Ordyna,Brian Marre
License: GPLv3+
"""

from picongpu.extra.utils.memory_calculator import MemoryCalculator

import numpy as np

"""
@file

This file contains a usage example of the ``MemoryCalculator`` class
for our :ref:`FoilLCT example <usage-examples-foilLCT>` and its ``4.cfg``.

It is an estimate for how much memory is used per device if the whole
target would be fully ionized but does not move much. Of course the real
memory usage depends on the case and the dynamics inside the simulation.

We calculate the memory of just one device per row of GPUs in laser
propagation direction. We hereby assume that particles are distributed
equally in transverse direction, like it is set up in the FoilLCT example.
"""

cell_size = 0.8e-6 / 384.0  # 2.083e-9 m
y0 = 0.5e-6  # position of foil front surface (m)
y1 = 1.5e-6  # position of foil rear surface (m)
L = 10e-9  # pre-plasma scale length (m)
L_cutoff = 4 * L  # pre-plasma length (m)

# number of vacuum cells in front of the target
vacuum_cells = int(np.ceil((y0 - L_cutoff) / cell_size))
# number of target cells over the depth of the target(between surfaces + pre-plasma)
target_cells = int(np.ceil((y1 - y0 + 2 * L_cutoff) / cell_size))

simulation_dimension = 2

# number of cells in the simulation in each direction
global_cell_extent = np.array((256, 1280))[:simulation_dimension]
# number of GPUs in each direction
global_gpu_extent = np.array((2, 2))[:simulation_dimension]

super_cell_size = np.array((16, 16, 1))

# how many cells each gpu in this row of this dimension handles
#   for example grid_distribution = [np.array((64,192)), np.array((960, 320))]
#   here we distribute cells evenly, but this is not required
if np.any(global_cell_extent % global_gpu_extent != 0):
    raise ValueError("global cell extent must be divisble by the global gpu extent")
grid_distribution = []
for dim in range(simulation_dimension):
    row_division = np.full(global_gpu_extent[dim], int(global_cell_extent[dim] / global_gpu_extent[dim]), dtype=np.int_)
    grid_distribution.append(row_division)

print(f"grid distribution: {grid_distribution}")

# get cell extent of each GPU: list of np.array[np.int_], one per simulation dimension, with each array entry being the
#   cell extent of the corresponding gpu in the simulation, indexation by [simulation_dimension, gpu_index[0], gpu_index[1], ...]
gpu_cell_extent = np.meshgrid(*grid_distribution)

# extent of cells filled with particles for each gpu
#   init
gpu_particle_cell_extent = np.empty_like(gpu_cell_extent, dtype=np.int_)

#   go through gpus and fill for each gpu
for gpu_index in np.ndindex(tuple(global_gpu_extent)):
    # calculate offset of gpu in cells
    gpu_cell_offset = np.empty(simulation_dimension)
    for dim in range(simulation_dimension):
        gpu_cell_offset[dim] = np.sum(grid_distribution[dim][: gpu_index[dim]])

    # figure out the extent of the particle filled cells of the gpu
    #   for our example figure out how many cells in y-direction belong to the foil + pre-plasma
    end_gpu_domain = gpu_cell_offset[1] + gpu_cell_extent[1][gpu_index]
    start_gpu_domain = gpu_cell_offset[1]

    start_foil = vacuum_cells
    end_foil = vacuum_cells + target_cells

    if (end_gpu_domain < start_foil) or (start_gpu_domain > end_foil):
        # completly outside
        for dim in range(simulation_dimension):
            gpu_particle_cell_extent[dim][gpu_index] = 0

    elif end_gpu_domain > end_foil:
        # partial and at the end
        for dim in range(simulation_dimension):
            if dim != 1:
                gpu_particle_cell_extent[dim][gpu_index] = gpu_cell_extent[dim][gpu_index]
            else:
                gpu_particle_cell_extent[dim][gpu_index] = gpu_cell_extent[dim][gpu_index] - (end_gpu_domain - end_foil)

    elif start_gpu_domain < start_foil:
        # partial and at the front
        for dim in range(simulation_dimension):
            if dim != 1:
                gpu_particle_cell_extent[dim][gpu_index] = gpu_cell_extent[dim][gpu_index]
            else:
                gpu_particle_cell_extent[dim][gpu_index] = gpu_cell_extent[dim][gpu_index] - (
                    start_foil - start_gpu_domain
                )

    else:
        # fully inside target
        for dim in range(simulation_dimension):
            gpu_particle_cell_extent[dim][gpu_index] = gpu_cell_extent[dim][gpu_index]

mc = MemoryCalculator(simulation_dimension=simulation_dimension, super_cell_size=super_cell_size)

# typical number of particles per cell which is multiplied later for each species and its relative number of particles
N_PPC = 6

# conversion factor to megabyte
megabyte = 1024 * 1024


def dimension_name(i: int) -> str:
    if i == 0:
        return "x"
    if i == 1:
        return "y"
    if i == 2:
        return "z"
    raise ValueError("dimensions over 3 are not supported")


for gpu_index in np.ndindex(tuple(global_gpu_extent)):
    gpu_header_string = "GPU ("
    for dim in range(simulation_dimension):
        gpu_header_string += dimension_name(dim) + f" = {gpu_index[dim]}, "
    print(gpu_header_string[:-2] + ")")
    print("* Memory requirement:")

    cell_extent = np.empty(simulation_dimension)
    for dim in range(simulation_dimension):
        cell_extent[dim] = gpu_cell_extent[dim][gpu_index]

    # field memory per GPU
    field_gpu = mc.memory_required_by_cell_fields(cell_extent, number_of_temporary_field_slots=2)
    print(f" + fields: {field_gpu / megabyte:.2f} MB")

    # memory for random number generator states
    rng_gpu = mc.memory_required_by_random_number_generator(cell_extent)

    # electron macroparticles per supercell
    e_PPC = N_PPC * (
        # H,C,N pre-ionization - higher weighting electrons
        3
        # electrons created from C ionization
        + (6 - 2)
        # electrons created from N ionization
        + (7 - 2)
    )

    particle_cell_extent = np.empty(simulation_dimension)
    for dim in range(simulation_dimension):
        particle_cell_extent[dim] = gpu_particle_cell_extent[dim][gpu_index]

    # particle memory per GPU - only the target area contributes here
    e_gpu = mc.memory_required_by_particles_of_species(
        particle_filled_cells=particle_cell_extent,
        species_attribute_list=["momentum", "position", "weighting"],
        particles_per_cell=e_PPC,
    )
    H_gpu = mc.memory_required_by_particles_of_species(
        particle_filled_cells=particle_cell_extent,
        species_attribute_list=["momentum", "position", "weighting"],
        particles_per_cell=N_PPC,
    )
    C_gpu = mc.memory_required_by_particles_of_species(
        particle_filled_cells=particle_cell_extent,
        species_attribute_list=["momentum", "position", "weighting", "boundElectrons"],
        particles_per_cell=N_PPC,
    )
    N_gpu = mc.memory_required_by_particles_of_species(
        particle_filled_cells=particle_cell_extent,
        species_attribute_list=["momentum", "position", "weighting", "boundElectrons"],
        particles_per_cell=N_PPC,
    )

    # memory for calorimeters
    cal_gpu = (
        mc.memory_required_by_calorimeter(number_energy_bins=1024, number_yaw_bins=360, number_pitch_bins=1) * 2
    )  # electrons and protons

    print(" + species:")
    print(f"  - e: {e_gpu / megabyte:.2f} MB")
    print(f"  - H: {H_gpu / megabyte:.2f} MB")
    print(f"  - C: {C_gpu / megabyte:.2f} MB")
    print(f"  - N: {N_gpu / megabyte:.2f} MB")
    print(f" + RNG states: {rng_gpu / megabyte:.2f} MB")
    print(f" + particle calorimeters: {cal_gpu / megabyte:.2f} MB")

    mem_sum = field_gpu + e_gpu + H_gpu + C_gpu + N_gpu + rng_gpu + cal_gpu
    print(f"* Sum of required GPU memory: {mem_sum / megabyte:.2f} MB")
