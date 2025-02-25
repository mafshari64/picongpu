"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu import Runner
from picongpu import picmi

import unittest


class TestDistribution(unittest.TestCase):
    """general test case to check if distributions compile"""

    def setUp(self):
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            lower_bound=[0, 0, 0],
            upper_bound=[3.40992e-5, 9.07264e-5, 2.1312e-6],
            lower_boundary_conditions=["open", "open", "periodic"],
            upper_boundary_conditions=["open", "open", "periodic"],
        )
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(time_step_size=1.39e-16, max_steps=int(2048), solver=solver)

        self.grid = grid
        self.solver = solver
        self.sim = sim

    def _compile_distribution(self, distribution):
        random_layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=2)
        species_hydrogen = picmi.Species(
            name="hydrogen",
            particle_type="H",
            charge_state=0,
            initial_distribution=distribution,
            picongpu_fixed_charge=True,
        )
        self.sim.add_species(species_hydrogen, random_layout)
        runner = Runner(self.sim)
        runner.generate(printDirToConsole=True)
        runner.build()

    def test_uniform(self):
        uniform_dist = picmi.UniformDistribution(density=8e24)
        self._compile_distribution(uniform_dist)

    def test_foil_pre_and_post(self):
        foil_dist = picmi.FoilDistribution(
            density=8.0e24,
            front=2.0e-5,
            thickness=1.0e-5,
            exponential_pre_plasma_length=1.0e-6,
            exponential_pre_plasma_cutoff=1.0e-5,
            exponential_post_plasma_length=1.0e-6,
            exponential_post_plasma_cutoff=1.0e-5,
        )
        self._compile_distribution(foil_dist)

    def test_foil_pre(self):
        foil_dist = picmi.FoilDistribution(
            density=8.0e24,
            front=2.0e-5,
            thickness=1.0e-5,
            exponential_pre_plasma_length=1.0e-6,
            exponential_pre_plasma_cutoff=1.0e-5,
        )
        self._compile_distribution(foil_dist)

    def test_foil_post(self):
        # with post-plasma only
        foil_dist = picmi.FoilDistribution(
            density=8.0e24,
            front=2.0e-5,
            thickness=1.0e-5,
            exponential_post_plasma_length=1.0e-6,
            exponential_post_plasma_cutoff=1.0e-5,
        )
        self._compile_distribution(foil_dist)

    def test_foil_nothing(self):
        # with no pre- or post-plasma
        foil_dist = picmi.FoilDistribution(density=8.0e24, front=2.0e-5, thickness=1.0e-5)
        self._compile_distribution(foil_dist)

    def test_gaussian(self):
        gaussian_dist = picmi.GaussianDistribution(
            center_front=2.0e-5,
            center_rear=3.0e-5,
            sigma_front=5.0e-6,
            sigma_rear=5.0e-6,
            power=4.0,
            factor=-1.0,
            vacuum_cells_front=50,
            density=8e24,
        )
        self._compile_distribution(gaussian_dist)

    # tests for analytic distribution and gaussian-bunch distribution have been
    #   removed for now, see issue #4367 for the test cases

    def test_temperature(self):
        uniform_dist = picmi.UniformDistribution(density=8e24, rms_velocity=[1e7, 1e7, 1e7])
        self._compile_distribution(uniform_dist)

    def test_velocity(self):
        uniform_dist = picmi.UniformDistribution(density=8e24, directed_velocity=[-5e6, 2.5e7, 0.55])
        self._compile_distribution(uniform_dist)
