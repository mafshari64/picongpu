"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from typing import Literal, Union

import typeguard
import warnings

from picongpu.picmi.diagnostics.util import diagnostic_converts_to
from ...pypicongpu.output.phase_space import PhaseSpace as PyPIConGPUPhaseSpace
from ..species import Species as PICMISpecies
from .timestepspec import TimeStepSpec


@diagnostic_converts_to(PyPIConGPUPhaseSpace)
@typeguard.typechecked
class PhaseSpace:
    """
    Specifies the parameters for the output of Phase Space of species such as electrons.

    This plugin extracts phase-space data from the simulation, allowing
    for detailed analysis of particle distributions in position-momentum space.

    Parameters
    ----------
    species: PICMISpecies
        Particle species to track (e.g., "electron" or "proton").

    period: int or TimeStepSpec
        Number of simulation steps between consecutive outputs (e.g., 10 for every 10 steps).
        Use 0 to disable output. Alternatively, a TimeStepSpec can be provided.
        Unit: steps (simulation time steps).

    spatial_coordinate: string
        Spatial coordinate used in phase space (e.g., 'x', 'y', 'z').

    momentum_coordinate: string
        Momentum coordinate used in phase space (e.g., 'px', 'py', 'pz').

    min_momentum: float
        Minimum value for the phase-space momentum range.
        Unit: kg*m/s (momentum in SI units).

    max_momentum: float
        Maximum value for the phase-space momentum range.
        Unit: kg*m/s (momentum in SI units).
    """

    def __init__(
        self,
        species: PICMISpecies,
        period: Union[int, TimeStepSpec],
        spatial_coordinate: Literal["x", "y", "z"],
        momentum_coordinate: Literal["px", "py", "pz"],
        min_momentum: float,
        max_momentum: float,
    ):
        if not isinstance(period, (int, TimeStepSpec)):
            raise TypeError("period must be an integer or TimeStepSpec")
        if isinstance(period, int):
            if period < 0:
                raise ValueError("period must be non-negative")
            self.period = (
                TimeStepSpec([slice(None, None, period)])("steps") if period > 0 else TimeStepSpec([])("steps")
            )
        else:
            self.period = period
        self.species = species
        self.spatial_coordinate = spatial_coordinate
        self.momentum_coordinate = momentum_coordinate
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum

    def check(self, dict_species_picmi_to_pypicongpu, *args, **kwargs):
        if not isinstance(self.species, PICMISpecies):
            raise TypeError("species must be a PICMISpecies")
        if not isinstance(self.species.name, str) or not self.species.name:
            raise TypeError("species must have a non-empty name")
        if not isinstance(self.period, TimeStepSpec):
            raise TypeError("period must be a TimeStepSpec")
        if not self.period.specs:
            warnings.warn("PhaseSpace is disabled because period is empty")
        if self.min_momentum >= self.max_momentum:
            raise ValueError(
                f"PhaseSpace's min_momentum should be smaller than max_momentum. "
                f"You gave: {self.min_momentum=} and {self.max_momentum=}."
            )
        if self.species not in dict_species_picmi_to_pypicongpu.keys():
            raise ValueError(f"Species {self.species} is not known to Simulation")

        # checks if PICMISpecies instance exists in the dictionary. If yes, it returns the corresponding PyPIConGPUSpecies instance.
        # self.species refers to the species attribute of the class  PhaseSpace(picmistandard.PICMI_PhaseSpace).
        if dict_species_picmi_to_pypicongpu.get(self.species) is None:
            raise ValueError(f"Species {self.species} is not mapped to a PyPIConGPUSpecies.")
