"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

import typeguard

from picongpu.picmi.diagnostics.util import diagnostic_converts_to

from ...pypicongpu.output.macro_particle_count import (
    MacroParticleCount as PyPIConGPUMacroParticleCount,
)
from ...pypicongpu.species.species import Species as PyPIConGPUSpecies
from ..species import Species as PICMISpecies
from .timestepspec import TimeStepSpec
from typing import Union


@diagnostic_converts_to(PyPIConGPUMacroParticleCount)
@typeguard.typechecked
class MacroParticleCount:
    """
    Specifies the parameters for counting the total number of macro particles of a given species.

    This plugin counts the total number of macro particles in the simulation,
    useful for tracking particle statistics and population dynamics.

    Parameters
    ----------
    species: PICMISpecies
        Name of the particle species to count (e.g., "electron", "proton").

    period: int or TimeStepSpec
        Number of simulation steps between consecutive counts.
        Unit: steps (simulation time steps).

    name: string, optional
        Optional name for the macro particle count plugin.
    """

    def __init__(
        self,
        species: PICMISpecies,
        period: Union[int, TimeStepSpec],
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

    def check(self, dict_species_picmi_to_pypicongpu: dict[PICMISpecies, PyPIConGPUSpecies], *args, **kwargs):
        if self.species not in dict_species_picmi_to_pypicongpu.keys():
            raise ValueError(f"Species {self.species} is not known to Simulation")
        pypicongpu_species = dict_species_picmi_to_pypicongpu.get(self.species)
        if pypicongpu_species is None:
            raise ValueError(f"Species {self.species} is not mapped to a PyPIConGPUSpecies.")
