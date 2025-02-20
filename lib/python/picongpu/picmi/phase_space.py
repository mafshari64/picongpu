"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ..pypicongpu import util
from ..pypicongpu.output.phase_space import PhaseSpace
from ..pypicongpu.species.species import Species as PyPIConGPUSpecies


from .species import Species as PICMISpecies

import picmistandard
import typeguard


@typeguard.typechecked
class PhaseSpace(picmistandard.PICMI_PhaseSpace):
    """PICMI object for Phase Space diagnostics"""

    def __init__(
        self,
        species: PICMISpecies,
        period: int,
        spatial_coordinate: str,
        momentum: str,
        min_momentum: float,
        max_momentum: float,
        **kw,
    ):
        if period <= 0:
            raise ValueError("Period must be > 0")
        if min_momentum >= max_momentum:
            raise ValueError("min_momentum must be less than max_momentum")

        self.species = species
        self.period = period
        self.spatial_coordinate = spatial_coordinate
        self.momentum = momentum
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum

        super().__init__(
            species,
            period,
            spatial_coordinate,
            momentum,
            min_momentum,
            max_momentum,
            **kw,
        )

    def get_as_pypicongpu(
        # to get the corresponding PyPIConGPUSpecies instance for the given PICMISpecies.
        self,
        dict_species_picmi_to_pypicongpu: dict[PICMISpecies, PyPIConGPUSpecies],
    ) -> PhaseSpace:
        # print(f"dict_species_picmi_to_pypicongpu keys: {list(dict_species_picmi_to_pypicongpu.keys())}")
        # print(f"self.species: {self.species}")

        util.unsupported("extra attributes", self.__dict__.keys())

        if self.species not in dict_species_picmi_to_pypicongpu:
            raise ValueError(f"Species {self.species} is not mapped in dict_species_picmi_to_pypicongpu!")

        # checks if PICMISpecies instance exists in the dictionary. If yes, it returns the corresponding PyPIConGPUSpecies instance.
        # self.species refers to the species attribute of the class  PhaseSpace(picmistandard.PICMI_PhaseSpace).
        pypicongpu_species = dict_species_picmi_to_pypicongpu.get(self.species)

        if pypicongpu_species is None:
            raise ValueError(f"Species {self.species} is not mapped to a PyPIConGPUSpecies.")

        # Print type before passing to PhaseSpace
        print(f"DEBUG: Mapped species: {pypicongpu_species}, Type: {type(pypicongpu_species)}")

        pypicongpu_phase_space = PhaseSpace(
            species=pypicongpu_species,
            period=self.period,
            spatial_coordinate=self.spatial_coordinate,
            momentum=self.momentum,
            min_momentum=self.min_momentum,
            max_momentum=self.max_momentum,
        )

        return pypicongpu_phase_space
