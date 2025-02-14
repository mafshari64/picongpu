"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ..pypicongpu import util, phase_space
import picmistandard
import typeguard


@typeguard.typechecked
class PhaseSpace(picmistandard.PICMI_PhaseSpace):
    """PICMI object for Phase Space diagnostics"""

    def __init__(
        self,
        species: str,
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

        super().__init__(**kw)

    def get_as_pypicongpu(self) -> phase_space.PhaseSpace:
        util.unsupported("extra attributes", self.__dict__.keys())

        pypicongpu_phase_space = phase_space.PhaseSpace(
            species=self.species,
            period=self.period,
            spatial_coordinate=self.spatial_coordinate,
            momentum=self.momentum,
            min_momentum=self.min_momentum,
            max_momentum=self.max_momentum,
        )

        return pypicongpu_phase_space
