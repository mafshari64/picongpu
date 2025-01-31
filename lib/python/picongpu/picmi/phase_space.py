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
        species_name: str,
        period: int,
        space: str,
        momentum: str,
        min_value: float,
        max_value: float,
        filter_type: str,
        **kw,
    ):
        if period <= 0:
            raise ValueError("Period must be > 0")
        if min_value >= max_value:
            raise ValueError("min_value must be less than max_value")

        self.species_name = species_name
        self.period = period
        self.space = space
        self.momentum = momentum
        self.min_value = min_value
        self.max_value = max_value
        self.filter_type = filter_type

        super().__init__(**kw)

    def get_as_pypicongpu(self) -> phase_space.PhaseSpace:
        util.unsupported("extra attributes", self.__dict__.keys())

        pypicongpu_phase_space = phase_space.PhaseSpace(
            species_name=self.species_name,
            period=self.period,
            space=self.space,
            momentum=self.momentum,
            min_value=self.min_value,
            max_value=self.max_value,
            filter_type=self.filter_type,
        )

        return pypicongpu_phase_space
