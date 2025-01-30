"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from pypicongpu import util
from pypicongpu.rendering import RenderedObject


import typeguard
from typing import Dict


@typeguard.typechecked
class PhaseSpace(RenderedObject):
    phase_space_species_name = util.build_typesafe_property(str)
    phase_space_period = util.build_typesafe_property(int)
    phase_space_space = util.build_typesafe_property(str)
    phase_space_momentum = util.build_typesafe_property(str)
    phase_space_min = util.build_typesafe_property(float)
    phase_space_max = util.build_typesafe_property(float)
    phase_space_filter = util.build_typesafe_property(str)

    def __init__(
        self,
        species_name: str,
        period: int,
        space: str,
        momentum: str,
        min_value: float,
        max_value: float,
        filter_type: str,
    ):
        self.phase_space_species_name = species_name
        self.phase_space_period = period
        self.phase_space_space = space
        self.phase_space_momentum = momentum
        self.phase_space_min = min_value
        self.phase_space_max = max_value
        self.phase_space_filter = filter_type

    def _get_serialized(self) -> Dict:
        """Return the serialized representation of the object."""
        return {
            "phase_space_species_name": self.phase_space_species_name,
            "phase_space_period": self.phase_space_period,
            "phase_space_space": self.phase_space_space,
            "phase_space_momentum": self.phase_space_momentum,
            "phase_space_min": self.phase_space_min,
            "phase_space_max": self.phase_space_max,
            "phase_space_filter": self.phase_space_filter,
        }
