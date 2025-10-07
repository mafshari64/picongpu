"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Pawel Ordyna, Masoud Afshari
License: GPLv3+
"""

import typeguard

from ...pypicongpu.output.auto import Auto as PyPIConGPUAuto
from ..copy_attributes import default_converts_to
from .timestepspec import TimeStepSpec
from typing import Union


@default_converts_to(PyPIConGPUAuto)
@typeguard.typechecked
class Auto:
    """
    Specifies the parameters for the Auto output.

    Parameters
    ----------
    period: int or TimeStepSpec
        Number of simulation steps between consecutive outputs.
        Unit: steps (simulation time steps).
    """

    def __init__(
        self,
        period: Union[int, TimeStepSpec],
    ) -> None:
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
