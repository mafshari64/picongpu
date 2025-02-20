"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Brian Edward Marre, Masoud Afshari
License: GPLv3+
"""

from ..rendering import RenderedObject

import typeguard


@typeguard.typechecked
class Plugin(RenderedObject):
    """general interface for all plugins"""

    def __init__(self):
        raise NotImplementedError("abstract base class only")
