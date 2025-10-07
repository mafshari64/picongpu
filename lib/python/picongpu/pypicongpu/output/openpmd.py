"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .. import util
from .timestepspec import TimeStepSpec
from .rangespec import RangeSpec
from .plugin import Plugin
from .openpmd_sources.source_base import SourceBase

import typeguard
import typing
from typing import Optional, List, Literal, Dict, Union


@typeguard.typechecked
class OpenPMD(Plugin):
    period = util.build_typesafe_property(TimeStepSpec)
    source = util.build_typesafe_property(Optional[List[SourceBase]])
    range = util.build_typesafe_property(Optional[RangeSpec])
    file = util.build_typesafe_property(Optional[str])
    ext = util.build_typesafe_property(Optional[Literal["bp", "h5", "sst"]])
    infix = util.build_typesafe_property(Optional[str])
    json = util.build_typesafe_property(Union[str, Dict, None])
    json_restart = util.build_typesafe_property(Union[str, Dict, None])
    data_preparation_strategy = util.build_typesafe_property(
        Optional[Literal["doubleBuffer", "adios", "mappedMemory", "hdf5"]]
    )
    toml = util.build_typesafe_property(Optional[str])
    particle_io_chunk_size = util.build_typesafe_property(Optional[int])
    file_writing = util.build_typesafe_property(Optional[Literal["create", "append"]])

    _name = "openpmd"

    def __init__(self):
        "do nothing"

    def _get_serialized(self) -> typing.Dict:
        return {
            "period": self.period.get_rendering_context(),
            "source": [s._get_serialized() for s in self.source] if self.source is not None else None,
            "range": self.range._get_serialized() if self.range else None,
            "file": self.file,
            "ext": self.ext,
            "infix": self.infix,
            "json": self.json,
            "json_restart": self.json_restart,
            "data_preparation_strategy": self.data_preparation_strategy,
            "toml": self.toml,
            "particle_io_chunk_size": self.particle_io_chunk_size,
            "file_writing": self.file_writing,
        }
