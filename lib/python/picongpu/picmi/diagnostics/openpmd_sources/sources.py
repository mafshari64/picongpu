"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

import typing
import typeguard
from ... import pypicongpu

from .source_base import SourceBase
from ...species import Species as PICMISpecies
from ...pypicongpu.output.openpmd_sources import (
    EnergyDensityCutoff as PyPIConGPUEnergyDensityCutoff,
    Momentum as PyPIConGPUMomentum,
)

# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


@typeguard.typechecked
class SourceBaseSpeciesFilter(SourceBase):
    """Common base for sources that use (species, filter)."""

    def __init__(self, species: PICMISpecies, filter: str = "species_all"):
        self.species = species
        self._filter = filter
        self.check()

    @property
    def filter(self) -> str:
        return self._filter

    def check(
        self,
        dict_species_picmi_to_pypicongpu: typing.Optional[typing.Dict[PICMISpecies, typing.Any]] = None,
        simulation_box: typing.Optional[typing.Tuple[int, ...]] = None,
    ) -> None:
        valid_filters = ["species_all", "fields_all", "custom_filter"]
        if not isinstance(self._filter, str):
            raise ValueError(f"Filter must be a string, got {type(self._filter)}")
        if self._filter not in valid_filters:
            raise ValueError(f"Filter must be one of {valid_filters}, got {self._filter}")
        if not isinstance(self.species, PICMISpecies):
            raise TypeError(f"Species must be a PICMISpecies, got {type(self.species)}")
        if dict_species_picmi_to_pypicongpu and self.species not in dict_species_picmi_to_pypicongpu:
            raise ValueError(f"Species {self.species} is not known to Simulation")
        if dict_species_picmi_to_pypicongpu and dict_species_picmi_to_pypicongpu.get(self.species) is None:
            raise ValueError(f"Species {self.species} is not mapped to a PyPIConGPUSpecies")
        if simulation_box and len(simulation_box) not in [1, 2, 3]:
            raise ValueError(f"Simulation box must have 1, 2, or 3 dimensions, got {len(simulation_box)}")

    def _map_species(self, dict_species_picmi_to_pypicongpu: typing.Dict[PICMISpecies, typing.Any]) -> typing.Any:
        try:
            return dict_species_picmi_to_pypicongpu[self.species]
        except KeyError:
            raise ValueError(f"Species {self.species} is not known to Simulation") from None

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: typing.Dict[PICMISpecies, typing.Any],
        time_step_size: float = 0.0,
        num_steps: int = 0,
        simulation_box: typing.Optional[typing.Tuple[int, ...]] = None,
    ) -> typing.Any:
        self.check(dict_species_picmi_to_pypicongpu, simulation_box)
        mapped_species = self._map_species(dict_species_picmi_to_pypicongpu)
        return getattr(pypicongpu.output.openpmd_sources, self.__class__.__name__)(
            species=mapped_species,
            filter=self._filter,
            simulation_box=simulation_box,
        )


@typeguard.typechecked
class SourceBaseFilterOnly(SourceBase):
    """Common base for sources that only use filter."""

    def __init__(self, filter: str = "species_all"):
        self._filter = filter
        self.check()

    @property
    def filter(self) -> str:
        return self._filter

    def check(
        self,
        dict_species_picmi_to_pypicongpu: typing.Optional[typing.Dict] = None,
        simulation_box: typing.Optional[typing.Tuple[int, ...]] = None,
    ) -> None:
        valid_filters = ["species_all", "fields_all", "custom_filter"]
        if not isinstance(self._filter, str):
            raise ValueError(f"Filter must be a string, got {type(self._filter)}")
        if self._filter not in valid_filters:
            raise ValueError(f"Filter must be one of {valid_filters}, got {self._filter}")
        if simulation_box and len(simulation_box) not in [1, 2, 3]:
            raise ValueError(f"Simulation box must have 1, 2, or 3 dimensions, got {len(simulation_box)}")

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: typing.Optional[typing.Dict] = None,
        time_step_size: float = 0.0,
        num_steps: int = 0,
        simulation_box: typing.Optional[typing.Tuple[int, ...]] = None,
    ) -> typing.Any:
        self.check(simulation_box=simulation_box)
        return getattr(pypicongpu.output.openpmd_sources, self.__class__.__name__)(
            filter=self._filter,
            simulation_box=simulation_box,
        )


# ---------------------------------------------------------------------------
# Sources with (species + filter, no extras)
# ---------------------------------------------------------------------------


@typeguard.typechecked
class BoundElectronDensity(SourceBaseSpeciesFilter):
    """
    Bound electron density diagnostic for PIConGPU.
    """

    pass


@typeguard.typechecked
class ChargeDensity(SourceBaseSpeciesFilter):
    """
    Charge density data source for openPMD output in PIConGPU.

    Calculates the charge density from a specified particle species, optionally
    filtered by a selection criterion, for particle-in-cell simulations.
    """

    pass


@typeguard.typechecked
class Counter(SourceBaseSpeciesFilter):
    """
    Particle counter data source for openPMD output in PIConGPU.

    Derives a scalar field representing the number of real particles per cell
    for a specified species, optionally filtered by a selection criterion.
    The particle count is based on the species' weighting attribute and assigned
    directly to the cell containing each particle. Intended primarily for debugging
    due to its non-physical deposition shape.
    """

    pass


@typeguard.typechecked
class Density(SourceBaseSpeciesFilter):
    """
    Particle density data source for openPMD output in PIConGPU.

    Derives a scalar field representing the number density (in m^-3) of a specified
    particle species, optionally filtered by a selection criterion.
    The density is calculated based on the species' weighting and position attributes
    and mapped to cells according to the PIC code's spatial shape assignment.
    """

    pass


@typeguard.typechecked
class Energy(SourceBaseSpeciesFilter):
    """
    Kinetic energy data source for openPMD output in PIConGPU.

    Derives a scalar field of summed kinetic energy (in Joules) for a specified particle species,
    optionally filtered. Uses weighting, momentum, and mass attributes, mapped to cells by the
    PIC code's spatial shape.
    """

    pass


@typeguard.typechecked
class EnergyDensity(SourceBaseSpeciesFilter):
    """
    Kinetic energy density data source for openPMD output in PIConGPU.

    Derives a scalar field of kinetic energy density (in J/m^3) for a specified particle species,
    optionally filtered, in particle-in-cell simulations. Uses weighting, momentum, and mass attributes,
    mapped to cells by the PIC code's spatial shape.
    """

    pass


@typeguard.typechecked
class LarmorPower(SourceBaseSpeciesFilter):
    """
    Radiated Larmor power data source for openPMD output in PIConGPU.

    Derives a scalar field of radiated power (in Joules) for a specified particle species,
    optionally filtered, using the Larmor formula in particle-in-cell simulations. Uses
    weighting, position, momentum, momentumPrev1, mass, and charge attributes, mapped to
    cells by the PIC code's spatial shape.
    """

    pass


@typeguard.typechecked
class MacroCounter(SourceBaseSpeciesFilter):
    """
    Macro-particle counter data source for openPMD output in PIConGPU.

    Derives a scalar field counting macro-particles per cell for a specified particle species,
    optionally filtered, in particle-in-cell simulations. Assigns each macro-particle directly
    to its cell via floor operation. Intended for debugging (e.g., validating particle memory).
    """

    pass


# ---------------------------------------------------------------------------
# Sources with (filter only)
# ---------------------------------------------------------------------------


@typeguard.typechecked
class Auto(SourceBaseFilterOnly):
    """
    Default data source for openPMD output in PIConGPU.

    Provides a convenient way to dump default simulation data (e.g., all particle species and fields)
    using the openPMD standard, with defaults determined by the PIC code.
    """

    pass


@typeguard.typechecked
class DerivedAttributes(SourceBaseFilterOnly):
    """
    Aggregated derived attributes data source for openPMD output in PIConGPU.

    Enables all particle-to-grid derived attributes (e.g., density, charge) for openPMD output
    in particle-in-cell simulations, with defaults determined by the PIC code.
    """

    pass


# ---------------------------------------------------------------------------
# Sources with extra parameters
# ---------------------------------------------------------------------------


@typeguard.typechecked
class EnergyDensityCutoff(SourceBaseSpeciesFilter):
    """
    Kinetic energy density data source with cutoff for openPMD output in PIConGPU.

    Derives a scalar field of kinetic energy density (in J/m^3) for a specified particle species,
    optionally filtered, including only particles with kinetic energy below a user-defined cutoff,
    in particle-in-cell simulations. Uses weighting, momentum, and mass attributes, mapped to cells
    by the PIC code's spatial shape.
    """

    def __init__(self, species: PICMISpecies, cutoff_max_energy: typing.Union[int, float], filter: str = "species_all"):
        self.cutoff_max_energy = cutoff_max_energy
        super().__init__(species, filter)
        self.check()

    def check(
        self,
        dict_species_picmi_to_pypicongpu: typing.Optional[typing.Dict[PICMISpecies, typing.Any]] = None,
        simulation_box: typing.Optional[typing.Tuple[int, ...]] = None,
    ) -> None:
        super().check(dict_species_picmi_to_pypicongpu, simulation_box)
        if not isinstance(self.cutoff_max_energy, (int, float)):
            raise ValueError(f"cutoff_max_energy must be a number, got {type(self.cutoff_max_energy)}")
        if self.cutoff_max_energy <= 0:
            raise ValueError(f"cutoff_max_energy must be positive, got {self.cutoff_max_energy}")

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: typing.Dict[PICMISpecies, typing.Any],
        time_step_size: float = 0.0,
        num_steps: int = 0,
        simulation_box: typing.Optional[typing.Tuple[int, ...]] = None,
    ) -> PyPIConGPUEnergyDensityCutoff:
        self.check(dict_species_picmi_to_pypicongpu, simulation_box)
        mapped_species = self._map_species(dict_species_picmi_to_pypicongpu)
        return PyPIConGPUEnergyDensityCutoff(
            species=mapped_species,
            filter=self._filter,
            cutoff_max_energy=self.cutoff_max_energy,
            simulation_box=simulation_box,
        )


@typeguard.typechecked
class Momentum(SourceBaseSpeciesFilter):
    """
    Momentum component data source for openPMD output in PIConGPU.

    Derives a scalar field of momentum (in kg·m/s) in a specified direction (x, y, z)
    for a specified particle species, optionally filtered, in particle-in-cell simulations.
    Uses weighting and momentum attributes, mapped to cells by the PIC code's spatial shape.
    Intended for debugging or analyzing particle dynamics.
    """

    def __init__(self, species: PICMISpecies, filter: str = "species_all", direction: str = "x"):
        self.direction = direction
        super().__init__(species, filter)

    def check(
        self,
        dict_species_picmi_to_pypicongpu: typing.Optional[typing.Dict[PICMISpecies, typing.Any]] = None,
        simulation_box: typing.Optional[typing.Tuple[int, ...]] = None,
    ) -> None:
        super().check(dict_species_picmi_to_pypicongpu, simulation_box)
        valid_directions = ["x", "y", "z"]
        if not isinstance(self.direction, str):
            raise TypeError(f"Direction must be a string, got {type(self.direction)}")
        if self.direction not in valid_directions:
            raise ValueError(f"Direction must be 'x', 'y', or 'z', got {self.direction}")

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: typing.Dict[PICMISpecies, typing.Any],
        time_step_size: float = 0.0,
        num_steps: int = 0,
        simulation_box: typing.Optional[typing.Tuple[int, ...]] = None,
    ) -> PyPIConGPUMomentum:
        self.check(dict_species_picmi_to_pypicongpu, simulation_box)
        mapped_species = self._map_species(dict_species_picmi_to_pypicongpu)
        return PyPIConGPUMomentum(
            species=mapped_species,
            filter=self._filter,
            direction=self.direction,
            simulation_box=simulation_box,
        )


@typeguard.typechecked
class MidCurrentDensityComponent(Momentum):
    """
    Current density component data source for openPMD output in PIConGPU.

    Derives a scalar field of current density (in A/m^2) in a specified direction (x, y, z)
    for a specified particle species, optionally filtered, in particle-in-cell simulations. Uses
    weighting, position, momentum, mass, and charge attributes, mapped to cells by the PIC code's
    spatial shape. Intended for debugging (e.g., validating current solvers).
    """

    pass


@typeguard.typechecked
class MomentumDensity(Momentum):
    """
    Momentum density component data source for openPMD output in PIConGPU.

    Derives a scalar field of momentum density (in kg·m/s/m^3) in a specified direction (x, y, z)
    for a specified particle species, optionally filtered, in particle-in-cell simulations. Uses
    position and momentum attributes, mapped to cells by the PIC code's spatial shape.
    """

    pass


@typeguard.typechecked
class WeightedVelocity(Momentum):
    """
    Weighted velocity component data source for openPMD output in PIConGPU.

    Derives a scalar field of weighted velocity (in m/s) in a specified direction (x, y, z)
    for a specified particle species, optionally filtered, in particle-in-cell simulations. Uses
    position, momentum, weighting, and mass attributes, mapped to cells by the PIC code's spatial
    shape. Use with AveragedAttribute to calculate average velocity.
    """

    pass
