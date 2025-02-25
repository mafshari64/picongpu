"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.operation import NoBoundElectrons

import unittest
import typeguard

from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.species.constant import GroundStateIonization
from picongpu.pypicongpu.species.constant.ionizationmodel import BSI
from picongpu.pypicongpu.species.constant.ionizationcurrent import None_
from picongpu.pypicongpu.species.attribute import BoundElectrons


class TestNoBoundElectrons(unittest.TestCase):
    def setUp(self):
        electron = Species()
        electron.name = "e"
        self.electron = electron

        self.species1 = Species()
        self.species1.name = "ion"
        self.species1.constants = [
            GroundStateIonization(
                ionization_model_list=[BSI(ionization_electron_species=self.electron, ionization_current=None_())]
            )
        ]

    def test_no_rendering_context(self):
        """results in no rendered code, hence no rendering context available"""
        # works:
        nbe = NoBoundElectrons()
        nbe.species = self.species1
        nbe.check_preconditions()

        with self.assertRaises(RuntimeError):
            nbe.get_rendering_context()

    def test_types(self):
        """typesafety is ensured"""
        nbe = NoBoundElectrons()
        for invalid_species in ["x", 0, None, []]:
            with self.assertRaises(typeguard.TypeCheckError):
                nbe.species = invalid_species

        # works:
        nbe.species = self.species1

    def test_ionizers_required(self):
        """species must have ionizers constant"""
        nbe = NoBoundElectrons()
        nbe.species = self.species1

        self.assertTrue(self.species1.has_constant_of_type(GroundStateIonization))

        # passes
        nbe.check_preconditions()

        # remove constant:
        self.species1.constants = []

        # now raises b/c ionizers constant is missing
        with self.assertRaisesRegex(AssertionError, ".*[Gg]roundStateIonization.*"):
            nbe.check_preconditions()

    def test_empty(self):
        """species is mandatory"""
        nbe = NoBoundElectrons()
        with self.assertRaises(Exception):
            nbe.check_preconditions()

        nbe.species = self.species1
        # now works:
        nbe.check_preconditions()

    def test_bound_electrons_attr_added(self):
        """adds attribute BoundElectrons"""
        nbe = NoBoundElectrons()
        nbe.species = self.species1

        # emulate initmanager behavior
        self.species1.attributes = []
        nbe.check_preconditions()
        nbe.prebook_species_attributes()

        self.assertTrue(self.species1 in nbe.attributes_by_species)
        self.assertEqual(1, len(nbe.attributes_by_species))

        self.assertEqual(1, len(nbe.attributes_by_species[self.species1]))
        self.assertTrue(isinstance(nbe.attributes_by_species[self.species1][0], BoundElectrons))
