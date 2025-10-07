"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

# from picongpu.picmi.copy_attributes import default_converts_to


# def diagnostic_converts_to(*args, **kwargs):
#     kwargs["conversions"] = {
#         "species": lambda self, *args, **kwargs: kwargs["dict_species_picmi_to_pypicongpu"].get(self.species)
#     } | kwargs.get("conversions", {})
#     return default_converts_to(*args, **kwargs)

from picongpu.picmi.copy_attributes import default_converts_to, has_attribute


def diagnostic_converts_to(*args, **kwargs):
    """
    Generalized diagnostic_converts_to:
    - Only adds special conversions for attributes that exist in the PICMI class.
    - Other attributes are copied automatically via default_converts_to.
    """
    cls = args[0] if args else None
    conversions = kwargs.get("conversions", {}).copy()

    if cls and has_attribute(cls, "species"):
        # only inject species mapping if it exists
        conversions["species"] = lambda self, *a, **kw: kw["dict_species_picmi_to_pypicongpu"].get(self.species)

    kwargs["conversions"] = conversions
    return default_converts_to(*args, **kwargs)
