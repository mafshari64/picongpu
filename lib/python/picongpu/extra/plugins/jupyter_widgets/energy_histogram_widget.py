"""
This file is part of PIConGPU.

Copyright 2017-2024 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""

from .base_widget import BaseWidget
from ..plot_mpl import EnergyHistogramMPL

from ipywidgets import widgets


class EnergyHistogramWidget(BaseWidget):
    """
    From within jupyter notebook this widget can be used in the following way:

      %matplotlib widget
      import matplotlib.pyplot as plt
      plt.ioff() # deactivating instant plotting is necessary!

      from picongpu.extra.plugins.jupyter_widgets import EnergyHistogramWidget

      display(EnergyHistogramWidget(run_dir_options="path/to/outputs"))
    """

    def __init__(self, run_dir_options, fig=None, output_widget=None, **kwargs):
        BaseWidget.__init__(self, EnergyHistogramMPL, run_dir_options, fig, output_widget, **kwargs)

    def _create_widgets_for_vis_args(self):
        """
        Create the widgets that are necessary for adjusting the
        visualization parameters of this special visualizer.

        Returns
        -------
        a dict mapping keyword argument names of the PIC visualizer
        to the jupyter widgets responsible for adjusting those values.
        """
        self.species = widgets.Dropdown(description="Species", options=["e"], value="e")
        self.species_filter = widgets.Dropdown(description="Species_filter", options=["all"], value="all")

        return {"species": self.species, "species_filter": self.species_filter}
