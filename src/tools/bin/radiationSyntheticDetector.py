#!/usr/bin/env python
#
# Copyright 2013-2024 Richard Pausch
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

import os
import numpy as np
from matplotlib import pyplot as plt
import argparse
from matplotlib.colors import LogNorm

__doc__ = """This tool analyzes spectra for different directions.
             By giving the data files to analyze, a statistic
             analysis is given for the selected region."""

parser = argparse.ArgumentParser(description=__doc__, epilog="For further questions please contact Richard Pausch.")

# Path to input files (several are possible) - necesairy argument
parser.add_argument(
    "path2Data",
    metavar="Path to input file",
    type=argparse.FileType("r"),
    nargs="+",
    help="""location of input file (giving several
                            files is possible)""",
)

# extend of the input data
parser.add_argument(
    "--dataExtend",
    "-d",
    nargs=4,
    metavar="float",
    type=float,
    default=(0.5, 3.5, 0.0, 360.0),
    help="""data extend as given in the input file:
                            omega_min, omega_max, theta_min, theta_max
                            [default=0.5 3.5 0.0 360.0]""",
)

# extend of the data to analyze
parser.add_argument(
    "--analyzerExtend",
    "-a",
    nargs=4,
    metavar="float",
    type=float,
    default=[],
    help="""region to be analysed : omega_min,
                            omega_max, theta_min, theta_max
                            [default= as dataExtend]""",
)

# use window output to show selected area
parser.add_argument(
    "--vis",
    dest="visual",
    action="store_true",
    help="""show data in logscale and selected
                            area for first data set""",
)

# get arguments and store them in args
args = parser.parse_args()

# flag to only plot first dataset and ignore further files
first = True

# do for each file given on the command line
for myfile in args.path2Data:
    # load data
    data = np.loadtxt(myfile)

    # set extent data:
    omega_min_data = args.dataExtend[0]
    omega_max_data = args.dataExtend[1]
    theta_min_data = args.dataExtend[2]
    theta_max_data = args.dataExtend[3]

    # set extent analyzer:
    omega_min_analy = args.analyzerExtend[0]
    omega_max_analy = args.analyzerExtend[1]
    theta_min_analy = args.analyzerExtend[2]
    theta_max_analy = args.analyzerExtend[3]

    # compute indeces range that is used for analysis
    omega_max_index = int((omega_max_analy - omega_min_data) / (omega_max_data - omega_min_data) * data.shape[1])
    omega_min_index = int((omega_min_analy - omega_min_data) / (omega_max_data - omega_min_data) * data.shape[1])
    theta_max_index = int((theta_max_analy - theta_min_data) / (theta_max_data - theta_min_data) * data.shape[0])
    theta_min_index = int((theta_min_analy - theta_min_data) / (theta_max_data - theta_min_data) * data.shape[0])

    # plot data and selected area
    if first and args.visual:
        # generate window
        plt.figure("Radiation Analysing Tool: selected data region")

        # set up supplot
        plt.subplot(
            111,
            autoscale_on=False,
            xlim=(omega_min_data, omega_max_data),
            ylim=(theta_min_data, theta_max_data),
        )

        # colorplot
        plt.imshow(
            data,
            extent=(omega_min_data, omega_max_data, theta_min_data, theta_max_data),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            norm=LogNorm(),
        )

        # selected region
        plt.axhspan(theta_min_analy, theta_max_analy, facecolor="black", alpha=0.3)
        plt.axvspan(omega_min_analy, omega_max_analy, facecolor="black", alpha=0.3)

        # labels
        fontsize = 16
        plt.xlabel(r"$\omega$", fontsize=fontsize)
        plt.ylabel(r"$\theta$", fontsize=fontsize)
        plt.show()

        # only first data set, therefore switch boolean flag:
        first = False

    # verbose user output to inform on loading progress and statistics
    print("file loaded:              {}".format(os.path.basename(myfile.name)))

    # regions:
    print("shape input data:         {}".format(np.shape(data)))
    print("range selected:           {}".format((omega_min_index, omega_max_index, theta_min_index, theta_max_index)))

    selectedOnly = data[theta_min_index:theta_max_index, omega_min_index:omega_max_index]

    print("shape of data analyzed:   {}".format(np.shape(selectedOnly)))
    print("")

    # statistics:
    my_sum = np.sum(selectedOnly)
    my_avg = np.average(selectedOnly)
    my_std = np.std(selectedOnly)
    my_size = np.size(selectedOnly)

    print("sum:         {}".format(my_sum))
    print("average:     {}".format(my_avg))
    print("std:         {}".format(my_std))
    print("")
    print("sum + error: {} +/- {}".format(my_size * my_avg, np.sqrt(my_size) * my_std))

    # finalize
    print("")
    print("")
