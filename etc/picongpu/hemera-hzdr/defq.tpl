#!/usr/bin/env bash
# Copyright 2013-2024 Axel Huebl, Richard Pausch, Rene Widera
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


# PIConGPU batch script for hemera' SLURM batch system

#SBATCH --partition=!TBG_queue
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes
#SBATCH --ntasks=!TBG_tasks
#SBATCH --ntasks-per-node=!TBG_mpiTasksPerNode
#SBATCH --cpus-per-task=!TBG_coresPerCPU
#SBATCH --mem=!TBG_memPerNode
# must be 1 else we can not use hyperthreading
#SBATCH --threads-per-core=1
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --chdir=!TBG_dstPath
# notify the job 240 sec before the wall time ends
#SBATCH --signal=B:SIGALRM@240
#!TBG_keepOutputFileOpen

#SBATCH -o stdout
#SBATCH -e stderr


## calculations will be performed by tbg ##
.TBG_queue="defq"

.TBG_queue=${TBG_partition:-"defq"}
.TBG_account=`if [ $TBG_partition == "defq_low" ] ; then echo "low"; else echo "defq"; fi`
# configure if the output file should be appended or overwritten
.TBG_keepOutputFileOpen=`if [ $TBG_partition == "defq_low" ] ; then echo "SBATCH --open-mode=append"; fi`

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"NONE"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

# number of available/hosted devices per node in the system
.TBG_numHostedDevicesPerNode=2

# 2 devices per node
.TBG_devicesPerNode=`if [ $TBG_tasks -gt $TBG_numHostedDevicesPerNode ] ; then echo $TBG_numHostedDevicesPerNode; else echo $TBG_tasks; fi`

# host memory per device
.TBG_memPerCPU="$((360000 / $TBG_numHostedDevicesPerNode))"
# host memory per node
.TBG_memPerNode="$((TBG_memPerCPU * TBG_devicesPerNode))"

# number of cores to block per device - we got 20 cpus per device
.TBG_coresPerCPU=20

# We only start 1 MPI task per device
.TBG_mpiTasksPerNode="$(( TBG_devicesPerNode * 1 ))"

# use ceil to caculate nodes
.TBG_nodes="$((( TBG_tasks + TBG_devicesPerNode - 1 ) / TBG_devicesPerNode))"

## end calculations ##

echo 'Running program...'

cd !TBG_dstPath

export MODULES_NO_OUTPUT=1
source !TBG_profile
if [ $? -ne 0 ] ; then
  echo "Error: PIConGPU environment profile under \"!TBG_profile\" not found!"
  exit 1
fi
unset MODULES_NO_OUTPUT

#set user rights to u=rwx;g=r-x;o=---
umask 0027

mkdir simOutput 2> /dev/null
cd simOutput
ln -s ../stdout output

if [ $? -eq 0 ] ; then
  # Run PIConGPU
  source !TBG_dstPath/tbg/handleSlurmSignals.sh mpiexec -np !TBG_tasks --bind-to none !TBG_dstPath/tbg/cpuNumaStarter.sh \
    !TBG_dstPath/input/bin/picongpu !TBG_author !TBG_programParams
fi
