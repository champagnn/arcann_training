#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2026/02/25
# Last modified: 2026/11/19
#----------------------------------------------
# You must keep the _R_VARIABLES_ in the file.
# You must keep the name file as job_mace_compress_ARCHTYPE_myHPCkeyword.sh.
#----------------------------------------------
# Project/Account
#SBATCH --account=_R_PROJECT_@_R_ALLOC_
# QoS/Partition/SubPartition
#SBATCH --qos=_R_QOS_
#SBATCH --partition=_R_PARTITION_
#SBATCH -C _R_SUBPARTITION_
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
# Walltime
#SBATCH -t _R_WALLTIME_
# Merge Output/Error
#SBATCH -o MACE_COMPRESS.%j
#SBATCH -e MACE_COMPRESS.%j
# Name of job
#SBATCH -J MACE_COMPRESS
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
#

#----------------------------------------------
# Files / Variables - They should not be changed
#----------------------------------------------

MACE_MODEL_VERSION="_R_MACE_VERSION_"
MACE_MODEL_FILE="_R_MACE_MODEL_FILE_"
MACE_MODEL_STYLE="_R_MACE_MODEL_STYLE_"

MACE_INSTALL="" #If you don't want to use a specific version of MACE, but rather your own you installed on a conda env, specify the path of this env here

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Load the environement
if [ -n "$MACE_INSTALL" ]; then
    # Activate the conda environment
    module load conda
    source activate ${MACE_INSTALL}
elif [ ${MACE_MODEL_VERSION} == "0.3.14" ]; then
    # Load the MACE module
    module load mace
else
    echo "MACE version ${MACE_MODEL_VERSION} is not available. Aborting..."
    exit 1
fi

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

# Check
[ -f ${MACE_MODEL_FILE} ] || { echo "${MACE_MODEL_FILE} does not exist. Aborting..."; exit 1; }

if [ "${MACE_MODEL_STYLE}" == "symmetrix" ]; then
    echo "# [$(date)] Converting model to CPU for symmetrix"
    mace_convert_device "${MACE_MODEL_FILE}" --target_device cpu --output_file "${MACE_MODEL_FILE}.cpu"
    echo "# [$(date)] Converting mace model to symmetrix"
    symmetrix_extract_mace --model "${MACE_MODEL_FILE}.cpu" --atomic-numbers _R_ATOMIC_NUMBERS_ --output "${MACE_MODEL_FILE}.json"
elif [ "${MACE_MODEL_STYLE}" == "mace" ]; then
    echo "# [$(date)] Converting mace model to LAMMPS libtorch"
    mace_create_lammps_model "${MACE_MODEL_FILE}" --format libtorch
elif [ "${MACE_MODEL_STYLE}" == "mliap" ]; then
    # cuequivariance is required to convert to mliap
    echo "# [$(date)] Converting mace model to LAMMPS MLIAP"
    mace_create_lammps_model "${MACE_MODEL_FILE}" --format mliap
fi
# done

sleep 5
exit 0
