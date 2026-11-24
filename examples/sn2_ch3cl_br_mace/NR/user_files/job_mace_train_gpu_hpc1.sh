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
# You must keep the name file as job_mace_train_ARCHTYPE_myHPCkeyword.sh.
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
#SBATCH -o MACE_TRAIN.%j
#SBATCH -e MACE_TRAIN.%j
# Name of job
#SBATCH -J MACE_TRAIN
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
#

#----------------------------------------------
# Files / Variables - They should not be changed
#----------------------------------------------

MACE_FONDATION_FILE="_R_MACE_FONDATION_FILE_" 
MACE_IN_FILE="_R_MACE_INPUT_FILE_"
MACE_LOG_FILE="_R_MACE_LOG_FILE_"
MACE_OUT_FILE="_R_MACE_OUTPUT_FILE_"
MACE_DATA_DIR="../data"

MACE_INSTALL="" #If you don't want to use a specific version of MACE, but rather your own you installed on a conda env, specify the path of this env here
#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Load the environment
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
[ -f "${MACE_IN_FILE}" ] || { echo "${MACE_IN_FILE} does not exist. Aborting..."; exit 1; }
[ -d ${MACE_DATA_DIR} ] || { echo "${MACE_DATA_DIR} does not exist. Aborting..."; exit 1; }

# Set the temporary work directory
export TEMPWORKDIR=${SCRATCH}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}/JOB-${SLURM_JOBID}"

# Copy or link the necessary files 
cp "${MACE_IN_FILE}" "${TEMPWORKDIR}" && echo "${MACE_IN_FILE} copied successfully"
mkdir -p "${TEMPWORKDIR}/data" || { echo "Could not create ${TEMPWORKDIR}/data. Aborting..."; exit 1; }
{ cp -r ${MACE_DATA_DIR}/* "${TEMPWORKDIR}/data" && echo "${MACE_DATA_DIR} copied successfully"; } || { echo "Could not copy ${MACE_DATA_DIR}. Aborting..."; exit 1; }

if [ -z "${MACE_FONDATION_FILE}" ]; then
    echo "No foundation model needed. Skipping link."
elif [ -f ${MACE_FONDATION_FILE} ]; then
    { ln -s "$(realpath "${MACE_FONDATION_FILE}")" "${SLURM_SUBMIT_DIR}" && echo "${MACE_FONDATION_FILE} linked successfully"; } || { echo "Could not link ${MACE_FONDATION_FILE}. Aborting..."; exit 1; }
else
    echo "${MACE_FONDATION_FILE} does not exist. Aborting..."
    exit 1
fi

# Go to the temporary work directory
cd "${TEMPWORKDIR}" || { echo "Could not go to ${TEMPWORKDIR}. Aborting..."; exit 1; }

# Run the MACE train
echo "# [$(date)] Running MACE train..."
mace_run_train --config=${MACE_IN_FILE} 1> ${MACE_LOG_FILE} 2> ${MACE_OUT_FILE} 
echo "# [$(date)] MACE train finished."

# This are useless files, so we remove them
if [ -f out.json ]; then rm out.json; fi
if [ -f input_v2_compat.json ]; then rm input_v2_compat.json; fi

# Move back data from the temporary work directory and scratch, and clean-up
rm -rf "${TEMPWORKDIR}/data"
mv ./* "${SLURM_SUBMIT_DIR}"
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

# Done
echo "Have a nice day !"

sleep 2
exit
