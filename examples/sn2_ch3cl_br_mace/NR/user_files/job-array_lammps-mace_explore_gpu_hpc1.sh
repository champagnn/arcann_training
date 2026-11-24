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
# You must keep the name file as job-array_lammps-mace_explore_ARCHTYPE_myHPCkeyword.sh.
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
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
# Walltime
#SBATCH -t _R_WALLTIME_
# Merge Output/Error
#SBATCH -o LAMMPS_MACE.%A_%a
#SBATCH -e LAMMPS_MACE.%A_%a
# Name of job
#SBATCH -J LAMMPS_MACE
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
#
#SBATCH --array=_R_ARRAY_START_-_R_ARRAY_END_%300
#

#----------------------------------------------
# This part use the job-array-params_lammps-mace_explore_ARCHTYPE_myHPCkeyword.lst created
# Don't forget to replace the array_line with the correct name of the file (namely ARCHTYPE and myHPCkeyword)
# The rest should not be changed
#----------------------------------------------

SLURM_ARRAY_TASK_ID_LINE=$((SLURM_ARRAY_TASK_ID + 2))
array_line=$(sed -n "${SLURM_ARRAY_TASK_ID_LINE}p" "job-array-params_lammps-mace_explore_ARCHTYPE_myHPCkeyword.lst")
IFS='/' read -ra array_param <<< "${array_line}"

JOB_PATH=${array_param[0]}
JOB_PATH="${JOB_PATH%_*}/${JOB_PATH##*_}"
JOB_PATH="${JOB_PATH%_*}/${JOB_PATH##*_}"

MACE_MODEL_VERSION=${array_param[1]}
IFS='" "' read -r -a MACE_MODEL_FILES <<< "${array_param[2]}"
LAMMPS_IN_FILE=${array_param[3]}
LAMMPS_LOG_FILE=${LAMMPS_IN_FILE/.in/.log}
LAMMPS_OUT_FILE=${LAMMPS_IN_FILE/.in/.out}
EXTRA_FILES=()
EXTRA_FILES+=("${array_param[4]}")
if [ -n "${array_param[5]}" ]; then
    EXTRA_FILES+=("${array_param[5]}")
fi
if [ -n "${array_param[6]}" ]; then
    IFS='" "' read -r -a PLUMED_FILES <<< "${array_param[6]}"
    EXTRA_FILES+=("${PLUMED_FILES[@]}")
fi

echo "--- DEBUG: MACE Files found: ${MACE_MODEL_FILES[*]}"
echo "--- DEBUG: LAMMPS Input: ${LAMMPS_IN_FILE}"

#----------------------------------------------
# Adapt the following lines to your HPC system
# It should be the close to the job_lammps-mace_explore_ARCHTYPE_myHPCkeyword.sh
#----------------------------------------------

# Load the environment
if [ -n "$MACE_INSTALL" ]; then
    # Activate the conda environment
    module load gcc cuda openmpi
    LAMMPS_EXE="${MACE_INSTALL}/lammps-install/lmp"

elif [ ${MACE_MODEL_VERSION} == "0.3.14" ]; then
    # Load the LAMMPS module build with MACE
    module load lammps_mace
    LAMMPS_EXE=$(command -v lmp) || { echo "Executable (lmp) not found. Aborting..."; exit 1 ; }
else
    echo "MACE version ${MACE_MODEL_VERSION} is not available. Aborting..."
    exit 1
fi

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}/${JOB_PATH}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}/${JOB_PATH}. Aborting..."; exit 1; }

# Check
[ -f "${LAMMPS_IN_FILE}" ] || { echo "${LAMMPS_IN_FILE} does not exist. Aborting..."; exit 1; }

# Set the temporary work directory
export TEMPWORKDIR=${SCRATCH}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}/${JOB_PATH}/JOB-${SLURM_JOBID}"

# Copy or link the necessary files 
cp "${LAMMPS_IN_FILE}" "${TEMPWORKDIR}" && echo "${LAMMPS_IN_FILE} copied successfully"
for f in "${MACE_MODEL_FILES[@]}"; do [ -f "${f}" ] && ln -s "$(realpath "${f}")" "${TEMPWORKDIR}" && echo "${f} linked successfully"; done
for f in "${EXTRA_FILES[@]}"; do [ -f "${f}" ] && cp "${f}" "${TEMPWORKDIR}" && echo "${f} copied successfully"; done

# Go to the temporary work directory
cd "${TEMPWORKDIR}" || { echo "Could not go to ${TEMPWORKDIR}. Aborting..."; exit 1; }

# Run the Lammps exploration (here with symmetrix)
echo "# [$(date)] Running LAMMPS..."
${LAMMPS_EXE} -k on g 1 -sf kk -pk kokkos newton on neigh half -in "${LAMMPS_IN_FILE}" -log "${LAMMPS_LOG_FILE}" -screen none > "${LAMMPS_OUT_FILE}" 2>&1
echo "# [$(date)] LAMMPS finished."

# Move back data from the temporary work directory and scratch, and clean-up
if [ -f log.cite ]; then rm log.cite ; fi
find ./ -type l -delete
mv ./* "${SLURM_SUBMIT_DIR}/${JOB_PATH}"
cd "${SLURM_SUBMIT_DIR}/${JOB_PATH}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}/${JOB_PATH}. Aborting..."; exit 1; }
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

# Done
echo "Have a nice day !"

sleep 2
exit