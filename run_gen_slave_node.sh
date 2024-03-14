#!/bin/bash
#SBATCH -o /share0/jhkang/log/%j.log

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo $CUDA_VISIBLE_DEVICES
gpustat

DIST_URL="tcp://$MASTER_NODE:$MASTER_PORT"
echo "num_nodes: ${num_nodes} | current node: ${node_idx}"
echo "num_gpus: $num_gpus | DIST_URL: ${DIST_URL}"

mnt="-B /share0/jhkang:/share0/jhkang "
docker=" /share0/jhkang/simg/3d.simg "
exec_file=" sh scripts/train.sh "
machine_cfg=" -g ${num_gpus} -m ${num_nodes} -k ${node_idx}"
config_file=" -d scannetpp -c semseg-pt-v3m1-0-base-top100 -n ptv3_aidc "
dataloader_cpu=" -o 1"
dist_url="-t ${DIST_URL}" 

ml purge
ml load singularity
singularity exec --nv $mnt $docker $exec_file $config_file $machine_cfg $dist_url $dataloader_cpu
