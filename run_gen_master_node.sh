#!/bin/bash
#SBATCH -o /share0/jhkang/log/%j.log

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo $CUDA_VISIBLE_DEVICES
gpustat

MASTER_NODE=$SLURM_NODELIST # sbatch output environment 사용
DIST_URL="tcp://$MASTER_NODE:$MASTER_PORT"
echo "num_nodes: ${num_nodes} | current node: ${node_idx}"
echo "num_gpus: $num_gpus | DIST_URL: ${DIST_URL}"
dist_url=" -t ${DIST_URL} " 
machine_cfg=" -g ${num_gpus} -m ${num_nodes} -k ${node_idx} "

ml purge
ml load singularity
singularity exec --nv $mnt $docker $exec_file $CONFIG_FILE $machine_cfg $dist_url $dataloader_cpu
