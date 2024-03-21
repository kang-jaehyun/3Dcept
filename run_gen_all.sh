#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
echo [SCRIPT_DIR] $SCRIPT_DIR

export num_gpus=2
export num_nodes=1
export MASTER_PORT=12359

slurm_suma_3090_big="-p big_suma_rtx3090 -q big_qos"
slurm_dell_3090_base="-p dell_rtx3090"
slurm_dell_3090_big="-p dell_rtx3090 -q big_qos"
slurm_suma_a6000_base="-p suma_a6000"
slurm_suma_a6000_big="-p suma_a6000 -q big_qos"
slurm_a100="-p suma_a100 -q a100_qos"
export SLURM_SETTING=$slurm_dell_3090_big

DURATION='3-0:0:0'
curr_date="$(date +'%m/%d-%H:%M:%S')"
JNAME="ptv3-g${num_gpus}n${num_nodes}-${curr_date}" 
echo "[MASTER] JNAME: ${JNAME} | DURATION: ${DURATION} | num_gpus: ${num_gpus}"

### Run master node
export CONFIG_FILE=" -d scannetpp -c semseg-pt-v3m1-0-base-top100 -n ptv3_aidc "
export node_idx=0
export mnt=" -B /share0/jhkang:/share0/jhkang "
export docker=" /share0/jhkang/simg/3d_20.simg "

# export exec_file=" sh scripts/train.sh "
export exec_file=" sh scripts/test.sh -w model_best"

export dataloader_cpu=" -o 1 "
export dist_url=" -t ${DIST_URL} " 

sbatch --gres=gpu:$num_gpus --cpus-per-task=8 $SLURM_SETTING -J $JNAME --time=$DURATION ./${SCRIPT_DIR}/run_gen_master_node.sh
sleep 10

### get master node address
master_node_jobID=$(squeue --name=${JNAME})
master_node_jobID=$(echo $master_node_jobID | awk '{print $9}')
echo "[NON-MASTER] $SCRIPT_DIR | MASTER-JID: ${master_node_jobID} | num_gpus: ${num_gpus} | JNAME: ${JNAME} | DURATION: ${DURATION}"

line=$(scontrol show job $master_node_jobID | grep '  NodeList=' | awk '{print $1}')
node_list=${line:9}
echo "[node_list] ${node_list}"

while [[ ${node_list} == '(null)' ]]
do
  sleep 180
  line=$(scontrol show job $master_node_jobID | grep '  NodeList=' | awk '{print $1}')
  node_list=${line:9}
  echo "[master_node_jid] ${master_node_jobID} [node_list] ${node_list}"
done

echo 'Run with master node: ' $node_list
export MASTER_NODE=$node_list
echo 'MASTER_NODE' $MASTER_NODE

### Run all non-master nodes
for i in $(seq 1 $(($num_nodes-1)))
do
  echo "node $i"
  export node_idx=$i
  sbatch --gres=gpu:$num_gpus --cpus-per-task=8 $SLURM_SETTING -J $JNAME --time=$DURATION ./${SCRIPT_DIR}/run_gen_slave_node.sh
done

