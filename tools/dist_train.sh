# export NCCL_SOCKET_IFNAME=ib0
NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
# [ -z "$NCCL_IB_HCA"] && NCCL_IB_HCA=mlx4_1;
export NCCL_IB_HCA
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106

nnodes=$1    ##节点数量
node_rank=$2 ##节点优先级
master_addr=$3 ##主进程IP -> 进 Node1 ifconfig 得到 IP
nproc_per_node=$4 ##每个节点多少张 GPU

config_path=$5
launcher=$6
resume_from=${7}

echo $nnodes $node_rank $master_addr $nproc_per_node
echo $config_path $launcher $resume_from

python3 -m torch.distributed.launch \
	--nproc_per_node=$nproc_per_node \
	--nnodes=$nnodes \
	--node_rank=$node_rank \
	--master_addr=$master_addr \
	./tools/train.py $config_path \
	--launcher $launcher

