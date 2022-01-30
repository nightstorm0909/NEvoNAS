#!/bin/bash
# bash ./s2/arch_search.sh cifar10 gpu outputs api_path data_path

echo script name: $0
echo $# arguments

dataset=$1
gpu=$2
channel=16
num_cells=5
max_nodes=4
output_dir=$3

#if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
#  data_path="../../../data"
#else
#  data_path="../../../data/ImageNet16"
#fi
#api_path="../../NAS-Bench-201-v1_1-096897.pth"
api_path=$4
data_path=$5
config_path="./s2/configs/CMAES-NAS.config"
config_root="./s2/configs"
#nas_config="./configs/s2_configs.cfg"
#nas_config=$4
record_filename=info.csv

epochs=50
train_epochs=0
train_discrete=0
local_fitness_flag=0
knn=5
novelty_threshold=0.0
mutate_rate=0.2
pop_size=20

if [ $local_fitness_flag -gt 0 ]
  then
  if [ $train_discrete -gt 0 ]
  then
    for index in {1..3..1}
    do
      python ./s2/arch_search.py --gpu ${gpu} --max_nodes ${max_nodes} --init_channel ${channel} --num_cells ${num_cells} \
                                     --dataset ${dataset} --data ${data_path} --output_dir ${output_dir} --record_filename ${record_filename} \
                                     --api_path ${api_path} --config_path ${config_path} --config_root ${config_root} \
                                     --epochs ${epochs} --train_epochs ${train_epochs} --knn ${knn} \
                                     --mutate_rate ${mutate_rate} --pop_size ${pop_size} --train_discrete --local_fitness_flag
    done
  else
    for index in {1..3..1}
    do
      python ./s2/arch_search.py --gpu ${gpu} --max_nodes ${max_nodes} --init_channel ${channel} --num_cells ${num_cells} \
                                     --dataset ${dataset} --data ${data_path} --output_dir ${output_dir} --record_filename ${record_filename} \
                                     --api_path ${api_path} --config_path ${config_path} --config_root ${config_root} \
                                     --epochs ${epochs} --train_epochs ${train_epochs} --knn ${knn} \
                                     --mutate_rate ${mutate_rate} --pop_size ${pop_size} --local_fitness_flag
    done
  fi
else
  if [ $train_discrete -gt 0 ]
  then
    for index in {1..3..1}
    do
      python ./s2/arch_search.py --gpu ${gpu} --max_nodes ${max_nodes} --init_channel ${channel} --num_cells ${num_cells} \
                                     --dataset ${dataset} --data ${data_path} --output_dir ${output_dir} --record_filename ${record_filename} \
                                     --api_path ${api_path} --config_path ${config_path} --config_root ${config_root} \
                                     --epochs ${epochs} --train_epochs ${train_epochs} --knn ${knn} \
                                     --mutate_rate ${mutate_rate} --pop_size ${pop_size} --train_discrete
    done
  else
    for index in {1..3..1}
    do
      python ./s2/arch_search.py --gpu ${gpu} --max_nodes ${max_nodes} --init_channel ${channel} --num_cells ${num_cells} \
                                     --dataset ${dataset} --data ${data_path} --output_dir ${output_dir} --record_filename ${record_filename} \
                                     --api_path ${api_path} --config_path ${config_path} --config_root ${config_root} \
                                     --epochs ${epochs} --train_epochs ${train_epochs} --knn ${knn} \
                                     --mutate_rate ${mutate_rate} --pop_size ${pop_size}
    done
  fi
fi
