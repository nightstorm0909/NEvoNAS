#!/bin/bash
# bash ./s1/arch_search_test1.sh cifar10 gpu outputs data_path 

echo script name: $0
echo $# arguments

config_path="./s2/configs/CMAES-NAS.config"
config_root="./s2/configs"

dataset=$1
gpu=$2
output_dir=$3
data_path=$4
epochs=50
train_epochs=0
train_discrete=0
init_channel=16
layers=8
knn=5
novelty_threshold=0.0
mutate_rate=0.2
pop_size=20

batch_size=64
valid_batch_size=1024

if [ $train_discrete -gt 0 ]
then
  python ./s1/arch_search.py --gpu ${gpu} --init_channel ${init_channel} --layers ${layers} --dataset ${dataset} --data_dir ${data_path} --output_dir ${output_dir} \
                                 --epochs ${epochs} --train_epochs ${train_epochs} --knn ${knn} --config_path ${config_path} --config_root ${config_root} \
                                 --mutate_rate ${mutate_rate} --pop_size ${pop_size} --train_discrete
else
  python ./s1/arch_search.py --gpu ${gpu} --init_channel ${init_channel} --layers ${layers} --dataset ${dataset} --data_dir ${data_path} --output_dir ${output_dir} \
                                 --epochs ${epochs} --train_epochs ${train_epochs} --knn ${knn} --config_path ${config_path} --config_root ${config_root} \
                                 --mutate_rate ${mutate_rate} --pop_size ${pop_size}
fi
