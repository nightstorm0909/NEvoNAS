# bash train_imagenet.sh batch_size genotype_dir dataset_dir saved_model.pt

python ./s1/test_imagenet.py --batch_size $1 --data $3 --dir $2 --auxiliary --model_path $4 --report_freq 100
