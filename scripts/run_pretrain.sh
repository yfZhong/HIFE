# scripts for train source models
python lib/train_source_models.py \
	--source_tasks $1 \
	--target_task $2 \
	--model_epoch $3 \
	--gpu $4 \
	--save_dir $5 \
	--lr $6 \
	--data_size $7 \
	--data_channel $8 \
	--batch_size $9 \

#Example
#mnist
#bash scripts/run_pretrain.sh mnist usps 10 0 model_v4 0.1 28 1 64
#usps
#bash scripts/run_pretrain.sh usps mnist 20 0 model_v4 0.1 28 1 64
#svhn
#bash scripts/run_pretrain.sh svhn mnist 30 0 model_v4 0.1 32 3 64


#cifar
#bash scripts/run_pretrain.sh cifar stl 50 0 model_v4 0.1 32 3 168
#stl
#bash scripts/run_pretrain.sh stl cifar 50 0 model_v4 0.1 32 3 168

#amazon
#bash scripts/run_pretrain.sh amazon webcam 30 0 model_v4 0.1 224 3 32