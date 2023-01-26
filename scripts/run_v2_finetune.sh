
log_dir=log/${9}/$2_$3_ft
mkdir -p ${log_dir}
log=${log_dir}/s$5_$8_$4.log
echo "Train --source_tasks $2 --target_task $3 --n_target_samples $5 --lr $8  --model_epoch $4" >>$log


python lib/main_v2_finetune.py \
  --gpu $1 \
  --source_tasks $2 \
  --target_task $3 \
  --model_epoch $4 \
  --n_target_samples $5 \
  --data_size $6 \
  --data_channel $7 \
  --lr $8 \
  --model_dir ${9} \
  --batch_size ${10} \
  --print_freq ${11} \
  >> $log

  #--d_lr $9 \
  #--hsic_lambda ${10} \
  #>>${log_dir}/../s${s}.log
  #--random_sample >>${log_dir}/../s${s}.log


# Example 
#bash scripts/run_v2_finetune.sh 7 amazon dslr 500 1 224 3 0.0001 model_v4 16 5
#bash scripts/run_v2_finetune.sh 7 amazon dslr 500 2 224 3 0.0001 model_v4 16 5
#bash scripts/run_v2_finetune.sh 7 amazon dslr 500 3 224 3 0.0001 model_v4 16 5
#bash scripts/run_v2_finetune.sh 7 amazon dslr 500 4 224 3 0.0001 model_v4 16 5
#bash scripts/run_v2_finetune.sh 7 amazon dslr 500 5 224 3 0.0001 model_v4 16 5
#bash scripts/run_v2_finetune.sh 7 amazon dslr 500 6 224 3 0.0001 model_v4 16 5
#bash scripts/run_v2_finetune.sh 7 amazon dslr 500 7 224 3 0.0001 model_v4 16 5