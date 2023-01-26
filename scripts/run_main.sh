# scripts for run FAHA
task=$2_$3_${21}
log_dir=log/${task}/${20}/
mkdir -p ${log_dir}
log=${log_dir}/s$7_te$5_ld${12}_se$6_tlr${11}_slt${13}_b${14}_ld${15}_tf${19}.log
echo "Train --source_tasks $2 --target_task $3 --n_target_samples $7 --t_lr ${11}
--lr_decay ${12} --s_lr ${13}  --t_epoch $5  --s_epoch $6  --beta ${14}  --lambda ${15}" >>$log

python lib/main.py \
        --gpu $1 \
        --source_tasks $2 \
        --target_task $3 \
        --source_model_number $4 \
        --t_epoch $5 \
        --s_epoch $6 \
        --n_target_samples $7 \
        --data_size $8 \
        --data_channel $9 \
        --resize_size ${10} \
        --t_lr ${11}  \
        --lr_decay ${12}  \
        --s_lr ${13} \
        --beta ${14} \
        --lambda ${15} \
        --model_dir ${16} \
        --batch_size ${17} \
        --print_freq ${18} \
        --apply_transform ${19} \
        --seed_id ${20} \
        --DA_type ${21} \
        >> $log

#bash scripts/run_main.sh 7 mnist usps 8 500 300 1 28 1 32 0.005 0.6 0.0003 0.1 0.5 model_v4 128 50 ODA
#bash scripts/run_main.sh 7 amazon dslr 8 100 100 1 224 3 256 0.005 0.6 0.0003 0.1 0.5 model_v4 8 50 ODA