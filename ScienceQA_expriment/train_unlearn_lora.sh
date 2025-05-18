BASE_MODEL="gcyzsl/O3_LLAMA2_ScienceQA"
for SCALE in 0.1
do
 for SEED in 0 1 2
 do
   for LABEL_K in "force"
   do
     OUTPUT_1="./SCALE_${SCALE}_seed_${SEED}_o_unlearn_lora_${LABEL_K}_checkpoints_5/lora_${LABEL_K}_random"
     LORA_W=""
     ADAPTER=""
     TYPE=""
     for UNLEAN_D in "biology" "physics" "chemistry" "economics" "earth-science"
     do
       DATAPATH_1="./data/scienceqa_random_${LABEL_K}_5/scienceqa_${UNLEAN_D}_train_random_${LABEL_K}.json"
       OUTPUT_1+="_${UNLEAN_D}_${LABEL_K}"
       python train_unlearn_lora_o.py \
            --base_model ${BASE_MODEL} \
            --data_path ${DATAPATH_1} \
            --output_dir ${OUTPUT_1} \
            --seed ${SEED} \
            --batch_size 128 \
            --micro_batch_size 8 \
            --num_epochs 15 \
            --learning_rate 3e-4 \
            --cutoff_len 256 \
            --val_set_size 1 \
            --lora_r 8 \
            --lora_alpha 16 \
            --lora_dropout 0.05 \
            --lora_weights ${LORA_W} \
            --ood_weight "1,1" \
            --orthogonal_loss_weight ${SCALE} \
            --train_on_inputs \
            --group_by_length \
            --add_eos_token \
            --resume_from_checkpoint ${ADAPTER}
       ADAPTER=${OUTPUT_1}
       TYPE+="_${UNLEAN_D}"

       TESTPATH_1="./data/scienceqa_RD_5/scienceqa_not${TYPE}_test_RD.json"
       python eval_lora.py \
           --test_dataset ${TESTPATH_1} \
           --base_model ${BASE_MODEL} \
           --seed ${SEED} \
           --lora_weights ${OUTPUT_1}

       TESTPATH_1="./data/scienceqa_SD_5/scienceqa${TYPE}_train_SD.json"
       python eval_lora.py \
           --test_dataset ${TESTPATH_1} \
           --base_model ${BASE_MODEL} \
           --seed ${SEED} \
           --lora_weights ${OUTPUT_1}

       TESTPATH_1="./data/scienceqa_SD_5/scienceqa${TYPE}_test_SD.json"
       python eval_lora.py \
           --test_dataset ${TESTPATH_1} \
           --base_model ${BASE_MODEL} \
           --seed ${SEED} \
           --lora_weights ${OUTPUT_1}

       TESTPATH_1="./data/commonqa/commonqa_test.json"
       python eval_lora.py \
           --test_dataset ${TESTPATH_1} \
           --base_model ${BASE_MODEL} \
           --seed ${SEED} \
           --lora_weights ${OUTPUT_1}

       TESTPATH_1="./data/openbookqa/openbookqa_test.json"
       python eval_lora.py \
           --test_dataset ${TESTPATH_1} \
           --base_model ${BASE_MODEL} \
           --seed ${SEED} \
           --lora_weights ${OUTPUT_1}

       LORA_W+="${OUTPUT_1},"
     done
   done
 done
done