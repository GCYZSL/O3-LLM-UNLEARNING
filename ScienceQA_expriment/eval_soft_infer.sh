BASE_MODEL="gcyzsl/O3_LLAMA2_ScienceQA"
OOD_SETTING="C"
for SCALE in 0.1
do
  for SEED in 0 1 2
  do
    for LABEL_K in "force"
    do
      OUTPUT_1="./SCALE_${SCALE}_seed_${SEED}_o_unlearn_lora_${LABEL_K}_checkpoints_5/lora_${LABEL_K}_random"
      TYPE=""
      for UNLEAN_D in "biology" "physics" "chemistry"
      do

        OUTPUT_1+="_${UNLEAN_D}_${LABEL_K}"
        TYPE+="_${UNLEAN_D}"

        TESTPATH_1="./data/scienceqa_RD_5/scienceqa_not${TYPE}_test_RD.json"
        python eval_o3.py \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --seed ${SEED} \
          --lora_weights ${OUTPUT_1} \
          --ood_type ${TYPE} \
          --ood_setting ${OOD_SETTING} \
          --ood_weights "./ood_checkpoints_scienceqa_${SEED}/"

        TESTPATH_1="./data/scienceqa_SD_5/scienceqa${TYPE}_train_SD.json"
        python eval_o3.py \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --seed ${SEED} \
          --lora_weights ${OUTPUT_1} \
          --ood_type ${TYPE} \
          --ood_setting ${OOD_SETTING} \
          --ood_weights "./ood_checkpoints_scienceqa_${SEED}/"

        TESTPATH_1="./data/scienceqa_SD_5/scienceqa${TYPE}_test_SD.json"
        python eval_o3.py \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --seed ${SEED} \
          --lora_weights ${OUTPUT_1} \
          --ood_type ${TYPE} \
          --ood_setting ${OOD_SETTING} \
          --ood_weights "./ood_checkpoints_scienceqa_${SEED}/"

        TESTPATH_1="./data/commonqa/commonqa_test.json"
        python eval_o3.py \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --seed ${SEED} \
          --lora_weights ${OUTPUT_1} \
          --ood_type ${TYPE} \
          --ood_setting ${OOD_SETTING} \
          --ood_weights "./ood_checkpoints_scienceqa_${SEED}/"

        TESTPATH_1="./data/openbookqa/openbookqa_test.json"
        python eval_o3.py \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --seed ${SEED} \
          --lora_weights ${OUTPUT_1} \
          --ood_type ${TYPE} \
          --ood_setting ${OOD_SETTING} \
          --ood_weights "./ood_checkpoints_scienceqa_${SEED}/"
      done
    done
  done
done