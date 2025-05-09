for SEED in 0 1 2
do
  for UNLEAN_D in "forget01" "forget05" "forget10"
  do
      python ./src/ood/run_ood.py \
           --unlearn_dataset "${UNLEAN_D}" \
           --ood_dataset "retain90" \
           --seed ${SEED}
  done
done