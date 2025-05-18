import json
import pandas

SCALE=str(0.1)
writer = pandas.ExcelWriter("all_results_scale_new_ood_01_retain.xlsx", engine='xlsxwriter')
sheet_name = "unlearn"
sheet = writer.book.add_worksheet(sheet_name)

row = 0 #
for LABEL_K in ["force"]:
    for SEED in [0, 1, 2]:
        OUTPUT_1 = f"./SCALE_{SCALE}_seed_{SEED}_o_unlearn_lora_{LABEL_K}_checkpoints_5/test_noretain_C_seed{SEED}_oodlora_lora_{LABEL_K}_random"
        TYPE=""
        results = []
        for UNLEAN_D in ["biology", "physics", "chemistry"]:
            OUTPUT_1 += f"_{UNLEAN_D}_{LABEL_K}"
            TYPE += f"_{UNLEAN_D}"

            f_name=f'{OUTPUT_1}_scienceqa_not{TYPE}_test_RD.json'
            with open(f_name, 'r') as f:
                RD = json.load(f)['acc']

            f_name = f'{OUTPUT_1}_scienceqa{TYPE}_train_SD.json'
            with open(f_name, 'r') as f:
                SU = json.load(f)['acc']

            f_name = f'{OUTPUT_1}_scienceqa{TYPE}_test_SD.json'
            with open(f_name, 'r') as f:
                DU = json.load(f)['acc']

            f_name = f'{OUTPUT_1}_commonqa_test.json'
            with open(f_name, 'r') as f:
                CQA = json.load(f)['acc']

            f_name = f'{OUTPUT_1}_openbookqa_test.json'
            with open(f_name, 'r') as f:
                OQA = json.load(f)['acc']

            print("SEED: ",  SEED, "LABEL_K: ", LABEL_K, "Stage: ", TYPE, "***********************")
            # print("SU, DU, RD, Common, Open ", SU, DU, RD, CQA, OQA)
            print("SU, DU, RD, Common, Open ", f"{SU},{DU},{RD},{CQA},{OQA}")

            sheet.write(row, 0, f"SEED: {SEED}")  # Adjust row index for clarity
            sheet.write(row, 1, f"LABEL_K: {LABEL_K}")
            sheet.write(row, 2, f"Stage: {TYPE}")

            results += [SU,DU,RD,CQA,OQA,""]

            row += 1
        for i in range(len(results)):
            sheet.write(row, i, results[i])
        row += 1

writer.close()
