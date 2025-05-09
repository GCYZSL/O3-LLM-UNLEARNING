import argparse
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
# sys.path.append("src/ood")
from .ood_utils import set_seed, collate_fn_mlm, collate_fn, detection_performance
from datasets import load_metric
from sklearn import svm
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import os
from .ood_model_selector import RobertaForSelector
import json
import warnings
import random
from .ood_data import load
import pickle
import math

warnings.filterwarnings("ignore")
torch.set_num_threads(10)

def range_print(x):
    print(np.min(x), np.mean(x), np.max(x))

def train(args, model, train_dataset, test_dataset, benchmarks):
    # train_dataloader_mlm = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn_mlm, shuffle=True, drop_last=True)
    train_dataloader_ran = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn_mlm,
                                      shuffle=False, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False,
                                  drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    acc_g = 0

    def detect_ood(acc_global):
        mean_list, precision_list, fea_list = model.sample_X_estimator(train_dataloader)

        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

        test_mah_vanlia = model.get_unsup_Mah_score(test_dataloader, mean_list, precision_list, fea_list)[:, 1:]
        train_mah_vanlia = model.get_unsup_Mah_score(train_dataloader, mean_list, precision_list, fea_list)[:, 1:]

        for _, ood_dataset in benchmarks:
            ood_dataloader = DataLoader(ood_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
            ood_mah_vanlia = model.get_unsup_Mah_score(ood_dataloader, mean_list, precision_list, fea_list)[:, 1:]

            ood_labels = np.ones(shape=(ood_mah_vanlia.shape[0],))
            test_labels = np.zeros(shape=(test_mah_vanlia.shape[0],))

            test_mah_scores = test_mah_vanlia
            ood_mah_scores = ood_mah_vanlia
            train_mah_scores = train_mah_vanlia

            np.random.shuffle(test_mah_scores)
            np.random.shuffle(ood_mah_scores)
            best_ours_AUROC = 0.0

            if args.ood == 'ocsvm':
                c_lr = svm.OneClassSVM(nu=0.01, kernel='linear', degree=2)
                c_lr.fit(train_mah_scores)
                test_scores = c_lr.decision_function(test_mah_scores)
                ood_scores = c_lr.decision_function(ood_mah_scores)
                train_scores = c_lr.decision_function(train_mah_scores)
                X_scores = np.concatenate((ood_scores, test_scores))
                Y_test = np.concatenate((ood_labels, test_labels))
                results = detection_performance(X_scores, Y_test, 'mah_logs', tag='TMP')
                neg_resuls = detection_performance(-X_scores, Y_test, 'feats_logs', tag='TMP')
                if sum(results["TMP"].values()) < sum(neg_resuls["TMP"].values()):
                    results = neg_resuls
                # if results['TMP']['AUROC'] > best_ours_AUROC:
                best_ours_AUROC = results['TMP']['AUROC']
                print(args.ood + ' current auroc: {:.3f}'.format(best_ours_AUROC))
                threshold = np.max(train_scores) # 99% of the training set as the threshold
                # print(threshold)

                range_print(train_scores)
                range_print(test_scores)
                range_print(ood_scores)

                test_labels_prediction = (test_scores <= threshold).astype(int)
                ood_labels_prediction = (ood_scores <= threshold).astype(int)
                Y_predict = np.concatenate((ood_labels_prediction, test_labels_prediction))
                acc = (Y_predict == Y_test).mean()

                gmm_w, x0 = weighting_func_gmm(train_scores, test_scores)
                ood_fit_index = random.sample(range(len(ood_scores)), int(0.5 * len(ood_scores)))
                ood_fit_scores = ood_scores[ood_fit_index]
                ood_fit_scores = ood_fit_scores.reshape(-1, 1)
                # gmm_retain = weighting_func_gmm_retain(ood_fit_scores)
                gmm_retain = None

                w_res_in_train = [obtain_weights(x, gmm_w, gmm_retain, x0) for x in train_scores]
                w_res_in_test = [obtain_weights(x, gmm_w, gmm_retain, x0) for x in test_scores]
                w_res_ood = [obtain_weights(x, gmm_w, gmm_retain, x0) for x in ood_scores]

                print(np.mean(np.array(w_res_in_train)), np.mean(np.array(w_res_in_test)), np.mean(np.array(w_res_ood)))

                ood_path = "./ood_checkpoints"
                if not os.path.exists(ood_path):
                    os.mkdir(ood_path)
                torch.save(mean_list, f"{ood_path}/{args.unlearn_dataset}_mean_list.pt")
                torch.save(precision_list, f"{ood_path}/{args.unlearn_dataset}_precision_list.pt")
                torch.save(fea_list, f"{ood_path}/{args.unlearn_dataset}_fea_list.pt")

                with open(f"{ood_path}/{args.unlearn_dataset}_ocsvm.pkl", "wb") as output_file:
                    pickle.dump(c_lr, output_file)
                with open(f"{ood_path}/{args.unlearn_dataset}_ocsvm_threshold.json", 'w') as f:
                    json.dump([threshold, x0], f)
                with open(f"{ood_path}/{args.unlearn_dataset}_gmm_w.pkl", "wb") as output_file:
                    pickle.dump(gmm_w, output_file)
                print("SAVE", "CURRENT BEST ACC: ", acc)
                acc_global = acc
                model.roberta.save_pretrained(f"{ood_path}/{args.unlearn_dataset}_roberta")
                return acc_global

    num_steps = 0
    acc_g = detect_ood(acc_g)
    mlm_loss_avg = []
    for epoch in range(int(args.num_train_epochs)):
        print("start training")
        model.zero_grad()
        for batch_ran, batch in zip(tqdm(train_dataloader_ran), tqdm(train_dataloader)):
            model.train()
            batch_ran = {key: value.to(args.device) for key, value in batch_ran.items()}
            batch = {key: value.to(args.device) for key, value in batch.items()}
            outputs = model(batch_ran, batch, num_steps, train_dataloader)
            mlm_loss, moco_loss = outputs
            loss = moco_loss + mlm_loss
            # loss = moco_loss
            loss.backward()
            num_steps += 1
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            # print('Step:', num_steps, 'mlm_loss: ', mlm_loss.item(), 'moco_loss: ', moco_loss.item())
        acc_g = detect_ood(acc_g)
        # print("Epoch Accuracy: ", acc_g)


def weighting_func_gmm(train_in_score, test_in_score):
    # 1. fit two gaussians
    mean1, std1 = norm.fit(train_in_score)
    mean2, std2 = norm.fit(test_in_score)

    # 2. build the gaussian mixture model
    gmm = GMM(n_components=2)
    gmm.means_ = np.array([[mean1], [mean2]])
    gmm.covariances_ = np.array([[[std2**2]], [[std2**2]]])
    gmm.weights_ = np.array([0.5, 0.5])  
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))

    # center: x0
    # x0 = minimize(lambda x: (gmm_cdf(x, gmm) - 0.5)**2, x0=0).x[0]
    x0 = (mean1 + mean2) / 2

    return gmm, x0

def weighting_func_gmm_retain(retain_in_score):
    # 2. build the gaussian mixture model
    gmm = GMM(n_components=1)
    gmm.fit(retain_in_score)

    return gmm

# cumulative prob func
def gmm_cdf(x, gmm):
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    cdf_vals = [w * norm.cdf(x, mean, std) for w, mean, std in zip(weights, means, stds)]
    return np.sum(cdf_vals)

# the cumulative prob for a point
def cumulative_probability(x, gmm):
    return gmm_cdf(x, gmm)

# 6. the cumulative prob for the symmtric point
def symmetric_cumulative_probability(x, x0, gmm):
    symmetric_x = 2 * x0 - x
    return gmm_cdf(symmetric_x, gmm)

def obtain_weights(input_x, gmm, gmm_retain, x0):
    cp_x = cumulative_probability(input_x, gmm)
    cp_symmetric_x = symmetric_cumulative_probability(input_x, x0, gmm)
    cp_sum = 1 - max(cp_x, cp_symmetric_x) + min(cp_x, cp_symmetric_x)
    w_res = cp_sum

    scaling_factor = 10
    cp_sum_scale = cp_sum * scaling_factor
    range_th = 2

    w_res = math.exp(cp_sum_scale - range_th) / (1 + math.exp(cp_sum_scale - range_th))
    
    if w_res >= 0.32:
        w_res = 1
    elif w_res < 0.32 and w_res > 0.2:
        w_res = 10 * (w_res ** 2)
    else:
        w_res = 0

    return w_res




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_train_epochs", default=1.0, type=float)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--loss", type=str, default="margin")
    parser.add_argument("--ood", type=str, default="ocsvm")

    parser.add_argument("--unlearn_dataset", default="clinc_id", type=str)
    parser.add_argument("--ood_dataset", type=str, default="clinc_ood")
    args = parser.parse_args()

    # wandb.init(project=args.project_name, name=args.task_name + '-' + str(args.alpha) + "_" + args.loss)
    # wandb.init(mode="disabled")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    model = RobertaForSelector(args.model_name_or_path, projection_dim=100)
    model.to(args.device)

    # datasets = ['rte', 'sst2', 'mnli', '20ng', 'trec', 'imdb', 'wmt16', 'multi30k']
    datasets = [args.unlearn_dataset, args.ood_dataset]
    benchmarks = ()

    for dataset in datasets:
        if dataset == args.unlearn_dataset:
            train_dataset, test_dataset = load(dataset, tokenizer, dataset_seed=1000, is_id=True)  # biology
        else:
            _, original_val_dataset = load(dataset, tokenizer, dataset_seed=1000, is_id=False)  # non biology
            # print(len(original_val_dataset))
            benchmarks = ((dataset, original_val_dataset),) + benchmarks
    train(args, model, train_dataset, test_dataset, benchmarks)


if __name__ == "__main__":
    main()