import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import Module
from transformers import RobertaModel, RobertaForMaskedLM
from sklearn.covariance import EmpiricalCovariance
from torch.nn.parameter import Parameter
from collections import deque
import sklearn.covariance
from tqdm import tqdm
from sklearn import svm
# from CodeBed.faster_mix_k_means_pytorch import K_Means
from peft import PeftModel
from peft import LoraConfig, get_peft_model


def entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # Normalize the embeddings to simplify cosine similarity calculation
        z1 = nn.functional.normalize(z1, p=2, dim=1)
        z2 = nn.functional.normalize(z2, p=2, dim=1)

        # Concatenate all positive pairs
        n_samples = z1.shape[0]  # Batch size
        z = torch.cat([z1, z2], dim=0)

        # Cosine similarity between all pairs
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Extract logits: positive and negatives
        sim_pos = torch.exp(sim_matrix[:n_samples, n_samples:])
        sim_negs = torch.exp(
            torch.cat([sim_matrix[:n_samples, :n_samples], sim_matrix[:n_samples, n_samples + 1:]], dim=1))

        # Calculate InfoNCE loss
        sum_negs = sim_negs.sum(dim=1)
        loss = -torch.log(sim_pos.diag() / sum_negs)
        return loss.mean()


class RobertaForSelector(nn.Module):
    def __init__(self, model_name, projection_dim):
        super().__init__()
        self.m = 0.999
        self.K = 128
        self.clu_k = 2
        self.layer_num = 13
        self.layer_num_moco = 1
        self.step_interval = 12800
        peft_config = LoraConfig(task_type="SEQ_CLS",
                                 r=8,  # Rank Number
                                 lora_alpha=32,  # Alpha (Scaling Factor)
                                 lora_dropout=0.1,  # Dropout Prob for Lora
                                 target_modules=["query", "key", "value"],
                                 # Which layer to apply LoRA, usually only apply on MultiHead Attention Layer
                                 bias='none', )

        roberta = RobertaForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
        peft_model = get_peft_model(roberta, peft_config)
        print('PEFT Model')
        peft_model.print_trainable_parameters()
        self.roberta = peft_model

        self.roberta_k = RobertaForMaskedLM.from_pretrained(model_name, output_hidden_states=True)

        # for param_q, param_k in zip(self.roberta.parameters(), self.roberta_k.parameters()):
        #     param_k.data.copy_(param_q.data)  # initialize
        #     param_k.requires_grad = False  # not update by gradient
        for param_k in self.roberta_k.parameters():
            # param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.roberta.parameters(), self.roberta_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        for queue, key in zip(self.queues, keys):
            queue.append(key)
            if len(queue) > self.K:
                queue.popleft()

    def forward(self, batch_mlm=None, batch=None, steps=0, dataloader=None):
        outputs = self.roberta(
            input_ids=batch_mlm["input_ids"],
            attention_mask=batch_mlm["attention_mask"],
            labels=batch_mlm["labels"],
        )

        # compute MLM loss
        mlm_loss = outputs.loss
        info_loss = 0

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # self._momentum_update_key_encoder()  # update the key encoder

            outputs_k = self.roberta_k(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        for i in range(13):
            z_1 = torch.mean(outputs.hidden_states[i], dim=1, keepdim=False)
            z_2 = torch.mean(outputs_k.hidden_states[i], dim=1, keepdim=False)
            # z_1_n, z_2_n = nn.functional.normalize(z_1, dim=0), nn.functional.normalize(z_2, dim=0)
            sim_mat = torch.einsum('nc,ck->nk', [z_1, z_2.T.detach()])
            s_dist = F.softmax(sim_mat, dim=1)
            info_loss += torch.mean(entropy(s_dist))

        return mlm_loss, info_loss

    def sample_X_estimator(self, dataloader):
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

        all_layer_features = []
        num_layers = 13
        for i in range(num_layers):
            all_layer_features.append([])

        # for batch in dataloader:
        for step, batch in enumerate(tqdm(dataloader)):
            self.eval()
            batch = {key: value.cuda() for key, value in batch.items()}
            outputs = self.roberta(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )

            # all_hidden_feats = outputs[1]
            all_hidden_feats = outputs.hidden_states

            for i in range(num_layers):
                layer_mean_fea = torch.mean(all_hidden_feats[i], dim=1, keepdim=False).detach()
                all_layer_features[i].append(layer_mean_fea.data.cpu())

        mean_list = []
        precision_list = []
        fea_list = []
        for i in range(num_layers):
            all_layer_features[i] = torch.cat(all_layer_features[i], axis=0)
            fea_list.append(F.normalize(all_layer_features[i], dim=-1))
            sample_mean = torch.mean(all_layer_features[i], axis=0)
            X = all_layer_features[i] - sample_mean
            group_lasso.fit(X.numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float()
            mean_list.append(sample_mean.cuda())
            precision_list.append(temp_precision.cuda())

        return mean_list, precision_list, fea_list

    def get_unsup_Mah_score(self, dataloader, sample_mean, precision, fea_list):
        total_mah_scores, total_cs_scores = [], []
        num_layers = 13
        for i in range(num_layers):
            total_mah_scores.append([])
            total_cs_scores.append([])

        # for batch in dataloader:
        for step, batch in enumerate(tqdm(dataloader)):
            batch_all_features = []
            self.eval()
            batch = {key: value.cuda() for key, value in batch.items()}
            outputs = self.roberta(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            # batch_all_features = outputs.hidden_states
            # all_hidden_feats = outputs[1]
            all_hidden_feats = outputs.hidden_states

            for i in range(num_layers):
                layer_mean_fea = torch.mean(all_hidden_feats[i], dim=1, keepdim=False).detach()
                batch_all_features.append(layer_mean_fea.data)

            for i in range(len(batch_all_features)):
                batch_sample_mean = sample_mean[i]
                out_features = batch_all_features[i]
                zero_f = out_features - batch_sample_mean
                gaussian_score = -0.5 * ((zero_f @ precision[i]) @ zero_f.t()).diag()
                out_feas = F.normalize(out_features, dim=-1)
                cs_score = out_feas @ fea_list[i].t().cuda()
                cs_score = torch.max(cs_score, dim=1)[0]
                all_score = -cs_score * 1000. + gaussian_score
                total_mah_scores[i].extend(all_score.cpu().numpy())

        for i in range(len(total_mah_scores)):
            total_mah_scores[i] = np.expand_dims(np.array(total_mah_scores[i]), axis=1)

        return np.concatenate(total_mah_scores, axis=1)


class RobertaForSelector_inference(nn.Module):
    def __init__(self, model_name, lora_path, projection_dim):
        super().__init__()
        self.m = 0.999
        self.K = 128
        self.clu_k = 2
        self.layer_num = 13
        self.layer_num_moco = 1
        self.step_interval = 12800
        peft_config = LoraConfig(task_type="SEQ_CLS",
                                 r=8,  # Rank Number
                                 lora_alpha=32,  # Alpha (Scaling Factor)
                                 lora_dropout=0.1,  # Dropout Prob for Lora
                                 target_modules=["query", "key", "value"],
                                 # Which layer to apply LoRA, usually only apply on MultiHead Attention Layer
                                 bias='none', )

        roberta = RobertaForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
        peft_model = PeftModel.from_pretrained(
            roberta,
            lora_path,
        )
        self.roberta = peft_model


        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        for queue, key in zip(self.queues, keys):
            queue.append(key)
            if len(queue) > self.K:
                queue.popleft()

    def forward(self, batch_mlm=None, batch=None, steps=0, dataloader=None):
        outputs = self.roberta(
            input_ids=batch_mlm["input_ids"],
            attention_mask=batch_mlm["attention_mask"],
            labels=batch_mlm["labels"],
        )

        # compute MLM loss
        mlm_loss = outputs.loss
        info_loss = 0

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # self._momentum_update_key_encoder()  # update the key encoder

            outputs_k = self.roberta_k(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        for i in range(13):
            z_1 = torch.mean(outputs.hidden_states[i], dim=1, keepdim=False)
            z_2 = torch.mean(outputs_k.hidden_states[i], dim=1, keepdim=False)
            # z_1_n, z_2_n = nn.functional.normalize(z_1, dim=0), nn.functional.normalize(z_2, dim=0)
            sim_mat = torch.einsum('nc,ck->nk', [z_1, z_2.T.detach()])
            s_dist = F.softmax(sim_mat, dim=1)
            info_loss += torch.mean(entropy(s_dist))

        return mlm_loss, info_loss

    def sample_X_estimator(self, dataloader):
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

        all_layer_features = []
        num_layers = 13
        for i in range(num_layers):
            all_layer_features.append([])

        # for batch in dataloader:
        for step, batch in enumerate(tqdm(dataloader)):
            self.eval()
            batch = {key: value.cuda() for key, value in batch.items()}
            outputs = self.roberta(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )

            # all_hidden_feats = outputs[1]
            all_hidden_feats = outputs.hidden_states

            for i in range(num_layers):
                layer_mean_fea = torch.mean(all_hidden_feats[i], dim=1, keepdim=False).detach()
                all_layer_features[i].append(layer_mean_fea.data.cpu())

        mean_list = []
        precision_list = []
        fea_list = []
        for i in range(num_layers):
            all_layer_features[i] = torch.cat(all_layer_features[i], axis=0)
            fea_list.append(F.normalize(all_layer_features[i], dim=-1))
            sample_mean = torch.mean(all_layer_features[i], axis=0)
            X = all_layer_features[i] - sample_mean
            group_lasso.fit(X.numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float()
            mean_list.append(sample_mean.cuda())
            precision_list.append(temp_precision.cuda())

        return mean_list, precision_list, fea_list

    def get_unsup_Mah_score_s(self, ood_input, sample_mean, precision, fea_list):
        total_mah_scores, total_cs_scores = [], []
        num_layers = 13
        for i in range(num_layers):
            total_mah_scores.append([])
            total_cs_scores.append([])


        batch_all_features = []
        self.eval()
        outputs = self.roberta(
            input_ids=ood_input['input_ids'].cuda(),
            attention_mask=ood_input['attention_mask'].cuda(),
        )
        # batch_all_features = outputs.hidden_states
        # all_hidden_feats = outputs[1]
        all_hidden_feats = outputs.hidden_states

        for i in range(num_layers):
            layer_mean_fea = torch.mean(all_hidden_feats[i], dim=1, keepdim=False).detach()
            batch_all_features.append(layer_mean_fea.data)

        for i in range(len(batch_all_features)):
            batch_sample_mean = sample_mean[i]
            out_features = batch_all_features[i]
            zero_f = out_features - batch_sample_mean
            gaussian_score = -0.5 * ((zero_f @ precision[i]) @ zero_f.t()).diag()
            out_feas = F.normalize(out_features, dim=-1)
            cs_score = out_feas @ fea_list[i].t().cuda()
            cs_score = torch.max(cs_score, dim=1)[0]
            all_score = -cs_score * 1000. + gaussian_score
            total_mah_scores[i].extend(all_score.cpu().numpy())

        for i in range(len(total_mah_scores)):
            total_mah_scores[i] = np.expand_dims(np.array(total_mah_scores[i]), axis=1)

        return np.concatenate(total_mah_scores, axis=1)
