import torch
from transformers import Trainer
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTrainer
from .KL import kl_loss

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss


class DPO(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_data = inputs["forget"]
        retain_data = inputs["retain"]
        input_ids = forget_data[0].clone()
        labels = forget_data[3]
        postions = forget_data[4]
        pad_id = input_ids[0][-1].item()
        for idx, position in enumerate(postions):
            input_ids[idx, position:] = labels[idx][position:].clone()
            mask = input_ids[idx] == -100
            input_ids[idx, mask] = pad_id
        forget_inputs = {
            "input_ids": input_ids,
            "attention_mask": forget_data[1],
            "labels": labels,
        }

        outputs = model(**forget_inputs)

        # forget_loss = outputs.loss
        forget_loss_current = get_batch_loss(outputs.logits, labels) 

        retain_inputs = {
            "input_ids": retain_data[0],
            "attention_mask": retain_data[1],
            "labels": retain_data[2],
        }

        retain_outputs = model(**retain_inputs)

        with torch.no_grad():
            infer_retain_outputs = self.infer_model(**retain_inputs)
            infer_forget_outputs = self.infer_model(**forget_inputs)
        prob_retain_p = torch.softmax(retain_outputs.logits, dim=-1)
        prob_retain_q = torch.softmax(infer_retain_outputs.logits, dim=-1)

        forget_loss_oracle = get_batch_loss(infer_forget_outputs.logits, labels)
        neg_log_ratios = forget_loss_current - forget_loss_oracle
        forget_loss = -F.logsigmoid(0.1 * neg_log_ratios).mean() * 2 / 0.1 

        loss =  forget_loss + self.gamma * kl_loss(
            prob_retain_p, prob_retain_q
        )

        return (loss, outputs) if return_outputs else loss