from .CL import CL, CL_FT, CL_KL, CL_FT_epoch
from .FT import FT, FT_l1
from .GA import GA, GA_FT, GA_KL, GA_FT_epoch, GA_KL_epoch
from .KL import KL, KL_FT
from .RL import RL
from .DPO import DPO

# GradAsc --> GA: negative accurate label loss
# GradDif --> GA_FT: negative accuracte label loss + retained data task loss
# PO --> CL_FT: random label loss + retained data task loss
# NPO --> DPO: a different version of random label loss + KL loss on retained data (ensure the logits similarity of models before and after training)
# SOGD --> GA_FT: the same loss as GradDif but with sophia optimizer 
# SOPO --> CL_FT: the same loss as PO but with sophia optimizer
# EUL --> KL_FT: kl loss on forget and retained set + retained data task loss


def get_unlearn_method(name, *args, **kwargs):
    if name == "FT":
        unlearner = FT(*args, **kwargs)
    elif name == "l1sparse":
        unlearner = FT_l1(*args, **kwargs)
    elif name == "GA":
        unlearner = GA(*args, **kwargs)
    elif name == "GA+FT":
        unlearner = GA_FT(*args, **kwargs)
    elif name == "GA+KL":
        unlearner = GA_KL(*args, **kwargs)
    elif name == "GA_FT_epoch":
        unlearner = GA_FT_epoch(*args, **kwargs)
    elif name == "GA_KL_epoch":
        unlearner = GA_KL_epoch(if_kl=True, *args, **kwargs)
    elif name == "RL":
        unlearner = RL(*args, **kwargs)
    elif name == "KL":
        unlearner = KL(if_kl=True, *args, **kwargs)
    elif name == "CL":
        unlearner = CL(*args, **kwargs)
    elif name == "CL+FT":
        unlearner = CL_FT(if_kl=True, *args, **kwargs)
    elif name == "CL+KL":
        unlearner = CL_KL(if_kl=True, *args, **kwargs)
    elif name == "DPO":
        unlearner = DPO(if_kl=True, *args, **kwargs)
    elif name == "CL_FT_epoch":
        unlearner = CL_FT_epoch(*args, **kwargs)
    elif name == "KL_FT":
        unlearner = KL_FT(if_kl=True, *args, **kwargs)
    else:
        raise ValueError("No unlearning method")

    return unlearner
