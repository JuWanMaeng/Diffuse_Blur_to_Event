# Author: Bingxin Ke
# Last modified: 2024-05-17

from .marigold_trainer import MarigoldTrainer
from .b2f_trainer import B2FTrainer
from .b2f_trainer_C import B2FTrainer_C


trainer_cls_name_dict = {
    "MarigoldTrainer": MarigoldTrainer,
    "B2FTrainer": B2FTrainer,
    "B2FTrainer_C": B2FTrainer_C
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]
