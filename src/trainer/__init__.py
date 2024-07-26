# Author: Bingxin Ke
# Last modified: 2024-05-17

from .marigold_trainer import MarigoldTrainer
from .b2f_trainer import B2FTrainer


trainer_cls_name_dict = {
    "MarigoldTrainer": MarigoldTrainer,
    "B2FTrainer": B2FTrainer,
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]
