from .b2e_trainer import B2ETrainer



trainer_cls_name_dict = {
    "B2ETrainer": B2ETrainer,
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]
