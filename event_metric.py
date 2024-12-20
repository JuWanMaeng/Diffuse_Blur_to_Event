import numpy as np
import torch


def metric_and_output(preds, gt):
    """
    Calculate the metrics for the predicted depth map.
    """
    num_of_pred = len(preds)
    # metrics = []

    # [-1,1] -> [0,1]
    gt = gt / 2 + 0.5         # [6,H,W]
    gt = gt.transpose(1,2,0)  # [H,W,6]

    # record the best pred by min rmse
    min_rmse = 10000    

    # record the best navie's rmse and reversed's rmse
    min_navie_rmse = 10000
    min_reversed_rmse = 10000

    avg_rmse = 0 # 여러개의 샘플들의 평균 rmse

    for pred in preds:
        # [-1,1] -> [0,1]
        pred = pred / 2 + 0.5
        
        # RMSE between pred and gt
        diff = pred - gt
        diff_power = diff ** 2
        navie_rmse = np.sqrt(np.mean(diff_power))
        once_rmse = navie_rmse  # one shot
        avg_rmse += navie_rmse

        # reverse pred order  # pred.shape = [6, H, W]
        reversed_pred = pred[::-1,:,:]
        reversed_diff = reversed_pred - gt
        reversed_diff = reversed_diff ** 2
        reversed_rmse = np.sqrt(np.mean(reversed_diff))

        rmse = min(navie_rmse, reversed_rmse)
        
        # metrics.append(rmse)

        if rmse < min_rmse:
            min_rmse = rmse
            best_pred = pred
            min_navie_rmse = navie_rmse
            min_reversed_rmse = reversed_rmse

    avg_rmse /= num_of_pred

    # return pred [0,1] to [-1,1]
    best_pred = best_pred * 2 - 1

    return min_navie_rmse, min_reversed_rmse, once_rmse, avg_rmse ,best_pred



    