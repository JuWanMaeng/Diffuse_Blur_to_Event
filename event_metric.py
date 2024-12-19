import numpy as np
import torch


def metric_and_output(preds, gt):
    """
    Calculate the metrics for the predicted depth map.
    """
    num_of_pred = len(preds)
    metrics = []

    # [-1,1] -> [0,1]
    gt = gt / 2 + 0.5         # [6,H,W]
    gt = gt.transpose(1,2,0)  # [H,W,6]

    # record the best pred by min rmse
    min_rmse = 10000    


    for pred in preds:
        # [-1,1] -> [0,1]
        pred = pred / 2 + 0.5
        
        # RMSE between pred and gt
        diff = pred - gt
        diff_power = diff ** 2
        navie_rmse = np.sqrt(np.mean(diff_power))

        # reverse pred order  # pred.shape = [6, H, W]
        reversed_pred = pred[::-1,:,:]
        reversed_diff = reversed_pred - gt
        reversed_diff = reversed_diff ** 2
        reversed_rmse = np.sqrt(np.mean(reversed_diff))

        rmse = min(navie_rmse, reversed_rmse)
        metrics.append(rmse)

        if rmse < min_rmse:
            min_rmse = rmse
            best_pred = pred

    return metrics, best_pred



    