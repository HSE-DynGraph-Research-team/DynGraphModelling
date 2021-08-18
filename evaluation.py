import math
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

# specify everywhere!
# batch_size=batch_size, data_mode=model_params['data_mode'], divide_by=model_params['divide_by']

def eval_node_bin_classification(
    model, 
    data,
    data_setting_mode, 
    batch_params, 
    eval_mode,
    **model_params):
    
    targets = []
    predicts = []

    for i, batch in enumerate(data(**batch_params)):
        size = batch[0].shape[0] if batch_params['data_mode']=='transaction' else batch.x.shape[0]
        if not size:
            continue
        pred_prob_batch = model.compute_clf_probability(
            batch,
            eval_mode=eval_mode,
            data_setting_mode=data_setting_mode,
            **model_params,
        )
        
        predicts.append(pred_prob_batch.cpu().detach().numpy().reshape(-1) if isinstance(pred_prob_batch, torch.Tensor) else pred_prob_batch.reshape(-1))
        if batch_params['data_mode']=='transaction':
            targets.append(batch[-1].reshape(-1))
        elif batch_params['data_mode']=='snapshot':
            targets.append(batch.y.reshape(-1))
        

    pred_prob = np.concatenate(predicts, axis=0)
    true_labels = np.concatenate(targets, axis=0)
    auc_roc = roc_auc_score(true_labels, pred_prob)
    return {'AUC ROC': auc_roc}


def eval_edge_prediction(
    model, 
    data,
    negative_edge_sampler, #new nodes sampler for inductive, known nodes for transductive 
    data_setting_mode,
    batch_params,
    eval_mode,
    **model_params,
    ):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []
    with torch.no_grad():
        for i, batch in enumerate(data(**batch_params)):
            size = batch[0].shape[0] if batch_params['data_mode']=='transaction' else batch.x.shape[0]
            if not size:
                continue
            _, negative_samples = negative_edge_sampler.sample(size)


            pos_prob, neg_prob = model.compute_edge_probabilities(
                batch,
                negative_samples,
                eval_mode=eval_mode,
                data_setting_mode=data_setting_mode,
                **model_params)
            if isinstance(pos_prob, torch.Tensor):
                pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()]) if model.device!='cpu' else np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            elif isinstance(pos_prob, np.array):
                pred_score = np.concatenate([pos_prob, neg_prob],axis=0)
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

    return {'AP': np.mean(val_ap), 'AUC':np.mean(val_auc)}
