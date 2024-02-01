"""
Script that computes some ML metrics based on a dictionary with experiments
results
"""

import json 

import sklearn.metrics as skm 
import numpy as np 


def compute_metrics(
    results_experiment : dict,
    **kwargs
) -> dict:
    """
    Function that computes some ML metrics based on a dictionary with results

    Parameters
    ----------
    results_experiment : dict
        Dictionary with results of experiments
    kwargs : dict
        Keyboard arguments to the sklearn functions 
        for computing the metrics. The 'average' value
        needs to be specified for multiclass 
        classification

    Returns
    -------
    metric_outputs : dict
        Dictionary with some ML metrics computed based on results_experiment
    """

    metric_outputs = {}
    
    if results_experiment['model_name'] != 'beta_1':
        iterations = results_experiment['input_args']['iterations']
        runs = results_experiment['input_args']['runs']

    """Check if results are binary of multiclass"""

    labels_keys = ['train_labels', 'val_labels', 'test_labels']
    all_labels = [
        label for key,values in results_experiment.items(
        ) if key in labels_keys for label in values
    ]
    n_labels = len(set(all_labels))

    
    def compute_roc_auc_score(
            labels : list[int], probs : list[list[float]], n_labels : int,
            **kwargs : dict
    ) -> float:
        """
        Function for computing the roc_auc_score. 
        Its behaviour in sklearn is slightly different when
        compared to other metrics.

        Parameters
        ----------
        labels : list
            List of labels as integers
        probs: list[list[float]]
            List with probabilities associated for each label
        n_labels : 
            Total number of labels
        kwargs : dict
        Keyboard arguments to the sklearn functions 
        for computing the metrics. Specify the 'average' value
        is needed for multiclass classification
        Returns
        -------
        roc_auc_score : float
            Value of the roc_auc_score
        """
        if n_labels == 2:
            try:
                roc_auc_score = skm.roc_auc_score(
                    labels, np.array(probs)[:,1], **kwargs
                )
            # ROC AUC score is not defined...
            # ...when the predictions contain only one class
                
            except ValueError:
                roc_auc_score = np.nan
        else:
            try:
                roc_auc_score = skm.roc_auc_score(
                    labels, np.array(probs), **kwargs
                )
            except ValueError:
                roc_auc_score = np.nan
        return roc_auc_score
    
    """Define a metric's functions dictionary"""

    metric_funcs_dict = {
        'precision': skm.precision_score,
        'accuracy' : skm.accuracy_score,
        'recall' : skm.recall_score,
        'roc_auc': compute_roc_auc_score,
        'confusion_matrix': skm.confusion_matrix,
        'f1': skm.f1_score
    }

    def compute_single_metric(
        metric : str,  n_labels : int,
        dataset  :str, run : int,
        iteration : int = None, **kwargs : dict
    ):
        """
        Function for computing one metric based on the results dictionary.
        Depending on the metric, probabilities or predictions will need to 
        be introduced. Works for pre-alpha-2 and alpha-3.

        Parameters
        ----------
        metric : str
            Name of the metric to compute
        n_labels : int
            Total number of labels in the experiment
        dataset : str
            Partition of the dataset for which the 
            metric is being computed. Must be in
            ('train', 'val', 'test')
        run : int
            Number of the run for which we want to 
            compute the metric
        iteration : str 
            Number of the iteration for which we want to
            compute the metric
        kwargs : dict
            Keyboard arguments to the sklearn functions 
            for computing the metrics. Specify the 'average' value
            is needed for multiclass classification
        Returns
        -------
        float
            Value of the computed metric
        """
        if metric != 'roc_auc' and iteration != None:
            try:
                return metric_funcs_dict[metric](
                    results_experiment[f'{dataset}_labels'],
                    results_experiment[
                        f'{dataset}_predictions'][run][iteration],
                    **kwargs
                )
            except TypeError:
                return metric_funcs_dict[metric](
                    results_experiment[f'{dataset}_labels'],
                    results_experiment[
                        f'{dataset}_predictions'][run][iteration]
                )
        elif metric != 'roc_auc' and iteration == None:
            try:
                return metric_funcs_dict[metric](
                    results_experiment[f'{dataset}_labels'],
                    results_experiment[
                        f'{dataset}_predictions'][run],
                    **kwargs
                )
            except TypeError:
                return metric_funcs_dict[metric](
                    results_experiment[f'{dataset}_labels'],
                    results_experiment[
                        f'{dataset}_predictions'][run]
                )
        elif metric == 'roc_auc' and iteration != None:
            return metric_funcs_dict[metric](
                results_experiment[f'{dataset}_labels'],
                results_experiment[
                    f'{dataset}_probabilities'][run][iteration],
                n_labels,
                **kwargs
            )
        elif metric == 'roc_auc' and iteration == None:
            return metric_funcs_dict[metric](
                results_experiment[f'{dataset}_labels'],
                results_experiment[
                    f'{dataset}_probabilities'][run],
                n_labels,
                **kwargs
            )
        
    def compute_single_metric_beta_1(
        metric : str,  **kwargs : dict
    ):
        """
        Function for computing one metric based on the results dictionary.
        Works for beta_1, as in this model iterations, runs and probabilities
        don't apply. This means that ROC auc score can't be computed, and that
        measures only apply to the test dataset. 

        Parameters
        ----------
        metric : str
            Name of the metric to compute
        kwargs : dict
            Keyboard arguments to the sklearn functions 
            for computing the metrics. Specify the 'average' value
            is needed for multiclass classification
        Returns
        -------
        float
            Value of the computed metric
        """
        try:
            return metric_funcs_dict[metric](
                results_experiment['test_labels'],
                results_experiment[
                    'test_predictions'],
                **kwargs
            )
        except TypeError:
            return metric_funcs_dict[metric](
                results_experiment['test_labels'],
                results_experiment[
                    'test_predictions']
            )

    """ Loop over the results to get all the metrics"""

    if results_experiment['model_name'] == 'beta_1':
        ## ROC AUC doesn't apply in this case
        del metric_funcs_dict['roc_auc']
        for m in metric_funcs_dict.keys():
            metric_outputs[f'{m}_test'] = compute_single_metric_beta_1(
                m, **kwargs
            )

    if (results_experiment['model_name'] == 'pre_alpha_2'
        or results_experiment['model_name'] == 'alpha_3'
    ):
        for m in metric_funcs_dict.keys():
            metric_outputs[f'{m}_train'] = [[] for _ in range(runs)]
            metric_outputs[f'{m}_val'] = [[] for _ in range(runs)]
            metric_outputs[f'{m}_test'] = []
            for i in range(runs):
                metric_outputs[f'{m}_test'].append(
                    compute_single_metric(
                        m, n_labels, 'test', i, **kwargs
                    )
                )
                for j in range(iterations):
                    for dataset in ('train', 'val'):
                        metric_outputs[f'{m}_{dataset}'][i].append(
                            compute_single_metric(
                                m, n_labels, dataset, i, j, **kwargs
                            )
                        )


    return metric_outputs