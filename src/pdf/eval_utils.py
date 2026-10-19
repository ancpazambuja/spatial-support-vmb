

"""@package eval_utisl.py
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple


############################
# basic stuff
from datetime import datetime
import time
import random
import sys
import os
import re
import pandas as pd

############################
import scipy.stats as st

import numpy as np
import ml_collections

############################
# experiment tracking
import mlflow
from mlflow import MlflowClient


############################
# internal modules
from src.experiment_tracking.mlflow import MLFlow
from src.experiment_tracking.mlflow_utils import MLFlowUtils

from src.utl.logging import Log as log

from src.utl.classes import Utils as utl
from src.utl.classes import Graph as graph

MetricVectorFetcher = Callable[[Any, str, str], Sequence[float]]
# signature example: fetch_metric_vector(mlflow_client, run_id, metric_name) -> list/np.ndarray

IndexParser = Callable[[Any], Sequence[int]]

class EvalUtils:
    def print_mccv_summary(summary=None, verbose=True):
        for k,v in summary.items():
            print(f"\n[ {k} ]")
            if verbose:
                for kk,vv in v.items():
                    print(f"<< {kk} >> {vv}")

    def get_model_name_paper(use_transformer=None, conv_kernel_size=None, conv_dilation_size=None):
        sep="."
        if use_transformer == "True":
            a="TransUNet"
        elif use_transformer == "False":
            a="U-Net"
        else:
            raise ValueError(f"Unexpected value from use_transformer = {use_transformer}")
        ks=" k="+str(conv_kernel_size)
        ds=" r="+str(conv_dilation_size)
        return  a+ks+ds

    def get_model_name(use_transformer=None, conv_kernel_size=None, conv_dilation_size=None, transposed_conv_kernel_size=None, optmizer=None):
        sep="."
        if use_transformer == "True":
            a="transunet"
        elif use_transformer == "False":
            a="unet"
        else:
            raise ValueError(f"Unexpected value from use_transformer = {use_transformer}")
        ks="cks_"+str(conv_kernel_size)+"x"+str(conv_kernel_size)
        ds="cds_"+str(conv_dilation_size)+"x"+str(conv_dilation_size)
        ts="tcks_"+str(transposed_conv_kernel_size)+"x"+str(transposed_conv_kernel_size)
        opt="opt_"+optmizer
        return  a+sep+ks+sep+ds+sep+ts+sep+opt


    def get_model_name_paper_from_df_line(df_line=None):
        arch = df_line["use_transformers"]
        ks = df_line["conv_kernel_size"]
        ds = df_line["conv_dilation_size"]
        return EvalUtils.get_model_name_paper(use_transformer=arch, conv_kernel_size=ks, conv_dilation_size=ds)
    

    def get_model_name_from_df_line(df_line=None):
        arch = df_line["use_transformers"]
        ks = df_line["conv_kernel_size"]
        ds = df_line["conv_dilation_size"]
        ts = df_line["transposed_conv_kernel_size"]
        opt = df_line["optimizer"]
        regex_pattern = r"\.Adam'"
        if re.search(regex_pattern, opt):
            opt = "Adam"
        else:
            opt = "AdamW"
        return EvalUtils.get_model_name(use_transformer=arch, conv_kernel_size=ks, conv_dilation_size=ds, transposed_conv_kernel_size=ts, optmizer=opt)


    def arch_name2arch_name_paper(arch_name=None):
        sep = "."
        parts = arch_name.split(sep)
        
        # Parse arch token -> use_transformer flag expected by get_model_name_paper
        arch_token = parts[0]
        if arch_token == "transunet":
            use_transformer = "True"
        elif arch_token == "unet":
            use_transformer = "False"
        else:
            raise ValueError(f"Unexpected arch token in arch_name = {arch_name}")
        
        # Parse conv_kernel_size from "cks_KxK"
        cks_match = re.match(r"cks_(\d+)x(\d+)", parts[1])
        if not cks_match:
            raise ValueError(f"Unexpected cks token in arch_name = {arch_name}")
        conv_kernel_size = cks_match.group(1)
        
        # Parse conv_dilation_size from "cds_DxD"
        cds_match = re.match(r"cds_(\d+)x(\d+)", parts[2])
        if not cds_match:
            raise ValueError(f"Unexpected cds token in arch_name = {arch_name}")
        conv_dilation_size = cds_match.group(1)

        ret = EvalUtils.get_model_name_paper(
            use_transformer=use_transformer,
            conv_kernel_size=conv_kernel_size,
            conv_dilation_size=conv_dilation_size,
        )

        print(f"old new = {arch_name}, {ret}")

        return ret

    # signature example: parse_indexes(raw_test_indexes) -> list[int]
    def get_metric_values_from_run(mlflow_=None, run_id=None, metric=None):
        hist = mlflow_.get_metric_history_by_run(run_id=run_id, metric=metric)
        values = [None]*len(hist)
        for m in hist:
            values[m.step] = m.value
        return values

    #def get_metric_values_from_run(mlflow_=None, run_id=None, metric=None):
    #    hist = mlflow_.get_metric_history_by_run(run_id=run_id, metric=metric)
    #    if not hist:
    #        return []
    #
    #    max_step = max(m.step for m in hist)
    #    values = np.full(max_step + 1, np.nan, dtype=np.float32)
    #
    #    for m in hist:
    #        values[m.step] = m.value
    #
    #    return values.tolist()
    
    def get_metrics(metrics=None):
        return { 
            'mse': metrics['mse_mean_test_set'],
            'mse_std': metrics['mse_std_test_set'],
            'mse_min': metrics['mse_min_test_set'],
            'mse_max': metrics['mse_max_test_set'],
            'mae': metrics['mae_mean_test_set'],
            'mae_std': metrics['mae_std_test_set'],
            'mae_min': metrics['mae_min_test_set'],
            'mae_max': metrics['mae_max_test_set'],
            'ssim': metrics['ssim_mean_test_set'],
            'ssim_std': metrics['ssim_std_test_set'],
            'ssim_min': metrics['ssim_min_test_set'],
            'ssim_max': metrics['ssim_max_test_set'],
        }
    
    def get_params(params=None):
        # treating exceptions
        attention_config = EvalUtils.get_attention_config_values(params)
    
        conv_kernel_size = eval(params['module__conv_kernel_size'])[0]
    
        transposed_conv_kernel_size = 2
        if 'module__transposed_conv_kernel_size' in params:
            transposed_conv_kernel_size = eval(params['module__transposed_conv_kernel_size'])[0]
    
        conv_dilation_size = 1
        if 'module__conv_dilation_size' in params:
            conv_dilation_size = eval(params['module__conv_dilation_size'])[0]
    
    
        
        return { 
            'seed': params['_x_seed'],
            'minibatch_size': params['batch_size'],
            'max_epochs': params['max_epochs'],
            'lr': params['lr'],
            'weight_decay': params['optimizer__weight_decay'],
            'optimizer': params['optimizer'],
            'n_filters_init': params['module__n_filters_init'],
            'conv_kernel_size': conv_kernel_size,
            'transposed_conv_kernel_size': transposed_conv_kernel_size,
            'conv_dilation_size': conv_dilation_size,
            'att_in_channels': params['module__attention_in_channels'],
            'use_transformers': params['module__use_transformers'],
            'att_hidden_size': attention_config["_x_att_hidden_size"],
            'att_num_heads': attention_config['_x_att_num_heads'],
            'att_num_layers': attention_config['_x_att_num_layers'],
            'att_att_dropout_rate': attention_config['_x_att_att_dropout_rate'],
            'att_dropout_rate': attention_config['_x_att_dropout_rate'],
            'att_mlp_dim': attention_config['_x_att_mlp_dim'],
            'test_indexes' : EvalUtils.parse_string(params['_x_test_indexes'],dtype=int),
            'train_indexes' : EvalUtils.parse_string(params['_x_train_indexes'],dtype=float),
        }
    
    def get_attention_config_values(params=None):
        att_config = ml_collections.ConfigDict(MLFlowUtils.string_to_dict(params['module__attention_config']))
        ret = { '_x_att_hidden_size': att_config["hidden_size"],
                '_x_att_num_heads' : att_config["transformer"]["num_heads"],
                '_x_att_num_layers': att_config["transformer"]["num_layers"],
                '_x_att_att_dropout_rate' : att_config["transformer"]["attention_dropout_rate"],
                '_x_att_dropout_rate' :  att_config["transformer"]["dropout_rate"],
                '_x_att_mlp_dim' : att_config["transformer"]["mlp_dim"]
        }
        return ret 
    
    
    
    def build_info(run_id=None, run_name=None, start_time=None, end_time=None):
        data_type = utl.get_dataset_type_from_name(run_name)
        assert data_type is not None and (data_type == 'deep' or data_type == 'shallow'), f"[ ERROR ] Unknown dataset type!"
        return {
            'run_id' : run_id,
            'run_name' : run_name,
            'data_type' : data_type,
            'start_time' : start_time,
            'end_time' : end_time,
        }

    def get_metric(run=None, prefix='mse'):
       ret_str = str(run.data.metrics[prefix])
       ret_str = ret_str +", " + str(run.data.metrics[prefix+'_std'])
       ret_str = ret_str +", " + str(run.data.metrics[prefix+'_min'])
       ret_str = ret_str +", " + str(run.data.metrics[prefix+'_max'])
       return ret_str

    def check_resolution(runs=None, res=None, verbose=True):
        check = False if ((runs is None) or (len(runs) == 0)) else True
        error_list = list()
        for i, (k,v) in enumerate(runs.items()):
            if k.find(res) < 0:
                error_list.append(k)
        if len(error_list) > 0:
            check = False
            if verbose > 0:
                log.warn(f"List of runs violating the target dataset resolution ({res}): {error_list}")
        return check
    
    
    def make_metric_vector_fetcher(get_metric_values_from_run_func):
        @lru_cache(maxsize=2048)
        def _fetch(mlflow_client_id, run_id, metric_name):
            # mlflow_client_id is just to make cache key stable if you use multiple clients
            vals = get_metric_values_from_run_func(mlflow_=mlflow_client_id, run_id=run_id, metric=metric_name)
            return np.asarray(vals, dtype=np.float32)
    
        def fetcher(mlflow_client, run_id, metric_name):
            # Use the object itself as an id (works if you use one client instance)
            return _fetch(mlflow_client, run_id, metric_name)
    
        return fetcher
    
    
    def parse_string(val, dtype=float):
        # If it's already a list or array, just return it
        if isinstance(val, (list, np.ndarray)):
            #print("11111111111111111")
            return val
            
        # If it's a string, clean it and split it into numbers
        if isinstance(val, str):
            #print("2222222222222222")
            # Remove brackets and newlines
            clean_str = val.replace('[', '').replace(']', '').replace('\n', ' ')
            # Split by whitespace and convert to the target data type
            return [dtype(x) for x in clean_str.split()]
        
        #print(f"333333333333333 type(val) = {type(val)}")
            
        return val
    
    
    def compute_corrected_ci(data_array, alpha=0.05, n_train=None, n_test=None):
        """
        Computes a 1-alpha CI corrected for overlapping data in random splits.
        """
        clean_data = np.array(data_array)[~np.isnan(data_array)]
        n = len(clean_data)
       
        # We need at least 2 data points to calculate variance
        if n < 2:
            return (np.nan, np.nan)
           
        mean_val = np.mean(clean_data)
       
        # 1. Calculate the raw standard deviation of the scores
        sd = np.std(clean_data, ddof=1)
       
        if sd == 0:
            return (mean_val, mean_val)
           
        # 2. Apply the Nadeau-Bengio correction to get the true Standard Error
        correction_factor = np.sqrt((1 / n) + (n_test / n_train))
        se_corrected = sd * correction_factor
       
        # 3. Build the interval
        return st.t.interval(1.0-alpha, df=n-1, loc=mean_val, scale=se_corrected)


    def compute_ci_95(series):
        clean_data = np.array(series)[~np.isnan(series)]
        n = len(clean_data)
        if n < 2:
            return (np.nan, np.nan)
        
        mean_val = np.mean(clean_data)
        sem = st.sem(clean_data, ddof=1)
        if sem == 0:
            return (mean_val, mean_val)
        return st.t.interval(0.95, df=n-1, loc=mean_val, scale=sem)


    def sort_summary(summary=None, attr_name=None, reverse=False):
        '''
        Sort matrix summary (dict returned by get_mccv_summary) by attr_name.
        '''
        return sorted(summary.items(), key=lambda item: item[1][attr_name], reverse=reverse)
    
    def fast_fail_mock(df_sorted=None, ci_name=None, reverse=False):
        total_models = len(df_sorted)
        return df_sorted.copy(), total_models
    
    def fast_fail(df_sorted=None, ci_name=None, reverse=False):
        """
        Filters a sorted DataFrame, keeping only models whose Confidence Intervals
        overlap with the best model (the Champion).
        
        Parameters:
        - df_sorted: DataFrame sorted by the main metric (best model at index 0).
        - ci_name: String name of the column containing the CI tuples (e.g., 'mse_95ci').
        - reverse: Boolean. False for "lower-is-better", True for "higher-is-better".
        """
        if df_sorted is None or df_sorted.empty or ci_name is None:
            return df_sorted, 0
            
        total_models = len(df_sorted)
        
        # 1. Extract the Champion's CI bounds (row 0)
        champion_ci = df_sorted.iloc[0][ci_name]
        champ_lower, champ_upper = champion_ci[0], champion_ci[1]
        
        # 2. Extract lower and upper bounds for every model into fast Pandas Series
        all_lowers = df_sorted[ci_name].apply(lambda x: x[0])
        all_uppers = df_sorted[ci_name].apply(lambda x: x[1])
    
        # 3. Apply the overlap logic
        if not reverse:
            # Lower is better (e.g., MSE). 
            # A model survives if its lower bound reaches down to/below the Champion's upper bound.
            survivor_mask = all_lowers <= champ_upper
        else:
            # Higher is better (e.g., SSIM). 
            # A model survives if its upper bound reaches up to/above the Champion's lower bound.
            survivor_mask = all_uppers >= champ_lower
            
        # 4. Filter the dataframe to keep only survivors
        df_filtered = df_sorted[survivor_mask].copy()
        
        return df_filtered, total_models
    
    
    def calculate_nadeau_bengio_ci(champ_scores, surv_scores, n_train, n_test, total_models=20, alpha=0.05):
        d = np.array(champ_scores) - np.array(surv_scores)
        n = len(d)
        mean_diff = np.mean(d)
       
        # 1. Calculate standard deviation (not standard error yet)
        sd_diff = np.std(d, ddof=1)
       
        # 2. Apply the Nadeau-Bengio Correction to the Standard Error
        correction_factor = np.sqrt((1 / n) + (n_test / n_train))
        se_corrected = sd_diff * correction_factor
       
        # 3. Apply Bonferroni
        alpha_adj = alpha / (total_models - 1)
        confidence_level = 1 - alpha_adj
       
        # 4. Calculate final CI
        ci_lower, ci_upper = st.t.interval(confidence_level, df=n-1, loc=mean_diff, scale=se_corrected)
       
        return mean_diff, (ci_lower, ci_upper)
    
    
    def calculate_adjusted_paired_ci(champion_scores, survivor_scores, total_models=20, alpha=0.05):
        # 1. Calculate the pairwise differences (d_i)
        d = np.array(champion_scores) - np.array(survivor_scores)
        n = len(d)
        
        # 2. Calculate mean of differences and standard error
        mean_diff = np.mean(d)
        se_diff = st.sem(d, ddof=1)
        
        # 3. Apply the Bonferroni correction
        num_comparisons = total_models - 1
        alpha_adj = alpha / num_comparisons
        
        # The confidence level for scipy is 1 - alpha_adj
        confidence_level = 1 - alpha_adj
        
        # 4. Calculate the adjusted Confidence Interval
        ci_lower, ci_upper = st.t.interval(confidence_level, df=n-1, loc=mean_diff, scale=se_diff)
        
        return mean_diff, (ci_lower, ci_upper)
    
    # Example usage:
    #mean_d, adjusted_ci = calculate_adjusted_paired_ci(champion_seeds, unet_seeds, total_models=20)
    #
    #print(f"Mean Difference: {mean_d:.5f}")
    #print(f"Adjusted Paired CI: [{adjusted_ci[0]:.5f}, {adjusted_ci[1]:.5f}]")
    #
    #if adjusted_ci[0] < 0 < adjusted_ci[1]:
    #    print("Result: Statistical Tie (CI contains zero)")
    #else:
    #    print("Result: Champion is statistically different!") 
    ##################################
    
    def build_paired_test_table(df_raw, y_order=None, n_train=None, n_test=None, total_models=None, metric_col='mse', alpha=0.05):
        """
        Executes Bonferroni-adjusted paired t-tests between the Champion and all surviving models.
        
        Parameters:
        - df_raw: The unlocked raw dataframe containing all seeds (e.g., df_raw_survivors)
        - y_order: List of surviving architecture names. IMPORTANT: The Champion MUST be index 0!
        - total_models: Integer representing the original number of models before fast-fail (e.g., 20).
        - metric_col: The string name of the column containing the scores.
        """
        
        champion_name = y_order[0]
        #survivors = y_order[1:]
        survivors = y_order[:]
        
        # 1. PIVOT THE DATA: This guarantees perfect seed-to-seed alignment!
        # Rows become seeds, columns become architectures.
        df_pivot = df_raw.pivot(index='seed', columns='arch_name', values=metric_col)
        # This mathematically averages the duplicates into a single, stable value!
        
        results = []
        
        # Extract the Champion's exact seed scores
        champion_scores = df_pivot[champion_name].dropna()
        
        # 2. Loop through the remaining survivors
        for survivor_name in survivors:
            survivor_scores = df_pivot[survivor_name].dropna()
            
            # Ensure we only compare seeds that BOTH models successfully completed
            common_seeds = champion_scores.index.intersection(survivor_scores.index)
            champ_aligned = champion_scores.loc[common_seeds]
            surv_aligned = survivor_scores.loc[common_seeds]
            
            # 3. Call your custom Bonferroni CI function
            mean_diff, (ci_lower, ci_upper) = EvalUtils.calculate_nadeau_bengio_ci(
                champ_aligned, 
                surv_aligned, 
                n_train=n_train, 
                n_test=n_test, 
                total_models=total_models, 
                alpha=alpha
            )
    
            # 4. Determine the statistical verdict
            # If the interval crosses exactly 0, it's a tie. 
            # (This works regardless of whether the metric is "lower is better" or "higher is better")
            if ci_lower < 0 < ci_upper:
                status = "Statistical Tie (Top Tier)"
            else:
                status = "Significantly Worse"
                
            results.append({
                'Architecture': survivor_name,
                'Mean Diff (Champ - Model)': mean_diff,
                f'Adj CI Lower': ci_lower,
                f'Adj CI Upper': ci_upper,
                'Verdict': status
            })
            
        # 5. Format and return as a clean DataFrame
        df_results = pd.DataFrame(results)
        
        # Optional: Round the float columns to 5 decimals for clean reading
        float_cols = ['Mean Diff (Champ - Model)', 'Adj CI Lower', 'Adj CI Upper']
        df_results[float_cols] = df_results[float_cols].round(5)
        
        return df_results

class DfHeatmapBuilder:
    '''
    This class builds the proper dataframe to heatmap plot.
    It uses level 2 interface from EvaluationMatrix for efficiency.
    '''
    def __init__( self, matrix=None, idx=None, metric="mse"):
        self.matrix = matrix
        self.idx = idx
        self.metric = metric

    def _heatmap_df_from_long_by_example_id(self, long_df=None, arch_col=None, seed_col=None):
        df = (long_df
              #.pivot_table(index=[arch_col, seed_col],  # the dataframe must not have repeated rows when it get here
              # So I prefer to use pivot, since it will give me erros if repeatitions occurr.
              .pivot(index=[arch_col, seed_col],
                           columns=self.matrix.example_id_col,
                           values=self.matrix.value_col,
              #            aggfunc="mean")
                          )
              .sort_index()
              .sort_index(axis=1))
        df.index.names = [arch_col, seed_col]
        df.columns.name = self.matrix.example_id_col
        return df

    def build(self):
        # Layer 3 long
        long = self.matrix.get_long_df(metric=self.metric, dataset_col=self.matrix.test_indexes_col, cache=True)
        df_long = self._heatmap_df_from_long_by_example_id(long_df=long, arch_col=self.matrix.arch_col, seed_col=self.matrix.seed_col)
        return df_long


class DfHeatmapBuilderByDict(DfHeatmapBuilder):
    '''
    This class is an alternative to build the proper dataframe to heatmap plot from the level 1 (dict) interface from EvaluationMatrix.
    It is used to check if level 1 and level 2 interfaces are equivalent.
    '''
    def __init__( self, matrix=None, idx=None, metric="mse", align=None, index_names=None):
        super().__init__(matrix=matrix, idx=idx, metric=metric)
        self.align = align
        if self.align is None:
            self.align=self.matrix.example_id_col

        self.index_names = index_names
        if self.index_names is None:
            self.index_names=(self.matrix.arch_col, self.matrix.seed_col)

    def _heatmap_df_from_dict_by_example_id(self, values_dict=None, examples_dict=None):
        # Build per-run dict: {example_id: value}
        aligned = {}
        for key in values_dict.keys():
            ex_ids = examples_dict[key]
            vals = values_dict[key]
            if len(ex_ids) != len(vals):
                raise ValueError(f"Mismatch for {key}: {len(ex_ids)} ids vs {len(vals)} vals")
            aligned[key] = dict(zip(ex_ids, vals))
    
        df = pd.DataFrame.from_dict(aligned, orient="index")  # columns become example_id
        df.index = pd.MultiIndex.from_tuples(df.index, names=list(self.index_names))
        df.columns.name = self.matrix.example_id_col
        df = df.sort_index(axis=1)  # sort example_id columns
        return df

    def _heatmap_df_from_dict_by_pos(self, values_dict=None, col_name="example_pos"):
        # keys -> rows, list positions -> columns
        df = pd.DataFrame.from_dict(values_dict, orient="index")
        df.index = pd.MultiIndex.from_tuples(df.index, names=list(self.index_names))
        df.columns.name = col_name
        return df 

    def build(self):
        # Layer 1 dicts
        vals_dict = self.matrix.get_metric_values(idx=self.idx, metric=self.metric)
        exid_dict = self.matrix.get_examples(idx=self.idx, dataset_col=self.matrix.test_indexes_col)
    
        if self.align == "pos":
            df_dict = self._heatmap_df_from_dict_by_pos(vals_dict=vals_dict)
        elif self.align == self.matrix.example_id_col:
            df_dict = self._heatmap_df_from_dict_by_example_id(values_dict=vals_dict, examples_dict=exid_dict)
        else:
            raise ValueError("align must be 'pos' or 'example_id'")

        return df_dict




class DfHeatmapExamplesCountBuilder:
    '''
    This class builds the proper dataframe to heatmap plot for counting occurrences.
    '''
    def __init__( self, matrix=None, idx=None, dataset_col=None, n_examples: int = None,
                  grid_shape=(25, 40), 
                 fill_value: int = 0,
                 sort=False,
    ):
        """
        Parameters
        ----------
        matrix : EvaluationMatrix
            Your EvaluationMatrix instance.
        idx : optional
            Slice/tuple for selecting runs. If None, uses all runs in matrix.df.
            Typically you can pass idx = pd.IndexSlice[surviving_archs, :] to restrict to a model subset.
        dataset_col : str, optional
            Column storing test indexes. Defaults to matrix.test_indexes_col.
        n_examples : int
            Total dataset size (default 1000).
        grid_shape : (int, int)
            Desired grid shape (rows, cols). Must multiply to n_examples. Default (20,50).
        drop_last_example : bool
            If you need to reproduce your dataset bug (dropping last test example in each run).
        """
        self.matrix = matrix
        self.idx = idx
        self.sort = sort

        if dataset_col is None:
            self.dataset_col = self.matrix.test_indexes_col
        if n_examples is None:
            n_examples = self.matrix.get_train_dataset_size(drop_last_example=False) + self.matrix.get_test_dataset_size(drop_last_example=False)
        self.n_examples = n_examples
        assert self.n_examples == 1000, f"Dataset size must be 1000"

        self.grid_shape = grid_shape
        self.grid_row_name = "example_id_block_row"
        self.grid_column_name = "example_id_block_col"


    def build(self)-> pd.DataFrame:
        """
        Build a 2D grid (DataFrame) for sns.heatmap where each cell corresponds to a dataset example_id
        and the value is how many times that example_id appears in the test set across all MCCV splits.
        Returns
        -------
        df_grid : pd.DataFrame
        Shape = grid_shape. Index/columns named so sns.heatmap labels are meaningful.
        """

        rows, cols = self.grid_shape
        if rows * cols != self.n_examples:
            raise ValueError(f"grid_shape {self.grid_shape} must multiply to n_examples={self.n_examples}")
    
        # Layer-1 safe API: {(arch, seed): [example_ids]}
        examples_dict = self.matrix.get_examples(
            idx=self.idx,
            dataset_col=self.dataset_col,
        )
    
        # Count occurrences across all selected runs/splits
        # Flatten all ids into one array
        all_ids = []
        for _, ex_ids in examples_dict.items():
            all_ids.extend(ex_ids)
    
        all_ids = np.asarray(all_ids, dtype=np.int32)
        counts = np.bincount(all_ids, minlength=self.n_examples).astype(np.int32)
        if self.sort:
            counts = np.sort(counts)[::-1]
    
        # Reshape in increasing example_id order (0..n_examples-1)
        grid = counts.reshape(rows, cols)
    
        df_grid = pd.DataFrame(grid)
        df_grid.index.name = self.grid_row_name
        df_grid.columns.name = self.grid_column_name
    
        return df_grid 
    

class DfLineplotExamplesCountBuilder(DfHeatmapExamplesCountBuilder):
    '''
    This class builds the proper dataframe to lineplot for counting occurrences.
    '''
    def __init__( self, matrix=None, idx=None, dataset_col=None, n_examples: int = None,
    ):
        super().__init__(matrix=matrix, idx=idx, dataset_col=dataset_col, n_examples=n_examples,
                         grid_shape=None, sort=False)
        self.x_name = "Example ID"
        self.y_name = "Nmb of Occurrences"
                        

    def build(self)-> pd.DataFrame:
        """
        Build a 2D grid (DataFrame) for sns.lineplot where each x value corresponds to a dataset example_id
        and the value is how many times that example_id appears in the test set across all MCCV splits.
        Returns
        -------
        df_grid : pd.DataFrame
        Shape = (1,1000). Index/columns named so sns.heatmap labels are meaningful.
        """

        # Layer-1 safe API: {(arch, seed): [example_ids]}
        examples_dict = self.matrix.get_examples(
            idx=self.idx,
            dataset_col=self.dataset_col,
        )

        # Count occurrences across all selected runs/splits
        # Flatten all ids into one array
        all_ids = []
        for _, ex_ids in examples_dict.items():
            all_ids.extend(ex_ids)
    
        all_ids = np.asarray(all_ids, dtype=np.int32)
        counts = np.bincount(all_ids, minlength=self.n_examples).astype(np.int32)
    
        df = pd.DataFrame(counts)
        df = pd.DataFrame({
            self.x_name: np.arange(len(counts)),
            self.y_name: counts
            })
    
        return df
    

class DfHeatmapExamplesMetricErrorBuilder:
    '''
    This class builds the proper dataframe to heatmap plot per example real-valued information 
    as mean mse error, mean std error, mean max error and mean min error.
    '''
    def __init__( self, matrix=None, arch_subset=None, dataset_col=None, n_examples: int = None,
                  grid_shape=(25, 40), 
                  fill_value: int = None,
                  agg_func: str = 'mean',
                  metric: str = 'mse',
                  sort_by_mean=False,
    ):
        """
        Parameters
        ----------
        matrix : EvaluationMatrix
            Your EvaluationMatrix instance.
        arch_name : optional
            Filter by arch_name list.
        dataset_col : str, optional
            Column storing the real-valued of interest. Defaults to 'mse'.
        n_examples : int
            Total dataset size (default 1000).
        grid_shape : (int, int)
            Desired grid shape (rows, cols). Must multiply to n_examples. Default (20,50).
        drop_last_example : bool
            If you need to reproduce your dataset bug (dropping last test example in each run).
        fill_value : int
            Fill value for missing examples (default np.NaN).
        """
        self.matrix = matrix
        self.arch_subset = arch_subset
        self.idx = self.matrix.get_index_slice()[arch_subset,:]

        if dataset_col is None:
            self.dataset_col = "mse"
        if n_examples is None:
            n_examples = self.matrix.get_train_dataset_size(drop_last_example=False) + self.matrix.get_test_dataset_size(drop_last_example=False)
        self.n_examples = n_examples
        assert self.n_examples == 1000, f"Dataset size must be 1000"

        self.grid_shape = grid_shape
        self.grid_row_name = "example_id_block_row"
        self.grid_column_name = "example_id_block_col"

        self.fill_value = fill_value
        if self.fill_value is None:
            self.fill_value = np.nan

        self.agg_func = agg_func
        self.metric = metric
        self.sort_by_mean = sort_by_mean


    def build(self)-> pd.DataFrame:
        """
        Build a 2D grid (DataFrame) for sns.heatmap where each cell corresponds to a dataset example_id
        and the value is how many times that example_id appears in the test set across all MCCV splits.
        Returns
        -------
        df_grid : pd.DataFrame
        Shape = grid_shape. Index/columns named so sns.heatmap labels are meaningful.
        """

        rows, cols = self.grid_shape
        if rows * cols != self.n_examples:
            raise ValueError(f"grid_shape {self.grid_shape} must multiply to n_examples={self.n_examples}")
    
        df_long = self.matrix.get_long_df(metric=self.metric, arch_subset=self.arch_subset)

        agg_func_metric_per_example = df_long.groupby(self.matrix.example_id_col)[self.matrix.value_col].agg(self.agg_func)

        if self.sort_by_mean:
            ordering = df_long.groupby(self.matrix.example_id_col)[self.matrix.value_col].agg('mean').sort_values(ascending=False).index.tolist()
        else:
            ordering = agg_func_metric_per_example.index.tolist()

        all_ids = list(range(self.n_examples))
        missing_ids = [i for i in all_ids if i not in ordering] # examples in none of test datasets

        if self.sort_by_mean:
            final_ordering = ordering + missing_ids
        else:
            final_ordering = list(range(self.n_examples))

        assert len(ordering) == len(agg_func_metric_per_example), f"[ INTERNAL ERROR ] {len(ordering)} != {len(agg_func_metric_per_example)}"
        assert len(final_ordering) == self.n_examples, f"[ INTERNAL ERROR ] final_ordering has size {len(final_ordering)}. Expected: {self.n_examples}"

        matriz_metric = np.full(self.n_examples, self.fill_value)

        for i, ii in enumerate(final_ordering):
            assert ii>=0 and ii<self.n_examples, f"[ INTERNAL ERROR ] i: {i}, ii: {ii}"
            if ii not in missing_ids:
                matriz_metric[i] = agg_func_metric_per_example[ii]

        matriz_25x40 = matriz_metric.reshape(self.grid_shape)
        df_heatmap = pd.DataFrame(
            matriz_25x40,
            index=[f"{i}" for i in range(25)],
            columns=[f"{i}" for i in range(40)]
        )
        return df_heatmap
    

@dataclass(frozen=True)
class LongViewKey:
    metric: str
    dataset_col: str
    arch_subset: Optional[Tuple[str, ...]]
    seed_subset: Optional[Tuple[Any, ...]]
    drop_last_example: bool


class EvaluationMatrix:
    """
    Layer 1 (safe): idx-based API returning dict keyed by (arch_name, seed).
    Layer 2: generator of atomic observations (arch, seed, example_id, value).
    Layer 3: long dataframe materialization + caching.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mlflow_client: Any,
        arch_col: str = "arch_name",
        arch_name_paper_col: str = "arch_name_paper",
        seed_col: str = "seed",
        run_id_col: str = "run_id",
        test_indexes_col: str = "test_indexes",
        train_indexes_col: str = "train_indexes",
        example_id_col: str= "example_id",
        value_col: str= "value",
        metric_vector_fetcher: Optional[MetricVectorFetcher] = None,
        enable_cache: bool = True,
        cache_max_items: int = 8,
        alpha=0.05,
    ):
        self.mlflow_ = mlflow_client

        self.arch_col = arch_col
        self.arch_name_paper_col = arch_name_paper_col
        self.seed_col = seed_col
        self.run_id_col = run_id_col
        self.test_indexes_col = test_indexes_col
        self.train_indexes_col = train_indexes_col
        self.example_id_col = example_id_col
        self.value_col = value_col
        self.alpha = alpha

        self.df = df.set_index([self.arch_col, self.seed_col]).sort_index()

        if metric_vector_fetcher is None:
            raise ValueError("metric_vector_fetcher must be provided.")
        self._fetch_metric_vector = metric_vector_fetcher

        self.enable_cache = enable_cache
        self.cache_max_items = cache_max_items
        self._long_cache: Dict[LongViewKey, pd.DataFrame] = {}

    # -----------------------------
    # Layer 1: safe idx-based access
    # -----------------------------
    @staticmethod
    def get_index_slice():
        return pd.IndexSlice

    def get_runs(self, idx=None) -> pd.DataFrame:
        '''
        Get a Dataframe with the sliced indexes attributes.
        '''
        runs = self.df.loc[idx, :]
        if isinstance(runs, pd.Series):
            runs = runs.to_frame().T
        return runs

    def get_attr(self, idx=None, attr_name: str = "") -> pd.Series:
        '''
        Get a Series with the sliced attribute.
        '''
        return self.get_runs(idx=idx)[attr_name]

    def get_metric(self, idx=None, metric: str = "mse") -> pd.Series:
        '''
        Get a Series with the sliced metric.
        '''
        return self.get_attr(idx=idx, attr_name=metric)

    def get_arch_names(self) -> list[str]:
        return self.df.index.get_level_values(self.arch_col).unique().tolist()

    def get_seeds(self, arch_name: Optional[str] = None) -> list[Any]:
        if arch_name is None:
            return self.df.index.get_level_values(self.seed_col).unique().tolist()
        return (
            self.df.xs(arch_name, level=self.arch_col)
                  .index.get_level_values(self.seed_col)
                  .tolist()
        )

    def get_metric_values(self, idx=None, idx2=None, metric='mse'):
        '''
        Get a dict (arch_name,seed) : list_of_metrics_values. 
        '''
        metric = metric + '_values'
        df_aux = self.df.loc[idx,:]
        df_aux = df_aux[[self.run_id_col]].map(lambda x: EvalUtils.get_metric_values_from_run(mlflow_=self.mlflow_, run_id=x, metric=metric)).rename(columns={self.run_id_col: metric})
        values=df_aux.to_dict()
        return values

    def get_train_dataset_size(self, drop_last_example=True):
        return self.get_dataset_size(dataset_col=self.train_indexes_col, drop_last_example=drop_last_example)

    def get_test_dataset_size(self, drop_last_example=True):
        return self.get_dataset_size(dataset_col=self.test_indexes_col, drop_last_example=drop_last_example)

    def get_dataset_size(self, dataset_col=None, drop_last_example=True):
        arch = self.get_arch_names()[0]
        seed = self.get_seeds(arch)[0]
        ex = self.get_examples(idx=(arch, seed), dataset_col=dataset_col, drop_last_example=drop_last_example)
        values = next(iter(ex.values()))
        return len(values)

    def get_total_number_of_metrics_values(self,  metric="ssim"):
        one_arch_name = self.get_arch_names()[0]
        one_seed = self.get_seeds(arch_name=one_arch_name)[0]
        idx = self.get_index_slice()

        values = self.get_metric_values(idx=idx[one_arch_name,one_seed],idx2=None,metric=metric)
        values = next(iter(values.values()))
        size = len(values)
        assert size == 99, f"[INTERNAL ERROR] size must be 99 but was {size}"
        assert size == self.get_test_dataset_size(), f"[INTERNAL ERROR] size must be {self.get_test_dataset_size()} but was {size}"
        return size

    def get_nmb_of_unique_arch_names(self) -> int:
        return len(self.get_arch_names())


    # --- parsing indexes (single source of truth) ---
    @staticmethod
    def _parse_indexes(raw) -> list[int]:
        if raw is None:
            return []
        if isinstance(raw, (list, tuple, np.ndarray)):
            return [int(x) for x in raw]
        if isinstance(raw, str):
            s = raw.strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1].strip()
                if not s:
                    return []
                if "," in s:
                    return [int(x.strip()) for x in s.split(",") if x.strip()]
                return [int(x) for x in s.split() if x.strip()]
            return [int(x) for x in s.split() if x.strip()]
        return [int(raw)]

    # --- Layer 1 public: get_examples returns dict keyed by (arch, seed) ---
    def get_examples(
        self,
        idx=None,
        dataset_col: Optional[str] = None,
        idx2=None,
        drop_last_example: bool = True,
    ) -> Dict[Tuple[str, Any], list[int]]:
        """
        Layer 1: returns {(arch, seed): [example_ids]} for all runs in idx.
        """
        if dataset_col is None:
            dataset_col = self.test_indexes_col

        runs = self.get_runs(idx=idx)
        out: Dict[Tuple[str, Any], list[int]] = {}

        for (arch, seed), row in runs.iterrows():
            ex = self._parse_indexes(row[dataset_col])

            if drop_last_example and len(ex) > 0:
                ex = ex[:-1]
            if idx2 is not None:
                ex = ex[idx2]

            out[(arch, seed)] = ex

        return out

    # --- Layer 1 public: metric vectors as dict keyed by (arch, seed) ---
    def get_metric_values(
        self,
        idx=None,
        idx2=None,
        metric: str = "mse",
    ) -> Dict[Tuple[str, Any], list[float]]:
        """
        Layer 1: returns {(arch, seed): [metric_values]} for all runs in idx.
        """
        runs = self.get_runs(idx=idx)
        out: Dict[Tuple[str, Any], list[float]] = {}

        metric_name = f"{metric}_values"
        for (arch, seed), row in runs.iterrows():
            run_id = row[self.run_id_col]
            vals = list(self._fetch_metric_vector(self.mlflow_, run_id, metric_name))

            if idx2 is not None:
                vals = vals[idx2]

            out[(arch, seed)] = vals

        return out


    def get_mccv_summary(self) -> dict:
        """
        Run-level summary using corrected CI over the 10 MCCV splits.
        Uses aggregate metrics already stored in df (mse/mae/ssim means per run).
        """
        compute_corrected_ci_partial = partial(
            EvalUtils.compute_corrected_ci,
            alpha=self.alpha,
            n_train=self.get_train_dataset_size(),
            n_test=self.get_test_dataset_size(),
        )

        # Reset only the seed level to group by arch
        df_unlocked = self.df.reset_index(level=self.seed_col)

        df_combined = df_unlocked.groupby(level=self.arch_col).agg(
            seeds_list=(self.seed_col, list),
            n=(self.seed_col, "size"),

            mse=("mse", "mean"),
            mse_std=("mse", "std"),
            mse_max=("mse", "max"),
            mse_min=("mse", "min"),
            mse_95ci=("mse", compute_corrected_ci_partial),

            mae=("mae", "mean"),
            mae_std=("mae", "std"),
            mae_max=("mae", "max"),
            mae_min=("mae", "min"),
            mae_95ci=("mae", compute_corrected_ci_partial),

            ssim=("ssim", "mean"),
            ssim_std=("ssim", "std"),
            ssim_max=("ssim", "max"),
            ssim_min=("ssim", "min"),
            ssim_95ci=("ssim", compute_corrected_ci_partial),

            conv_kernel_size=("conv_kernel_size", "last"),
            conv_dilation_size=("conv_dilation_size", "last"),
            transposed_conv_kernel_size=("transposed_conv_kernel_size", "last"),
            arch_name_paper=(self.arch_name_paper_col, "last")
        )
        return df_combined.to_dict("index")
    
    def get_examples_in_splits_count(self) -> pd.Series:
        """
        Counts how many times each example_id appears across seeds (test splits).
        """
        # Keep one row per seed (test split definition per seed)
        df_aux = self.df.reset_index().drop_duplicates(subset=[self.seed_col], keep="last").copy()
        counts = df_aux[self.test_indexes_col].apply(self._parse_indexes).explode().value_counts()

        # If you know the global dataset size, reindex appropriately.
        # Otherwise return raw counts.
        return counts.sort_index()
    

    def get_metric_values_by_metric( self, idx=None, metrics=("mse", "mae", "ssim"), idx2=None):
        """
        Layer‑1 API (backward compatible)
    
        Returns:
            {
              "mse_values": {(arch, seed): [...]},
              "mae_values": {(arch, seed): [...]},
              ...
            }
        """
        out = {}
        for metric in metrics:
            key = f"{metric}_values"
            out[key] = self.get_metric_values(
                idx=idx,
                metric=metric,
                idx2=idx2
            )
        return out
                                           

    # -----------------------------
    # Layer 1.5: single-run helpers (internal)
    # -----------------------------
    def get_examples_one(
        self,
        key: Tuple[str, Any],
        dataset_col: Optional[str] = None,
        idx2=None,
        drop_last_example: bool = True,
    ) -> list[int]:
        """
        Internal helper: returns [example_ids] for a single (arch, seed).
        Does NOT change your Layer-1 API.
        """
        return self.get_examples(
            idx=key,
            dataset_col=dataset_col,
            idx2=idx2,
            drop_last_example=drop_last_example,
        )[key]
    

    # -----------------------------
    # Layer 2: observation view (generator)
    # -----------------------------
    def iter_long_rows(
        self,
        metric: str = "mse",
        dataset_col: Optional[str] = None,
        arch_subset: Optional[Sequence[str]] = None,
        seed_subset: Optional[Sequence[Any]] = None,
        drop_last_example: bool = True,
    ) -> Iterable[tuple[str, Any, int, float]]:
        """
        Yields (arch, seed, example_id, value) for each test example in each run.
        """
        if dataset_col is None:
            dataset_col = self.test_indexes_col

        runs = self.df
        if arch_subset is not None:
            runs = runs.loc[runs.index.get_level_values(self.arch_col).isin(arch_subset)]
        if seed_subset is not None:
            runs = runs.loc[runs.index.get_level_values(self.seed_col).isin(seed_subset)]

        metric_name = f"{metric}_values"

        for (arch, seed), row in runs.iterrows():
            run_id = row[self.run_id_col]
            ex_ids = self._parse_indexes(row[dataset_col])
            if drop_last_example and len(ex_ids) > 0:
                ex_ids = ex_ids[:-1]

            vals = list(self._fetch_metric_vector(self.mlflow_, run_id, metric_name))

            if len(vals) != len(ex_ids):
                raise ValueError(
                    f"Length mismatch for (arch={arch}, seed={seed}): "
                    f"{len(vals)} {metric} values vs {len(ex_ids)} example ids."
                )

            for eid, v in zip(ex_ids, vals):
                yield (arch, seed, int(eid), float(v))

    # -----------------------------
    # Layer 3: analysis view (long df + caching)
    # -----------------------------
    def get_long_df(
        self,
        metric: str = "mse",
        dataset_col: Optional[str] = None,
        arch_subset: Optional[Sequence[str]] = None,
        seed_subset: Optional[Sequence[Any]] = None,
        drop_last_example: bool = True,
        cache: Optional[bool] = None,
        dtype_float: str = "float32",
    ) -> pd.DataFrame:
        if dataset_col is None:
            dataset_col = self.test_indexes_col

        use_cache = self.enable_cache if cache is None else cache
        key = LongViewKey(
            metric=metric,
            dataset_col=dataset_col,
            arch_subset=tuple(arch_subset) if arch_subset is not None else None,
            seed_subset=tuple(seed_subset) if seed_subset is not None else None,
            drop_last_example=drop_last_example,
        )

        if use_cache and key in self._long_cache:
            return self._long_cache[key]

        rows = list(self.iter_long_rows(
            metric=metric,
            dataset_col=dataset_col,
            arch_subset=arch_subset,
            seed_subset=seed_subset,
            drop_last_example=drop_last_example,
        ))

        long_df = pd.DataFrame(rows, columns=[self.arch_col, self.seed_col, self.example_id_col, self.value_col])
        long_df[self.arch_col] = long_df[self.arch_col].astype("category")
        # seed might be non-int; use safe conversion
        try:
            long_df[self.seed_col] = long_df[self.seed_col].astype("int16")
        except Exception:
            pass
        long_df[self.example_id_col] = long_df[self.example_id_col].astype("int32")
        long_df[self.value_col] = long_df[self.value_col].astype(dtype_float)

        if use_cache:
            self._cache_put(key, long_df)

        return long_df

    def _cache_put(self, key: LongViewKey, df: pd.DataFrame):
        if not self.enable_cache:
            return
        if len(self._long_cache) >= self.cache_max_items:
            oldest_key = next(iter(self._long_cache.keys()))
            self._long_cache.pop(oldest_key, None)
        self._long_cache[key] = df

    def clear_cache(self):
        self._long_cache.clear()



def test_arch_name2arch_name_paper():
    # Casos de teste: (use_transformer, conv_kernel_size, conv_dilation_size,
    #                  transposed_conv_kernel_size, optimizer)
    test_cases = [
        ("False", 3, 1, 2, "Adam"),
        ("False", 5, 2, 2, "AdamW"),
        ("False", 7, 4, 4, "Adam"),
        ("True",  3, 1, 2, "Adam"),
        ("True",  5, 2, 2, "AdamW"),
        ("True",  7, 4, 4, "AdamW"),
    ]

    for use_transformer, cks, cds, tcks, opt in test_cases:
        # 1) Gera o nome interno (fonte do parsing)
        arch_name = EvalUtils.get_model_name(
            use_transformer=use_transformer,
            conv_kernel_size=cks,
            conv_dilation_size=cds,
            transposed_conv_kernel_size=tcks,
            optmizer=opt,
        )

        # 2) Gera o nome esperado do paper diretamente
        expected_paper_name = EvalUtils.get_model_name_paper(
            use_transformer=use_transformer,
            conv_kernel_size=cks,
            conv_dilation_size=cds,
        )

        # 3) Aplica o mapeamento via método sob teste
        actual_paper_name = EvalUtils.arch_name2arch_name_paper(arch_name)

        assert actual_paper_name == expected_paper_name, (
            f"Falha no mapeamento.\n"
            f"  arch_name        = {arch_name!r}\n"
            f"  expected (paper) = {expected_paper_name!r}\n"
            f"  actual   (paper) = {actual_paper_name!r}"
        )

    # 4) Testes de valores literais explícitos (não dependem de get_model_name)
    literal_cases = [
        ("unet.cks_3x3.cds_1x1.tcks_2x2.opt_Adam",   "U-Net, k=3, r=1"),
        ("unet.cks_5x5.cds_2x2.tcks_2x2.opt_AdamW",  "U-Net, k=5, r=2"),
        ("transunet.cks_7x7.cds_4x4.tcks_4x4.opt_Adam", "TransUnet, k=7, r=4"),
    ]
    for arch_name, expected in literal_cases:
        actual = EvalUtils.arch_name2arch_name_paper(arch_name)
        assert actual == expected, (
            f"Falha no caso literal.\n"
            f"  arch_name = {arch_name!r}\n"
            f"  expected  = {expected!r}\n"
            f"  actual    = {actual!r}"
        )

    # 5) Testes de erro: arch_name malformado deve levantar ValueError
    invalid_cases = [
        "resnet.cks_3x3.cds_1x1.tcks_2x2.opt_Adam",   # arch desconhecida
        "unet.kernel_3x3.cds_1x1.tcks_2x2.opt_Adam",  # token cks inválido
        "unet.cks_3x3.dilation_1x1.tcks_2x2.opt_Adam" # token cds inválido
    ]
    for bad_name in invalid_cases:
        try:
            EvalUtils.arch_name2arch_name_paper(bad_name)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Esperava ValueError para arch_name inválido: {bad_name!r}"
            )

    print("Todos os testes de arch_name2arch_name_paper passaram.")