
"""@package analyse_evaluation_results.py

   1. For each mccv/arch on test set
      1.1 mse mean, max, min, std, 95-CI (CLT-based)
      1.2 show mse values visually ()
   2. For each arch over all mccv splits
      2.1 mse mean, max, min, std, CI (10 samples)
   3. For each mccv see if max and min examples are the same over archs


   4. For each mccv, choose some examples to see how archs behave 

   ARCHS DEFS:
     - use transformer? (number of layers is just an additional info)
     - dilation size (1, 2, 3)
     - kernel size (3x3, 5x5, 7x7, 9x9, 13x13)
     * ignore transpose_conv_kernel_size
     * ignore number of unet layers (since there is just 5 for now)

   ARCHS NAMES:
     - <unet|transunet>,
     - cks_<3x3|5x5|7x7|13x13>, kernel size
     - cds_<1x1|2x2|3x3>, dilation size


"""


############################
# basic stuff
import os
from datetime import datetime
import time
import random
import sys
import struct
from glob import glob
import pickle
import json
import pandas as pd
import warnings

############################
# Latex
from pylatex import Document, Section, Subsection, Figure, Command
from pylatex.utils import NoEscape

############################
# science stuff
from scipy import ndimage
from scipy import interpolate
import scipy.stats as st
#from skimage import io as skio
from scipy import fftpack

import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
# import cv2

#get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors0


############################
# image
from PIL import Image
import segyio


############################
# experiment tracking
import mlflow
from mlflow import MlflowClient


############################
# ml stuff
import sklearn
from sklearn.model_selection import GridSearchCV

############################
# Parallelism
#from multiprocessing import Pool
#from torch.multiprocessing.pool import ThreadPool
from multiprocessing.pool import ThreadPool

############################
# internal modules
from src.experiment_tracking.mlflow_utils import MLFlowUtils, MLFlowDataAndModelsLogger
from src.experiment_tracking.mlflow import MLFlow
from src.experiment_tracking.mlflow_skorch_callbacks import ParamLogger

from src.processing.gs_grid import RefitGrid, GridSearchElementResult
from src.processing.task_processing import Task, BuildTasks, TaskProcessingGlobals, RefitTaskProcessing
from src.processing.ml_models_builder import SkorchTransUNetModelBuilder, ScoredSkorchModel
from src.processing.ml_models import TransUNet
from src.processing.ml_models_utils import MLModelsUtils

from src.utl.logging import Log as log

from src.utl.classes import Utils as utl
from src.utl.classes import Graph as graph

from src.visualization.data_viewer import DataAndModelsViewer as viewer
from src.visualization.gs_viewer import GSViewer as gsviewer

from src.preprocessing.pipeline import Preprocessing

from src.apps.command_line_interface import AnalyseEvaluationsCLI
from src.apps.global_vars import AnalyseEvaluationsGlobalVars

from src.pdf.sections_builder import SectionsBuilder
from src.pdf.eval_sections_builder import EvalListBuilder
from src.pdf.gs_sections_builder import GSPDFBuilder
from src.pdf.eval_utils import EvaluationMatrix, DfHeatmapBuilder, DfHeatmapBuilderByDict
from src.pdf.eval_utils import EvalUtils as eval_utl 
#parse_string, make_metric_vector_fetcher, get_metric_values_from_run
#from src.pdf.section import Section as mySection

#from src.preprocessing.local_pipeline import PreprocessingPipe
#from src.processing.train_data import TrainDataInfo, TrainData, Jobs, ProductionOutlierClassification

def filter_remove_conv_kernel_size_9(df):
    """Remove linhas onde o modelo é transunet E conv_dilation_size > 1."""
    # Condição que identifica o que queremos REMOVER
    condicao_remover = (df['conv_kernel_size'] == 9)
    
    # Mantemos apenas o inverso (~) da condição de remoção
    return df[~condicao_remover]


def filter_remove_transunet_with_dilation(df):
    """Remove linhas onde o modelo é transunet E conv_dilation_size > 1."""
    # Condição que identifica o que queremos REMOVER
    condicao_transunet = (df['arch_name'].str.contains('transunet', case=False, na=False))
    condicao_r_gt_1 = (df['conv_dilation_size'] > 1)
    condicao_remover = condicao_transunet & condicao_r_gt_1
    
    # Mantemos apenas o inverso (~) da condição de remoção
    return df[~condicao_remover]


#def filter_only_dilation_one(df):
#    """Mantém apenas registros onde conv_dilation_size é igual a 1."""
#    return df[df['conv_dilation_size'] == 1]


def process_dataframes_recursive(d, keep, is_debug, filters=None):
    """Navega recursivamente aplicando os filtros customizados, drop_duplicates e reset_index."""
    for key, value in d.items():
        if key == "self":
            df_current = value
            
            # Aplica cada filtro da lista sequencialmente (se houver algum)
            if filters:
                for filter_func in filters:
                    df_current = filter_func(df_current)
            
            # Aplica as regras de negócio fixas (duplicadas e índice)
            df_updated = drop_duplicates(df=df_current, keep=keep, debug=is_debug)
            d[key] = df_updated.reset_index(drop=True)
            
        elif isinstance(value, dict):
            # Continua descendo na árvore
            process_dataframes_recursive(value, keep, is_debug, filters)

            
def inject_filtered_subsets_recursive(d):
    """Gera e injeta os novos sub-dataframes ('best_model' e 'best_kernel') em cada nível."""
    for key, value in d.items():
        if isinstance(value, dict) and "self" in value:
            current_df = value["self"]
            
            # Cria e adiciona as novas chaves no mesmo nível do 'self'
            value["best_model"] = filter_df_best_per_akr_model(df=current_df)
            value["best_kernel"] = filter_df_best_per_kernel(df=current_df)
            
            # Continua a busca nos sub-dicionários (ex: 'adam', 'adamw')
            inject_filtered_subsets_recursive(value)


def select_ith_mccv_session(df=None, session_index=-1, time_gap='1h', debug=False):
    """
    Groups MLflow runs into sessions based on time proximity.
    - Validates that no raw session exceeds 13 executions (prevents merging distinct sessions).
    - Keeps only the first execution per seed if a session has duplicate runs.
    - Discards sessions with fewer than 8 executions.
    - Renumbers surviving sessions sequentially before selecting the target index.
    """
    df_clean = df.copy()
    
    # 1. Ensure start_time is a proper datetime object
    df_clean['start_time'] = pd.to_datetime(df_clean['start_time'])
    
    # 2. Sort chronologically per architecture
    df_clean = df_clean.sort_values(by=['arch_name', 'start_time'])
    
    # 3. Calculate time differences and identify raw boundaries
    time_diff = df_clean.groupby('arch_name')['start_time'].diff()
    is_new_session = time_diff.isna() | (time_diff > pd.Timedelta(time_gap))
    
    # 4. Create initial (raw) session IDs
    df_clean['raw_session_id'] = is_new_session.groupby(df_clean['arch_name']).cumsum() - 1
    
    # --- NEW RULE: Check for merged sessions (> 13 executions) ---
    # We must do this BEFORE deduplication so we can catch massively merged sessions
    raw_run_counts = df_clean.groupby(['arch_name', 'raw_session_id']).size()
    invalid_sessions = raw_run_counts[raw_run_counts > 13]
    
    if not invalid_sessions.empty:
        raise ValueError(
            f"Data Integrity Error: Found sessions with more than 13 executions!\n"
            f"The time_gap of '{time_gap}' is likely too large and merged separate MCCV sessions.\n"
            f"Check these specific architectures and raw sessions:\n{invalid_sessions}"
        )

    # --- RULE 1: Handle duplicated seeds (Keep first execution per seed) ---
    # If a session had 11-13 runs (e.g., due to restarts), this safely trims it back down to <= 10
    df_clean = df_clean.drop_duplicates(subset=['arch_name', 'raw_session_id', 'seed'], keep='first')
    
    # --- RULE 2: Discard sessions with < 8 executions ---
    df_clean = df_clean.groupby(['arch_name', 'raw_session_id']).filter(lambda x: len(x) >= 8)
    
    if df_clean.empty:
        return pd.DataFrame(columns=df.columns)

    # --- RULE 3: Renumber surviving sessions to close gaps ---
    df_clean['session_id'] = (
        df_clean.groupby('arch_name')['raw_session_id']
        .rank(method='dense')
        .astype(int) - 1
    )

    if debug:
        final_counts = df_clean.groupby(['arch_name', 'session_id']).size()
        print("Surviving executions per gapless session:\n", final_counts)

    # 5. Filter for the requested session index per architecture
    def get_target_session(group):
        sessions = group['session_id'].unique()
        try:
            target = sessions[session_index]
            return group[group['session_id'] == target]
        except IndexError:
            return pd.DataFrame()

    final_df = df_clean.groupby('arch_name', group_keys=False).apply(get_target_session)
    
    # Clean up by dropping temporary ID columns
    return final_df.drop(columns=['raw_session_id', 'session_id']).copy()


def drop_duplicates(df=None, keep='first', debug=False ):
    if keep == 'first':
        index=0
    elif keep == 'last':
        index=-1
    else:
        index=1
    return select_ith_mccv_session(df=df, session_index=index, time_gap='7d', debug=debug)

#    df_clean = df.sort_values(by='start_time', ascending=True).drop_duplicates(
#        subset=['arch_name', 'seed'], 
#        keep=keep
#    ).copy()
#
#    if debug:
#        counts = df_clean.groupby('arch_name')['seed'].nunique()
#        print(f"counts < 10 = {counts[counts < 10]}")
#    #assert (counts > 8).all(), counts[counts <= 8]
#    #assert (counts == 10).all(), counts[counts != 10]
#
#    return df_clean

def filter_df_optimizer(df, to_pass='Adam'):
    # O regex busca exatamente: ponto + o nome desejado + aspas simples
    # Exemplo para 'Adam': \.Adam'
    # Esta implementacao se baseia na expectativa dos nomes terem esse formato:
    # <class 'torch.optim.adamw.AdamW'>
    regex_pattern = rf"\.{to_pass}'"
    
    filtro = df['optimizer'].str.contains(regex_pattern, regex=True)
    return df[filtro].copy()

def filter_df_best_per_akr_model(df=None):
    '''
    a=arch
    k=kernel size
    r=dilation size
    '''
    # 1. Identifica o nome da melhor arquitetura para cada trio akr
    best_archs = (
        df.groupby(['arch_name', 'conv_kernel_size', 'use_transformers', 'conv_dilation_size'])['mse']
        .mean()
        .reset_index()
        .sort_values('mse')
        .drop_duplicates(subset=['conv_kernel_size','use_transformers', 'conv_dilation_size'])['arch_name']
    )
    #print("-------------------------------")
    #print(best_archs)
    #print("-------------------------------")
    #os._exit(0)
    # 2. Filtra o DataFrame original mantendo apenas as linhas dessas arquiteturas
    return  df[df['arch_name'].isin(best_archs)].copy()

def filter_df_best_per_kernel(df=None):
    # 1. Identifica o nome da melhor arquitetura para cada conv_kernel_size
    best_archs = (
        df.groupby(['arch_name', 'conv_kernel_size', 'use_transformers'])['mse']
        .mean()
        .reset_index()
        .sort_values('mse')
        .drop_duplicates(subset=['conv_kernel_size','use_transformers'])['arch_name']
    )
    #print("-------------------------------")
    #print(best_archs)
    #print("-------------------------------")
    #os._exit(0)
    # 2. Filtra o DataFrame original mantendo apenas as linhas dessas arquiteturas
    return  df[df['arch_name'].isin(best_archs)].copy()

def split_per_optimizer(df=None):
    df_adam = filter_df_optimizer(df=df,to_pass='Adam')
    df_adamw = filter_df_optimizer(df=df,to_pass='AdamW')
    return df_adam, df_adamw 


class ShallowUnetFilter():
    def __init__(self):
        self.df=None

    def filter(self, df):
        self.df=df
        idxs = self.df['weight_decay'] != 0.001
        idxs = idxs & (self.df['weight_decay'] != 0.01)
        idxs = idxs & (self.df['weight_decay'] != 0.1)
        df2 = self.df[idxs] 
        log.info(f"idxs = {idxs}")

        #return self.df[ (self.df['weight_decay'] != 0.001) & (self.df['weight_decay'] != 0.01) & (self.df['weight_decay'] != 0.1) ]
        return df2

class DeepUnetFilter():
    def __init__(self):
        self.df = None 

    def filter(self, df):
        self.df=df
        idxs = self.df['weight_decay'] != 0.1
        df2 = self.df[idxs] 
        return df2

class ShallowTransunetFilter():
    def __init__(self):
        self.df = None 
        
    def filter(self, df):
        self.df=df
        df2 = self.df
        return df2

class DeepTransunetFilter():
    def __init__(self):
        self.df = None 

    def filter(self, df):
        self.df=df
        df2 = self.df
        return df2

def make_evaluation_report( df=None, mlflow_=None, i_name='arch_name', j_name='seed', title='', author='', dataset='', alpha=0.05, label='', verbose=True):
    metric_vector_fetcher = eval_utl.make_metric_vector_fetcher(eval_utl.get_metric_values_from_run)
    matrix = EvaluationMatrix(df=df, mlflow_client=mlflow_, metric_vector_fetcher=metric_vector_fetcher)
    #print(f"mccv_summary = {eval_utl.print_mccv_summary(matrix.get_mccv_summary(), verbose=False)}")
    #matrix = EvaluationMatrix(df=df, mlflow_=mlflow_, i='arch_name', j='seed')
    list_of_sections = EvalListBuilder.build(df=df, matrix=matrix, alpha=alpha, dataset=dataset, label=label, verbose=verbose)
    sections = SectionsBuilder.build_from_list(l=list_of_sections)
    return GSPDFBuilder.build_from_sections(title=title, author=author, sections=sections, verbose=verbose)
##  UTILITIES
 
def create_pd_from_runs(mlflow_=None, runs=None):
    # run_name, run_id, parent_run_id, parent_run_name,
    # minibatch_size, max_epochs, lr, weight_decay, 
    # seed, dataset_type, n_filters_init, kernel_size, att_in_channels,
    # use_transformers, att_img_size, num_layers, num_heads, mlp_dim, hidden_size
    log.info(f"Creating pandas dataframe from {len(runs)} runs...")
    list_of_dicts = list()
    for i, (k,v) in enumerate(runs.items()):
        #log.printMessage(f"[{i}] - [{k}]")
        # get att_config structure
        run = v
        #log.info(f"[RUN][{run}]")
        run_dict = run.data.to_dictionary()
        if run_dict['params'] is not None and len(run_dict['params'].keys()) > 0:
            assert run_dict['params'] is not None, f"Error! There is no 'params' attribute associated to this run!"
            assert len(run_dict['params'].keys()) > 0, f"Error! run_dict['params'] has zero attributes!"
            params_dict = eval_utl.get_params(run_dict['params']) 
            metrics_dict = eval_utl.get_metrics(run_dict['metrics']) 
            info_dict = eval_utl.build_info( run_id=run.info.run_id, run_name=run.info.run_name, start_time=run.info.start_time, end_time=run.info.end_time)
            list_of_dicts.append(params_dict|metrics_dict|info_dict)
    log.ok()


    ret = pd.DataFrame(list_of_dicts)
    column_dtypes = {
        'seed' : 'int',
        'lr' : 'float',
        'optimizer' : 'string',
        'weight_decay' : 'float',
        'minibatch_size' : 'int',
        'max_epochs' : 'int',
        'n_filters_init' : 'int',
        'conv_kernel_size' :'int',
        'transposed_conv_kernel_size' :'int',
        'conv_dilation_size' :'int',
        'att_in_channels' : 'int',
        'use_transformers' :'string',
        'att_hidden_size' : 'int',
        'att_num_heads' : 'int',
        'att_num_layers' : 'int',
        'att_att_dropout_rate' : 'float',
        'att_dropout_rate' :'float',
        'att_mlp_dim' : 'int',
        'run_id' : 'string',
        'run_name' : 'string',
        'data_type' : 'string',
        'start_time' : 'datetime64[ms]',
        'end_time' : 'datetime64[ms]',

#        'test_indexes' : 'string',
#        'train_indexes' : 'string',
    }
    ret = ret.astype(column_dtypes)

    # Add arch_name for each run
    column_arch_name = ret.apply(eval_utl.get_model_name_from_df_line, axis=1)
    column_model_name_convention = ret.apply(eval_utl.get_model_name_paper_from_df_line, axis=1)
    ret['arch_name'] = column_arch_name
    ret['arch_name_paper'] = column_model_name_convention

    #for k,v in ret.items():
    #    print(f"k = {k}, v = {v}") #['conv_dilation_size'],v['transposed_conv_kernel_size']}")


    return ret[ ret['data_type'] == 'shallow'], ret[ ret['data_type'] == 'deep'] 


def print_all_runs(runs):
    print(f"NUMBER OF GS RUNS:  {len(runs)}")
    for i, (k,v) in enumerate(runs.items()):
        log.printMessage(f"[{i}] - [{k}]")
        children = v["children"]
        if children is not None:
            for j, (kk, vv) in enumerate(children.items()):
                print(f"[{j}] {kk}, {vv}")
        else:
            print("DOESN'T HAVE CHILDREN")

def print_run_info(runs=None, verbose=True):
    log.info(f"NUMBER OR RUNS: {len(runs)}")
    for i, (k,r) in enumerate(runs.items()):
        log.info(f"\n[{i} -- {k} ]")
        log.info(f"\t\trun_id: {r.info.run_id}")
        log.info(f"\t\tmetrics (MSE  - mean, std, min, max): {eval_utl.get_metric(run=r, prefix='mse')}")
        log.info(f"\t\tmetrics (MAE  - mean, std, min, max): {eval_utl.get_metric(run=r, prefix='mae')}")
        log.info(f"\t\tmetrics (SSIM - mean, std, min, max): {eval_utl.get_metric(run=r, prefix='ssim')}")

def print_lib_versions(verbose=False):
    if verbose:
        print("Python version")
        print (sys.version)
        print("Version info.")
        print (sys.version_info)
        print("")
        print("Matplotlib version")
        print('{}'.format(matplotlib.__version__))
        print("Pandas version")
        print('{}'.format(pd.__version__))
        print("")


#########
# Create mlflow logging instance and set global vars' run name and run id.
def init_mlflow_and_set_execution_ids(gvars):
    mlflow_ = MLFlow(
            uri=gvars.get_experiment_tracking_uri(), 
            experiment_name=gvars.get_experiment_tracking_name(),
            verbose=True
        )

    mlflow_.info()

    run_id, run_name = mlflow_.init()
    assert run_id is None and run_name is None, f"Internal Error! run_id and run_name must be None!"

    if gvars.is_verbose():
        log.info(f"Adding parameters to earlier GS executions!")

    experiment_id = MLFlowUtils.get_experiment_id_by_experiment_name(gvars.get_experiment_tracking_name())
    gvars.set_experiment_tracking_id(experiment_id)

    return mlflow_

def Level1Level2APITest(matrix=None, i="unet.cks_3x3.cds_1x1.tcks_2x2", j=79):
    log.printMessage(f"Level1Level2APITest")
    arch_name = i
    seed = j
    Slice = matrix.get_index_slice()
    idx = Slice[arch_name, seed]
    df_heatmap_1 = DfHeatmapBuilderByDict(matrix=matrix,idx=idx,metric='mse').build()
    df_heatmap_2 = DfHeatmapBuilder(matrix=matrix,idx=idx,metric='mse').build()

    # They should have the same numeric content (up to column ordering / dtype)
    # Compare on the intersection of columns to be safe:
    common_cols = df_heatmap_1.columns.intersection(df_heatmap_2.columns)
    diff = (df_heatmap_1[common_cols] - df_heatmap_2[common_cols]).abs().max().max()
    print("max abs diff:", diff)


def SimpleTest_EvaluationMatrix(matrix=None, i="unet.cks_3x3.cds_1x1.tcks_2x2", j=79):
    log.printMessage(f"SimpleTest_EvaluationMatrix")
    arch_name = i
    seed = j

    Slice = matrix.get_index_slice()
    #idx = Slice[arch_name:'unet.cks_3x3.cds_1x1.tcks_4x4' , seed:seed+200]
    idx = Slice[arch_name, seed]
    
    runs = matrix.get_runs(idx=idx)
    log.info(f"Nmb of runs: {len(runs)}")

    n_train_dataset_size = matrix.get_train_dataset_size()
    n_test_dataset_size  = matrix.get_test_dataset_size()
    log.info(f"n_train_dataset_size = {n_train_dataset_size}")
    log.info(f"n_test_dataset_size = {n_test_dataset_size}")

    ################################################################
    run_names = matrix.get_attr(idx=idx,attr_name="run_name")

    log.info(f"Accessing via index ([arch_name]):")
    log.info(f"run_names[i]): {run_names[i]}")
    for ix, name in enumerate(run_names[i]):
        log.info(f"[ {ix} ] run_name: {name}")

    log.info(f"Accessing via loc[arch_name]:")
    log.info(f"run_namesloc[i]): {run_names.loc[i]}")
    for ix, v in enumerate(run_names.loc[i]):
        log.info(f"[ {ix} ] value: {v}")

    log.info(f"Accessing the values directly:")
    for ix,r in enumerate(run_names):
        log.info(f"[ {ix} ]run_name: {r}")
    ################################################################

    ssim = matrix.get_metric(idx=idx,metric="ssim")
    for ix, v in enumerate(ssim):
        log.info(f"[ {ix} ] ssim: {v}")

    all_ssim_values = matrix.get_metric_values(idx=idx, metric="ssim")
    for ix, (k,v) in enumerate(all_ssim_values.items()):
        log.info(f"\n[ {ix} ] all_ssim_values: key: {k}, value: {v}")
    log.info(f"len all_ssim_values[0]={len(next(iter(all_ssim_values.values())))}")
    n_values = matrix.get_total_number_of_metrics_values( metric='ssim')
    assert n_values == n_test_dataset_size, f"[ INTERNAL ERROR ] Vector of metrics' size ({n_values} differs from size of test set ({n_test_dataset_size})!"

    idx2 = slice(0,10,1)
    ssim_values_2 = matrix.get_metric_values(idx=idx,idx2=None, metric="ssim")
    for ix, v in enumerate(ssim_values_2.items()):
        log.info(f"[ {ix} ] ssim: {v}")
    log.info(f"len ssim_values[0]={len(next(iter(ssim_values_2.values())))}")

    new_idx = matrix.get_index_slice()
    examples = matrix.get_examples(idx=new_idx[:,:],idx2=None)
    n_test_set = matrix.get_test_dataset_size()
    for k,v in examples.items():
        assert n_test_set == len(v), f"[ INTERNAL ERROR ] n_test_set != len(v) -- {n_test_set} != {len(v)}"
        log.info(f"[ {k} ] examples: {v}")
    


if __name__ == "__main__":
    args = AnalyseEvaluationsCLI().parse_args()
    gvars = AnalyseEvaluationsGlobalVars(
        verbose_level=args.verbose_level, # means no verbose
        mlflow_uri=args.experiment_tracking_uri,
        mlflow_experiment=args.experiment_tracking_name,
        mlflow_experiment_id = None, # to be set after mlflow initialization
        mlflow_experiment_prefix = args.experiment_prefix, 
        data_resolution=args.dataset_resolution,
        with_filter=args.filter,
    )

    print_lib_versions(verbose=gvars.is_verbose())
    pd.options.mode.copy_on_write = True
    #log.info(f"gvars = {gvars}")
    #log.info(f"args = {args}")

    log.printMessage(f"Reporting evaluation's results!")

#########
# Init environment for mlflow
mlflow_ = init_mlflow_and_set_execution_ids(gvars)

log.info(f"Experiment id: {mlflow_.get_experiment_id()}!")
log.info(f"[ DUMMY ] Experiment name: {mlflow_.get_experiment_name()}!")
log.info(f"[ ACTUAL ] Experiment prefix: {gvars.get_experiment_prefix()}!")

#########
# List all experiments with suffix in mlflow
experiments = mlflow_.get_experiments_by_prefix(prefix=gvars.get_experiment_prefix())
experiments_names = [
    e.name for e in experiments 
]
experiments_ids = [
    e.experiment_id for e in experiments 
]
experiments = None

log.printMessage(f"EXPERIMENTS")
for n,i in zip(experiments_names, experiments_ids):
    log.info(f"{n,i}")

assert len(experiments_ids) == 10, log.err(f"Unexpected number of experiments! value: {len(experiments_ids)}, expected: 10")

# List all executions in mlflow experiments
refit_all_runs = mlflow_.get_all_refit_runs(filter_substring=f"%\_UNET%3x3%{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)
refit_all_runs = refit_all_runs | mlflow_.get_all_refit_runs(filter_substring=f"%\_UNET%5x5%{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)
refit_all_runs = refit_all_runs | mlflow_.get_all_refit_runs(filter_substring=f"%\_UNET%7x7%{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)
refit_all_runs = refit_all_runs | mlflow_.get_all_refit_runs(filter_substring=f"%\_UNET%9x9%{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)
refit_all_runs = refit_all_runs | mlflow_.get_all_refit_runs(filter_substring=f"%\_UNET%13x13%{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)

#refit_all_runs = mlflow_.get_all_refit_runs(filter_substring=f"%\_TRANSUNET%{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)

refit_all_runs = refit_all_runs | mlflow_.get_all_refit_runs(filter_substring=f"%\_TRANSUNET%3x3%{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)
refit_all_runs = refit_all_runs | mlflow_.get_all_refit_runs(filter_substring=f"%\_TRANSUNET%5x5%{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)
refit_all_runs = refit_all_runs | mlflow_.get_all_refit_runs(filter_substring=f"%\_TRANSUNET%7x7%{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)
refit_all_runs = refit_all_runs | mlflow_.get_all_refit_runs(filter_substring=f"%\_TRANSUNET%9x9%{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)
refit_all_runs = refit_all_runs | mlflow_.get_all_refit_runs(filter_substring=f"%\_TRANSUNET%13x13%{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)

#refit_all_runs = mlflow_.get_all_refit_runs(filter_substring=f"{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)
#print_run_info(runs=refit_all_runs, verbose=True)

#unet_runs = mlflow_.get_all_runs(filter_substring='_UNET')
#transunet_runs = mlflow_.get_all_runs(filter_substring='_TRANSUNET')
#all_runs = mlflow_.get_all_runs(filter_substring='3x3%%shallow')
#all_runs = mlflow_.get_all_runs(filter_substring='3x3%%deep')
#print_all_runs(all_runs)
assert eval_utl.check_resolution(runs=refit_all_runs, res=gvars.get_data_resolution(), verbose=gvars.is_verbose()), f"Dataset resolution mismatch! res = {gvars.get_data_resolution()}"


#"""
#Estrutura do Dicionário de Dados do Pipeline de Experimentos.
#
#Este dicionário centraliza e organiza os DataFrames do Pandas contendo os runs 
#do MLflow. Os dados são estruturados de forma aninhada por profundidade de 
#arquitetura ('shallow' e 'deep') e, em seguida, por tipo de otimizador usado 
#('adam' e 'adamw').
#
#A árvore de dados passa por etapas automáticas de limpeza (remoção de modelos 
#transunet com dilation > 1, drop de duplicadas de seeds e reset_index) antes 
#de gerar e injetar as tabelas auxiliares de métricas nos nós correspondentes.
#
#Esquema da Estrutura do Dicionário:
#-----------------------------------
#data = {
#    "shallow": {
#        "self": pd.DataFrame,        # Dados shallow limpos (Todos os otimizadores juntos)
#        "best_model": pd.DataFrame,  # Melhor configuração por modelo para a ramificação shallow
#        "best_kernel": pd.DataFrame, # Melhor configuração por kernel para a ramificação shallow
#        "adam": {
#            "self": pd.DataFrame,        # Apenas runs shallow utilizando o otimizador Adam
#            "best_model": pd.DataFrame,  # Melhor configuração por modelo para shallow + Adam
#            "best_kernel": pd.DataFrame  # Melhor configuração por kernel para shallow + Adam
#        },
#        "adamw": {
#            "self": pd.DataFrame,        # Apenas runs shallow utilizando o otimizador AdamW
#            "best_model": pd.DataFrame,  # Melhor configuração por modelo para shallow + AdamW
#            "best_kernel": pd.DataFrame  # Melhor configuração por kernel para shallow + AdamW
#        }
#    },
#    "deep": {
#        "self": pd.DataFrame,        # Dados deep limpos (Todos os otimizadores juntos)
#        "best_model": pd.DataFrame,  # Melhor configuração por modelo para a ramificação deep
#        "best_kernel": pd.DataFrame, # Melhor configuração por kernel para a ramificação deep
#        "adam": {
#            "self": pd.DataFrame,        # Apenas runs deep utilizando o otimizador Adam
#            "best_model": pd.DataFrame,  # Melhor configuração por modelo para deep + Adam
#            "best_kernel": pd.DataFrame  # Melhor configuração por kernel para deep + Adam
#        },
#        "adamw": {
#            "self": pd.DataFrame,        # Apenas runs deep utilizando o otimizador AdamW
#            "best_model": pd.DataFrame,  # Melhor configuração por modelo para deep + AdamW
#            "best_kernel": pd.DataFrame  # Melhor configuração por kernel para deep + AdamW
#        }
#    }
#}
#
#Exemplos de Acesso:
#-------------------
#>>> # Acessar o DataFrame limpo de execuções do Deep com AdamW:
#>>> df_deep_adamw = data["deep"]["adamw"]["self"]
#
#>>> # Obter o melhor modelo geral para a arquitetura Shallow (Adam + AdamW combinados):
#>>> df_best_shallow = data["shallow"]["best_model"]
#"""



df_shallow, df_deep = create_pd_from_runs(mlflow_=mlflow_, runs=refit_all_runs)
df_shallow_adam, df_shallow_adamw = split_per_optimizer(df=df_shallow)
df_deep_adam, df_deep_adamw = split_per_optimizer(df=df_deep)

# Sua estrutura desejada pronta para o pipeline
df_data = { 
    "shallow": {
        "self": df_shallow, 
        "adam": {"self": df_shallow_adam}, 
        "adamw": {"self": df_shallow_adamw}
    },
    "deep": {
        "self": df_deep, 
        "adam": {"self": df_deep_adam}, 
        "adamw": {"self": df_deep_adamw}
    }
}


# Lista de filtros que você quer aplicar em TODOS os DataFrames da árvore
filtros_ativos = [
    #filter_remove_transunet_with_dilation,
    #filter_remove_conv_kernel_size_9,
    # filter_only_dilation_one  # <--- Descomente aqui se quiser aplicar o filtro geral de dilatação antiga
]

is_debug = True
keep='first'
# Passo 1: Filtra, limpa duplicadas e reseta índices em um único comando
process_dataframes_recursive(df_data, keep=keep, is_debug=is_debug, filters=filtros_ativos)

# Passo 2: Injeta as sub-tabelas (best_model, best_kernel) já baseadas nos dados filtrados
inject_filtered_subsets_recursive(df_data)





## split between adam and adamw
#df_shallow_adam, df_shallow_adamw = split_per_optimizer(df=df_shallow)
#df_deep_adam, df_deep_adamw = split_per_optimizer(df=df_deep)
#
## The lines above remove dilation. It is necessary only when trying to replicate the original Deep Tomography
#
## The lines above remove dilation. It is necessary only when trying to replicate the original Deep Tomography
##df_shallow = df_shallow[df_shallow['conv_dilation_size'] == 1]
##df_deep = df_deep[df_deep['conv_dilation_size'] == 1]
#
#
#
## CRITICAL: Because there are more than one execution per seed, just one can be used (or the mean, but the mean 
## of some attritutes is undefined).
## 1. Drop duplicate seeds, keeping only the final execution for each pair
##df_shallow = df_shallow.drop_duplicates( subset=['arch_name', 'seed'], keep='last').copy()  # .copy() prevents SettingWithCopy warnings later
#df_shallow = drop_duplicates(df=df_shallow, keep=keep, debug=is_debug)
#df_deep = drop_duplicates(df=df_deep, keep=keep, debug=is_debug)
#
#df_shallow_adam = drop_duplicates(df=df_shallow_adam, keep=keep, debug=is_debug)
#df_shallow_adamw = drop_duplicates(df=df_shallow_adamw, debug=is_debug)
##
#df_deep_adam = drop_duplicates(df=df_deep_adam, keep=keep, debug=is_debug)
#df_deep_adamw = drop_duplicates(df=df_deep_adamw, keep=keep, debug=is_debug)
#
################ REMOVE
##df_shallow = df_shallow[df_shallow['seed'].isin([7,79,181,307,433,571,701,853,997,1129])]#,79,181,307])]
#
## 2. Reset the index if needed (just in case the index got scrambled)
#df_shallow_adam = df_shallow_adam.reset_index(drop=True)
#df_shallow_adamw = df_shallow_adamw.reset_index(drop=True)
#df_deep_adam = df_deep_adam.reset_index(drop=True)
#df_deep_adamw = df_deep_adamw.reset_index(drop=True)
#
##print(f"best: {df_deep_adam[df_deep_adam['arch_name'] == 'transunet.cks_5x5.cds_1x1.tcks_6x6.opt_Adam']['mse']}")
##os._exit()
#
##Adam and AdamW together
#df_shallow_best_per_model_all = filter_df_best_per_akr_model(df=df_shallow)
#df_deep_best_per_model_all = filter_df_best_per_akr_model(df=df_deep)
#
#df_shallow_best_per_kernel_all = filter_df_best_per_kernel(df=df_shallow)
#df_deep_best_per_kernel_all = filter_df_best_per_kernel(df=df_deep)
#
## Auxiliary dataframes with subsets
#df_shallow_best_per_model_adam = filter_df_best_per_akr_model(df=df_shallow_adam)
#df_shallow_best_per_model_adamw = filter_df_best_per_akr_model(df=df_shallow_adamw)
#
#df_shallow_best_per_kernel_adam = filter_df_best_per_kernel(df=df_shallow_adam)
#df_shallow_best_per_kernel_adamw = filter_df_best_per_kernel(df=df_shallow_adamw)
#
#df_deep_best_per_model_adam = filter_df_best_per_akr_model(df=df_deep_adam)
#df_deep_best_per_model_adamw = filter_df_best_per_akr_model(df=df_deep_adamw)
#
#df_deep_best_per_kernel_adam = filter_df_best_per_kernel(df=df_deep_adam)
#df_deep_best_per_kernel_adamw = filter_df_best_per_kernel(df=df_deep_adamw)





##### OBS.: The code below didn't work because is not possible to take the mean of tuples and strings. 
# 1. Aggregate the repeated trials (pseudoreplicates) by taking their mean.
# This ensures exactly 1 row per (arch_name, seed) pair.
#df_shallow = df_shallow.groupby(['arch_name', 'seed']).mean().reset_index()
#df_deep = df_deep.groupby(['arch_name', 'seed']).mean().reset_index()


################################### TEST ####################################
#metric_vector_fetcher = eval_utl.make_metric_vector_fetcher(get_metric_values_from_run)
#shallow_matrix = EvaluationMatrix(df=df_shallow, mlflow_client=mlflow_, metric_vector_fetcher=metric_vector_fetcher)
#SimpleTest_EvaluationMatrix(shallow_matrix)
#Level1Level2APITest(shallow_matrix)
#os._exit(0)
################################### TEST ####################################


log.line()
reports = {
    #SHALLOW - All
#    'shallow' : { 'df' : df_data["shallow"]["self"] }, 
#    'shallow.best_per_model' : { 'df' : df_data["shallow"]["best_model"] }, 
#    'shallow.best_per_kernel' : { 'df' : df_data["shallow"]["best_kernel"] }, 
    #Adam
#    'shallow.adam' : { 'df' : df_data["shallow"]["adam"]["self"] }, 
    'shallow.adam.best_per_model'    : { 'df' : df_data["shallow"]["adam"]["best_model"] },
    'shallow.adam.best_per_kernel'    : { 'df' : df_data["shallow"]["adam"]["best_kernel"] }, 
    
    # AdamW
#    'shallow.adamw' : { 'df' : df_data["shallow"]["adamw"]["self"] }, 
#    'shallow.adamw.best_per_model'    : { 'df' : df_data["shallow"]["adamw"]["best_model"] },
#    'shallow.adamw.best_per_kernel'    : { 'df' : df_data["shallow"]["adamw"]["best_kernel"] }, 
    #DEEP - All
#    'deep' : { 'df' : df_data["deep"]["self"] }, 
#    'deep.best_per_model' : { 'df' : df_data["deep"]["best_model"] }, 
#    'deep.best_per_kernel' : { 'df' : df_data["deep"]["best_kernel"] }, 
    #Adam
#    'deep.adam' : { 'df' : df_data["deep"]["adam"]["self"] }, 
    'deep.adam.best_per_model'    : { 'df' : df_data["deep"]["adam"]["best_model"] }, 
    'deep.adam.best_per_kernel'    : { 'df' : df_data["deep"]["adam"]["best_kernel"] }, 
    # AdamW
#    'deep.adamw' : { 'df' : df_data["deep"]["adamw"]["self"] }, 
#    'deep.adamw.best_per_model'    : { 'df' : df_data["deep"]["adamw"]["best_model"] }, 
#    'deep.adamw.best_per_kernel'    : { 'df' : df_data["deep"]["adamw"]["best_kernel"] }, 
}

for k,v in reports.items():
    log.info(f"Making {k} report!")

    df = v['df']

    data_type = df["data_type"].unique()
    #assert len(data_type) == 1, f"[ INTERNAL ERROR ] Must be only one data type in df at this point."
    data_type = data_type[0]

    report_filename = f"{k}--datatype_{data_type}--resolution_{gvars.get_data_resolution()}"
    label = f"All Architectures and  {data_type.upper()} Dataset"
    print (report_filename)
    print (label)

    doc = make_evaluation_report(
        df=df, 
        mlflow_=mlflow_,
        i_name='arch_name',
        j_name='seed',
        title=f"Deep Tomography 2.0 Evaluation Report: {label}", 
        author=f"Antonio Azambuja",
        dataset=data_type, 
        alpha=0.05,
        label=label, 
        verbose=True
    )
    doc.generate_pdf(report_filename, clean_tex=False)
    print(report_filename)


