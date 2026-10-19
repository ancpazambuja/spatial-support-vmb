
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
     - kernel size (3x3, 5x5, 7x7, 13x13)
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

def get_filtered_df_best_per_kernel(df=None):
    # 1. Identifica o nome da melhor arquitetura para cada conv_kernel_size
    best_archs = (
        df.groupby(['arch_name', 'conv_kernel_size', 'use_transformers'])['mse']
        .mean()
        .reset_index()
        .sort_values('mse')
        .drop_duplicates(['conv_kernel_size','use_transformers'])['arch_name']
    )
    print("-------------------------------")
    print(best_archs)
    print("-------------------------------")
    # 2. Filtra o DataFrame original mantendo apenas as linhas dessas arquiteturas
    return  df[df['arch_name'].isin(best_archs)]

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
    ret['arch_name'] = column_arch_name

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
refit_all_runs = refit_all_runs | mlflow_.get_all_refit_runs(filter_substring=f"%\_UNET%13x13%{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)

#refit_all_runs = mlflow_.get_all_refit_runs(filter_substring=f"%\_TRANSUNET%{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)

refit_all_runs = refit_all_runs | mlflow_.get_all_refit_runs(filter_substring=f"%\_TRANSUNET%3x3%{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)
refit_all_runs = refit_all_runs | mlflow_.get_all_refit_runs(filter_substring=f"%\_TRANSUNET%5x5%{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)
refit_all_runs = refit_all_runs | mlflow_.get_all_refit_runs(filter_substring=f"%\_TRANSUNET%7x7%{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)
refit_all_runs = refit_all_runs | mlflow_.get_all_refit_runs(filter_substring=f"%\_TRANSUNET%13x13%{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)

#refit_all_runs = mlflow_.get_all_refit_runs(filter_substring=f"{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)
#print_run_info(runs=refit_all_runs, verbose=True)

#unet_runs = mlflow_.get_all_runs(filter_substring='_UNET')
#transunet_runs = mlflow_.get_all_runs(filter_substring='_TRANSUNET')
#all_runs = mlflow_.get_all_runs(filter_substring='3x3%%shallow')
#all_runs = mlflow_.get_all_runs(filter_substring='3x3%%deep')
#print_all_runs(all_runs)
assert eval_utl.check_resolution(runs=refit_all_runs, res=gvars.get_data_resolution(), verbose=gvars.is_verbose()), f"Dataset resolution mismatch! res = {gvars.get_data_resolution()}"

df_shallow, df_deep = create_pd_from_runs(mlflow_=mlflow_, runs=refit_all_runs)

# The lines above remove dilation. It is necessary only when trying to replicate the original Deep Tomography
df_shallow = df_shallow[df_shallow['conv_dilation_size'] == 1]
df_deep = df_deep[df_deep['conv_dilation_size'] == 1]

# CRITICAL: Because there are more than one execution per seed, just one can be used (or the mean, but the mean 
# of some attritutes is undefined).
# 1. Drop duplicate seeds, keeping only the final execution for each pair
df_shallow = df_shallow.drop_duplicates( subset=['arch_name', 'seed'], keep='last').copy()  # .copy() prevents SettingWithCopy warnings later

############### REMOVE
#df_shallow = df_shallow[df_shallow['seed'].isin([7,79,181,307,433,571,701,853,997,1129])]#,79,181,307])]


df_deep = df_deep.drop_duplicates( subset=['arch_name', 'seed'], keep='last').copy()  # .copy() prevents SettingWithCopy warnings later
# 2. Reset the index if needed (just in case the index got scrambled)
df_shallow = df_shallow.reset_index(drop=True)
df_deep = df_deep.reset_index(drop=True)

# Auxiliary dataframes with subsets
df_shallow_best_per_kernel = get_filtered_df_best_per_kernel(df=df_shallow)
df_deep_best_per_kernel = get_filtered_df_best_per_kernel(df=df_deep)

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


reports = {
#    'Shallow_Dataset_All' : { 'df' : df_shallow }, 
    'Shallow_Dataset_Best_For_Each_Kernels'    : { 'df' : df_shallow_best_per_kernel }, 
#    'Deep_Dataset_All'    : { 'df' : df_deep }, 
    'Deep_Dataset_Best_For_Each_Kernels'    : { 'df' : df_deep_best_per_kernel }, 
}

for k,v in reports.items():
    log.info(f"Making {k} report!")

    df = v['df']

    data_type = df["data_type"].unique()
    #assert len(data_type) == 1, f"[ INTERNAL ERROR ] Must be only one data type in df at this point."
    data_type = data_type[0]

    report_filename = f"{k.upper()}--FINAL_EVALUATION_REPORT--DATATYPE_{data_type}--RESOLUTION_{gvars.get_data_resolution()}"
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


