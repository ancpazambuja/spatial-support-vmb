
"""@package analyse_evaluation_examples.py
This module shows images of examples (y,\hat{y}) from a list of examples ids and models (run ids). 
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

from src.apps.command_line_interface import AnalyseEvaluationsExamplesCLI
from src.apps.global_vars import AnalyseEvaluationsExamplesGlobalVars

from src.pdf.sections_builder import SectionsBuilder
from src.pdf.eval_sections_builder import EvalListBuilder
from src.pdf.gs_sections_builder import GSPDFBuilder
from src.pdf.eval_utils import EvaluationMatrix, DfHeatmapBuilder, DfHeatmapBuilderByDict
from src.pdf.eval_utils import EvalUtils as eval_utl 
#parse_string, make_metric_vector_fetcher, get_metric_values_from_run
#from src.pdf.section import Section as mySection

#from src.preprocessing.local_pipeline import PreprocessingPipe
#from src.processing.train_data import TrainDataInfo, TrainData, Jobs, ProductionOutlierClassification

def print_run_info(runs):
    log.info(f"len(runs) = {len(runs)}")
    for r in runs:
        log.info(f"run_id = {r.info.run_id}")

def make_evaluation_report( df=None, mlflow_=None, title='', author='', dataset='', label='', verbose=True):
    metric_vector_fetcher = eval_utl.make_metric_vector_fetcher(eval_utl.get_metric_values_from_run)
    matrix = EvaluationMatrix(df=df, mlflow_client=mlflow_, metric_vector_fetcher=metric_vector_fetcher)
    #matrix = EvaluationMatrix(df=df, mlflow_=mlflow_, i='arch_name', j='seed')
    list_of_sections = EvalListBuilder.build(df=df, matrix=matrix, alpha=alpha, dataset=dataset, label=label, verbose=verbose)
    sections = SectionsBuilder.build_from_list(l=list_of_sections)
    return GSPDFBuilder.build_from_sections(title=title, author=author, sections=sections, verbose=verbose)
EvalListBuilder.build

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


if __name__ == "__main__":
    args = AnalyseEvaluationsExamplesCLI().parse_args()
    gvars = AnalyseEvaluationsExamplesGlobalVars(
        verbose_level=args.verbose_level, # means no verbose
        mlflow_uri=args.experiment_tracking_uri,
        mlflow_experiment=args.experiment_tracking_name,
        mlflow_experiment_id = None, # to be set after mlflow initialization
        mlflow_experiment_prefix = args.experiment_prefix, 
        model_path=args.model_path,
        list_of_run_ids=args.list_of_run_ids,
        list_of_hardest_examples=args.list_of_hardest_examples,
        list_of_easiest_examples=args.list_of_easiest_examples,
    )

    pd.options.mode.copy_on_write = True
    log.info(f"gvars = {gvars}")
    log.info(f"args = {args}")

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
runs = mlflow_.get_runs_by_list_of_ids(
    ids_list=gvars.get_list_of_run_ids(), 
    experiment_ids=experiments_ids
)

print_run_info(runs)

#refit_all_runs = mlflow_.get_all_refit_runs(filter_substring=f"{gvars.get_data_resolution()}-res", experiment_ids=experiments_ids)
#print_run_info(runs=refit_all_runs, verbose=True)

#unet_runs = mlflow_.get_all_runs(filter_substring='_UNET')
#transunet_runs = mlflow_.get_all_runs(filter_substring='_TRANSUNET')
#all_runs = mlflow_.get_all_runs(filter_substring='3x3%%shallow')
#all_runs = mlflow_.get_all_runs(filter_substring='3x3%%deep')
#print_all_runs(all_runs)
assert eval_utl.check_resolution(runs=runs, res=gvars.get_data_resolution(), verbose=gvars.is_verbose()), f"Dataset resolution mismatch! res = {gvars.get_data_resolution()}"

df = create_pd_from_runs(mlflow_=mlflow_, runs=runs)
df = df.reset_index(drop=True)

print(df.head())

log.info(f"Making report!")

data_type = df["data_type"].unique()
#assert len(data_type) == 1, f"[ INTERNAL ERROR ] Must be only one data type in df at this point."
data_type = data_type[0]

report_filename = f"{data_type.upper()}--FINAL_EVALUATION_REPORT--RESOLUTION_{gvars.get_data_resolution()}"
label = f"Hardest and Easiest Examples -- {data_type.upper()} Dataset"
print (report_filename)
print (label)

doc = make_evaluation_report(
    df=df, 
    mlflow_=mlflow_,
    title=f"Deep Tomography 2.0 Evaluation Report: HARDEST / EASIEST -- {label}", 
    dataset=data_type,
    label=label, 
    verbose=True
)

doc.generate_pdf(report_filename, clean_tex=False)
print(report_filename)


