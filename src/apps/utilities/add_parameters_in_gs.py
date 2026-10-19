
"""@package add_parameters_in_gs.py
This Given a MlFlow's run id, performs the final training with the same
parameters. 
Tipically, the MlFlow's run will be the best gridsearchcv run. And,
this program here will refit based on the same parameters, 
which will be loaded from MlFlow server.
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
import ml_collections

############################
# science stuff
from scipy import ndimage
from scipy import interpolate
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

from src.preprocessing.pipeline import Preprocessing

from src.apps.command_line_interface import AddParametersInGSCLI
from src.apps.global_vars import AddParametersInGSGlobalVars

#from src.preprocessing.local_pipeline import PreprocessingPipe
#from src.processing.train_data import TrainDataInfo, TrainData, Jobs, ProductionOutlierClassification
def get_mlflow_run(experiment_uri=None, experiment_id=None, experiment_name=None, run_id=None, run_name=None):
    mlflow_client, run = MLFlowUtils.create_client_and_get_run(
            uri=experiment_uri,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            execution_id=run_id
        )
    return mlflow_client, run, run.data.to_dictionary()

def get_conv_kernel_size_values(params=None):
    ret = { '_x_conv_kernel_size': eval(params['module__conv_kernel_size'])[0] }
    return ret 

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
 
def update_all_gs_runs(gvars=None, mlflow_=None, gs_runs=None):

    mlflow_client = mlflow_.create_client()

    log.info(f"Updating {len(gs_runs)} runs...")
    for i, (k,v) in enumerate(gs_runs.items()):
        log.printMessage(f"[{i}] - [{k}]")
        children = v["children"]
        if children is not None:
            for j, (kk, vv) in enumerate(children.items()):
                log.info(f"[{j}] {kk}") #, {vv}")
                run = vv
                run_dict = vv.data.to_dictionary()
                assert run_dict['params'] is not None, f"Error! There is no 'params' attribute associated to this run!"
                assert len(run_dict['params'].keys()) > 0, f"Error! run_dict['params'] has zero attributes!"
                # get att_config structure
                params_dict_att = get_attention_config_values(run_dict['params']) 
                params_dict_conv = get_conv_kernel_size_values(run_dict['params']) 
                params_dict = params_dict_att | params_dict_conv
                exist, keys = MLFlowUtils.check_if_params_already_exist(mlflow_client=mlflow_client, run=run, params=params_dict)
                if exist:
                    log.info(f"{kk}\nHad already been updated before!")
                    continue;
                #assert not exist, f"Some keys already exist! Program will abort! Existed keys = {keys}"

                log.warn(f"No updating was done. In order to update, uncomment the line below and comment this.")
                # line below is commented to avoid run the program and change mlflow database by mistake
                # uncomment if you really want to update the dataset
                #MLFlowUtils.log_parameters_by_run(mlflow_client=mlflow_client, run=run, params=params_dict)
        else:
            log.warn("DOESN'T HAVE CHILDREN")
    log.ok()

# The implementation bellow is slow and allocate twice more mlflow run objects. 
# But this implementation was tested more times than the above. So that, I keep it here.
#def update_all_gs_runs(gvars=None, gs_runs=None):
#    log.info(f"Updating {len(gs_runs)} runs...")
#    for i, (k,v) in enumerate(gs_runs.items()):
#        log.printMessage(f"[{i}] - [{k}]")
#        children = v["children"]
#        if children is not None:
#            for j, (kk, vv) in enumerate(children.items()):
#                log.info(f"[{j}] {kk}") #, {vv}")
#                # get att_config structure
#                mlflow_client, run, run_dict = get_mlflow_run(
#                    experiment_uri=gvars.get_experiment_tracking_uri(), 
#                    experiment_id=gvars.get_experiment_tracking_id(), 
#                    experiment_name=gvars.get_experiment_tracking_name(), 
#                    run_id=vv.info.run_id,
#                )
#                assert vv.info.run_name == run.info.run_name, f"run_name parameter and run_name returned by create_and_get_run are not equal ({vv.info.run_name}, {run_data.dict['run_name']})"
#                assert run_dict['params'] is not None, f"Error! There is no 'params' attribute associated to this run!"
#                assert len(run_dict['params'].keys()) > 0, f"Error! run_dict['params'] has zero attributes!"
#                params_dict_att = get_attention_config_values(run_dict['params']) 
#                params_dict_conv = get_conv_kernel_size_values(run_dict['params']) 
#                params_dict = params_dict_att | params_dict_conv
#                exist, keys = MLFlowUtils.check_if_params_already_exist(mlflow_client=mlflow_client, run=run, params=params_dict)
#                assert not exist, f"Some keys already exist! Program will abort! Existed keys = {keys}"
#
#                log.warn(f"No updating was done. In order to update, uncomment the line below and comment this.")
#                # line below is commented to avoid run the program and change mlflow database by mistake
#                # uncomment if you really want to update the dataset
#                #MLFlowUtils.log_parameters_by_run(mlflow_client=mlflow_client, run=run, params=params_dict)
#        else:
#            log.warn("DOESN'T HAVE CHILDREN")
#    log.ok()

def print_all_gs_runs(gs_runs):
    print(f"NUMBER OF GS RUNS:  {len(gs_runs)}")
    for i, (k,v) in enumerate(gs_runs.items()):
        log.printMessage(f"[{i}] - [{k}]")
        children = v["children"]
        if children is not None:
            for j, (kk, vv) in enumerate(children.items()):
                #print(type(vv))
                print(f"[{j}] {kk}, {vv}")
        else:
            print("DOESN'T HAVE CHILDREN")

def print_run_info(runs):
    print(f"NUMBER OR RUNS: {len(runs)}")
    for i, r in enumerate(runs):
        print(f"\n[{i} -- {r.info.run_name} ]")
        print(f"run_id: {r.info.run_id}")
        print(f"lifecycle_stage: {r.info.lifecycle_stage}")
        print(f"metrics: {r.data.metrics}")
        # Exclude mlflow system tags
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        print(f"tags: {tags}")

def print_lib_versions(verbose=False):
    if verbose:
        print("Python version")
        print (sys.version)
        print("Version info.")
        print (sys.version_info)
        print("")
        print("Matplotlib version")
        print('{}'.format(matplotlib.__version__))
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


if __name__ == "__main__":
    args = AddParametersInGSCLI().parse_args()
    gvars = AddParametersInGSGlobalVars(
        verbose_level=args.verbose_level, # means no verbose
        mlflow_uri=args.experiment_tracking_uri,
        mlflow_experiment=args.experiment_tracking_name,
        mlflow_experiment_id = None, # to be set after mlflow initialization
    )

    print_lib_versions(verbose=gvars.is_verbose())
    log.info(f"gvars = {gvars}")
    log.info(f"args = {args}")

    log.printMessage(f"Adding parameters to earlier GS executions!")

#########
# Init environment for mlflow
mlflow_ = init_mlflow_and_set_execution_ids(gvars)

log.info(f"Experiment id: {mlflow_.get_experiment_id()}!")
log.info(f"Experiment name: {mlflow_.get_experiment_name()}!")

#########
# List all executions in the mlflow experiment
#runs = mlflow_.get_all_runs()
#print_run_info(runs)
gs_runs = mlflow_.get_all_gs_runs()
#gs_runs = mlflow_.get_all_gs_runs(filter_substring="34.03")
print_all_gs_runs(gs_runs)
update_all_gs_runs(gvars=gvars, mlflow_=mlflow_, gs_runs=gs_runs)
