
"""@package add_train_and_test_sets_indexes_in_refit.py
Add a list of indexes of the train and test dataset in the given MlFlow's run id.
This is needed to evaluate the model trained without data leakage. 

**REMEMBER**:

When the model was trained (the final training, not the grid search), we have to know
the training dataset and **never** use a sample from the training dataset to evaluate the
model performance.

When the final training (refit) I forget to write it. So, in this program, I will recover this information
from the x_seed parameter and will write it into two lists:
 - one with the indexes of the training samples;
 - another with the indexes of the test samples.
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

from src.apps.command_line_interface import AddTrainTestIndexesInCLI
from src.apps.global_vars import AddTrainTestIndexesInGlobalVars

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

def get_x_seed_value(params=None):
    ret = int(params['_x_seed'])
    return ret 

def build_train_test_indexes(seed=0,test_perc=0.1):
    # It is important to replicate here the same initialization of seeds 
    # used during training. And, of course, the same split process.
    utl.set_seed(seed)
    train_indexes, test_indexes = utl.two_split(nsamples=1000, seed=seed, test_perc=test_perc )
    return {
         '_x_train_indexes': train_indexes,
         '_x_test_indexes': test_indexes,
         '_x_test_perc': test_perc, 
          }


def add_train_and_test_sets_indexes_to_all_gs_runs(gvars=None, mlflow_=None, gs_runs=None):

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
                x_seed = 0
                x_test_perc = 0.1 # All hyper parameters tuning and refits have used this value.
                params_dict = build_train_test_indexes(seed=x_seed, test_perc=x_test_perc)
                exist, keys = MLFlowUtils.check_if_params_already_exist(mlflow_client=mlflow_client, run=run, params=params_dict)
                if exist:
                    log.info(f"{kk}\nHad already been updated before!")
                    continue
                #assert not exist, f"Some keys already exist! Program will abort! Existed keys = {keys}"

                log.warn(f"No updating was done. In order to update, uncomment the line below and comment this.")
                # line below is commented to avoid run the program and change mlflow database by mistake
                # uncomment if you really want to update the dataset
                #MLFlowUtils.log_parameters_by_run(mlflow_client=mlflow_client, run=run, params=params_dict)
        else:
            log.warn("DOESN'T HAVE CHILDREN")
    log.ok()




def add_train_and_test_sets_indexes_to_all_refit_runs(gvars=None, mlflow_=None, refit_runs=None):

    mlflow_client = mlflow_.create_client()

    log.info(f"Updating {len(refit_runs)} runs...")
    for i, (k,run) in enumerate(refit_runs.items()):
        #log.printMessage(f"[{i}] - [{run}]")
        log.printMessage(f"[{i}] - [{k}]")
        run_dict = run.data.to_dictionary()
        assert run_dict['params'] is not None, f"Error! There is no 'params' attribute associated to this run!"
        assert len(run_dict['params'].keys()) > 0, f"Error! run_dict['params'] has zero attributes!"

        x_seed = get_x_seed_value(run_dict['params']) 
        log.warn(f"x_seed = {x_seed}")
        x_test_perc = 0.1 # All hyper parameters tuning and refits have used this value.
        params_dict = build_train_test_indexes(seed=x_seed, test_perc=x_test_perc)
        #print(params_dict)
        #print(",".join(map(str, params_dict["_x_test_indexes"])))

        exist, keys = MLFlowUtils.check_if_params_already_exist(mlflow_client=mlflow_client, run=run, params=params_dict)
        if exist:
            log.info(f"{k}\nHad already been updated before!")
            continue
        
        log.warn(f"No updating was done. In order to update, uncomment the line below and comment this.")
        # line below is commented to avoid run the program and change mlflow database by mistake
        # uncomment if you really want to update the dataset
        #MLFlowUtils.log_parameters_by_run(mlflow_client=mlflow_client, run=run, params=params_dict)
    log.ok()



def print_all_runs(runs):
    print(f"NUMBER OF RUNS:  {len(runs)}")
    for i, (k,v) in enumerate(runs.items()):
        log.printMessage(f"[{i}] - [{k}, {v}]")

def print_run_info(runs):
    print(f"NUMBER OF RUNS: {len(runs)}")
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
    args = AddTrainTestIndexesInCLI().parse_args()
    gvars = AddTrainTestIndexesInGlobalVars(
        verbose_level=args.verbose_level, # means no verbose
        mlflow_uri=args.experiment_tracking_uri,
        mlflow_experiment=args.experiment_tracking_name,
        mlflow_experiment_id = None, # to be set after mlflow initialization
        type=args.type,
    )

    print_lib_versions(verbose=gvars.is_verbose())
    log.info(f"gvars = {gvars}")
    log.info(f"args = {args}")

    log.printMessage(f"Adding parameters to earlier REFIT executions!")

#########
# Init environment for mlflow
mlflow_ = init_mlflow_and_set_execution_ids(gvars)

log.info(f"Experiment id: {mlflow_.get_experiment_id()}!")
log.info(f"Experiment name: {mlflow_.get_experiment_name()}!")


##################################################################
# List all executions in the mlflow experiment
################
if gvars.get_type() == 'gs':
    #all_runs = mlflow_.get_all_gs_runs(filter_substring="34.03")
    all_runs = mlflow_.get_all_gs_runs()
    add_train_and_test_sets_indexes_to_all_gs_runs(gvars=gvars, mlflow_=mlflow_, gs_runs=all_runs)
    #print_all_runs(all_runs)
elif gvars.get_type() == 'refit':
    #all_runs = mlflow_.get_all_refit_runs(filter_substring="13x13")
    all_runs = mlflow_.get_all_refit_runs()
    add_train_and_test_sets_indexes_to_all_refit_runs(gvars=gvars, mlflow_=mlflow_, refit_runs=all_runs)
    #print_all_runs(all_runs)
else:
    raise ValueError(f"--type must be 'gs' or 'refit'. (Actual: {gvars.get_type()})")

