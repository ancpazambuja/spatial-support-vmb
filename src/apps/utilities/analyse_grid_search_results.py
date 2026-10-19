
"""@package analyse_grid_search_results.py
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
import pandas as pd


############################
# Latex
from pylatex import Document, Section, Subsection, Figure, Command
from pylatex.utils import NoEscape

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
from src.visualization.gs_viewer import GSViewer as gsviewer

from src.preprocessing.pipeline import Preprocessing

from src.apps.command_line_interface import AddParametersInGSCLI
from src.apps.global_vars import AddParametersInGSGlobalVars

from src.pdf.sections_builder import SectionsBuilder
from src.pdf.gs_sections_builder import GSListBuilder, GSPDFBuilder
#from src.pdf.section import Section as mySection

#from src.preprocessing.local_pipeline import PreprocessingPipe
#from src.processing.train_data import TrainDataInfo, TrainData, Jobs, ProductionOutlierClassification

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

def make_doc(title=None, list_of_sections=None):

    # Create a new document
    doc = Document()
    
    # Add title and author
    doc.preamble.append(Command('title', title))
    doc.preamble.append(Command('author', 'Antonio Azambuja'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))
    
    add_sections_to_pdf(doc=doc, list_of_sections=list_of_sections)

    return doc

def make_gs_report_table_from_df(df=None, title='', author='', name='', filter=None,
                                 arch=None, dataset=None, verbose=False):
    if filter is not None:
        log.info(f"Filtering...")
        df = filter.filter(df)

    list_of_sections = GSListBuilder.build_from_df(df=df, arch=arch, dataset=dataset, label=name, verbose=verbose)
    sections = SectionsBuilder.build_from_list(l=list_of_sections)
    return GSPDFBuilder.build_from_sections(title=title, author=author, sections=sections, verbose=verbose)

def add_sections_to_pdf(doc=None, list_of_sections=None):
    log.info(f"INSIDE ADD_SECTIONS_TO_PDF")
    log.info(f"list_of_sections = {list_of_sections}")
    for s in list_of_sections:
        with doc.create(Section(s.title)) as sec:
            log.info(f"s['text'] = {s.text} ")
            sec.append(s.text)
            if s.figures is not None:
                figures = s.figures
                for f in figures:
                    with sec.create(Figure(position='h!')) as plot:
                        plot.add_image(f['filename'], width=NoEscape(r'0.8\textwidth'))
                        plot.add_caption(f['caption'])
            if s.subsections is not None:
                add_sections_to_pdf(doc=sec, list_of_sections=s.subsections)
                
        #log.info(f"type of s = {type(s)}")
        #log.info(f"s = {s}")
        #log.info(f"title = {s['title']}  --  text = {s['text']}")

##  UTILITIES
def get_gs_metrics(metrics=None):
    return { 
        'mean_valid_score': metrics['mean_valid_score'],
        'std_valid_score': metrics['std_valid_score']
    }

def get_gs_params(params=None):
    return { 
        'minibatch_size': params['batch_size'],
        'max_epochs': params['max_epochs'],
        'lr': params['lr'],
        'weight_decay': params['optimizer__weight_decay'],
        'n_filters_init': params['module__n_filters_init'],
        'conv_kernel_size': params['_x_conv_kernel_size'],
        'att_in_channels': params['module__attention_in_channels'],
        'use_transformers': params['module__use_transformers'],
        'att_hidden_size': params["_x_att_hidden_size"],
        'att_num_heads': params['_x_att_num_heads'],
        'att_num_layers': params['_x_att_num_layers'],
        'att_att_dropout_rate': params['_x_att_att_dropout_rate'],
        'att_dropout_rate': params['_x_att_dropout_rate'],
        'att_mlp_dim': params['_x_att_mlp_dim'],
    }


def build_gs_info(run_id=None, run_name=None, parent_run_id=None, parent_run_name=None):
    data_type = utl.get_dataset_type_from_name(parent_run_name)
    assert data_type is not None and (data_type == 'deep' or data_type == 'shallow'), f"[ ERROR ] Unknown dataset type!"
    return {
        'run_id' : run_id,
        'run_name' : run_name,
        'parent_run_id' : parent_run_id,
        'parent_run_name' : parent_run_name,
        'data_type' : data_type,
    }
 
def create_pd_from_gs_runs(gvars=None, gs_runs=None):
    # run_name, run_id, parent_run_id, parent_run_name,
    # minibatch_size, max_epochs, lr, weight_decay, 
    # seed, dataset_type, n_filters_init, kernel_size, att_in_channels,
    # use_transformers, att_img_size, num_layers, num_heads, mlp_dim, hidden_size
    log.info(f"Creating pandas dataframe from {len(gs_runs)} runs...")
    list_of_dicts = list()
    for i, (k,v) in enumerate(gs_runs.items()):
        log.printMessage(f"[{i}] - [{k}]")
        children = v["children"]
        if children is not None:
            for j, (kk, vv) in enumerate(children.items()):
                log.info(f"[{j}] {kk}") #, {vv}")
                # get att_config structure
                run = vv
                run_dict = run.data.to_dictionary()
                assert run_dict['params'] is not None, f"Error! There is no 'params' attribute associated to this run!"
                assert len(run_dict['params'].keys()) > 0, f"Error! run_dict['params'] has zero attributes!"
                params_dict = get_gs_params(run_dict['params']) 
                metrics_dict = get_gs_metrics(run_dict['metrics']) 
                info_dict = build_gs_info(
                    run_id=run.info.run_id, run_name=run.info.run_name,
                    parent_run_id=run.data.tags['mlflow.parentRunId'], parent_run_name=run.data.tags['run_name'],
                )
                #print(f"params = {params_dict|metrics_dict|info_dict}")
                list_of_dicts.append(params_dict|metrics_dict|info_dict)
        else:
            log.warn("DOESN'T HAVE CHILDREN")
    log.ok()


    ret = pd.DataFrame(list_of_dicts)
    column_dtypes = {
        'lr' : 'float',
        'weight_decay' : 'float',
        'minibatch_size' : 'int',
        'max_epochs' : 'int',
        'n_filters_init' : 'int',
        'conv_kernel_size' :'int',
        'att_in_channels' : 'int',
        'use_transformers' :'bool',
        'att_hidden_size' : 'int',
        'att_num_heads' : 'int',
        'att_num_layers' : 'int',
        'att_att_dropout_rate' : 'float',
        'att_dropout_rate' :'float',
        'att_mlp_dim' : 'int',
        'run_id' : 'string',
        'run_name' : 'string',
        'parent_run_id' : 'string',
        'parent_run_name' : 'string',
        'data_type' : 'string',
    }
    ret = ret.astype(column_dtypes)

    for k,v in ret.items():
        print(k,v)

    return ret[ ret['data_type'] == 'shallow'], ret[ ret['data_type'] == 'deep'] 

def check_resolution(gs_runs=None, res=None, verbose=True):
    check = False if ((gs_runs is None) or (len(gs_runs) == 0)) else True
    error_list = list()
    for i, (k,v) in enumerate(gs_runs.items()):
        if k.find(res) < 0:
            error_list.append(k)
    if len(error_list) > 0:
        check = False
        if verbose > 0:
            log.warn(f"List of gs_runs violating the target dataset resolution ({res}): {error_list}")
    return check
    

def print_all_gs_runs(gs_runs):
    print(f"NUMBER OF GS RUNS:  {len(gs_runs)}")
    for i, (k,v) in enumerate(gs_runs.items()):
        log.printMessage(f"[{i}] - [{k}]")
        children = v["children"]
        if children is not None:
            for j, (kk, vv) in enumerate(children.items()):
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
        data_resolution=args.dataset_resolution,
        with_filter=args.filter,
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
#print_run_info(runs)
gs_all_runs = mlflow_.get_all_gs_runs(filter_substring=f"{gvars.get_data_resolution()}-res")
#gs_unet_runs = mlflow_.get_all_gs_runs(filter_substring='_UNET')
#jgs_transunet_runs = mlflow_.get_all_gs_runs(filter_substring='_TRANSUNET')
#gs_all_runs = mlflow_.get_all_gs_runs(filter_substring='3x3%%shallow')
#gs_all_runs = mlflow_.get_all_gs_runs(filter_substring='3x3%%deep')
#print_all_gs_runs(gs_all_runs)
assert check_resolution(gs_runs=gs_all_runs, res=gvars.get_data_resolution(), verbose=gvars.is_verbose()), f"Dataset resolution mismatch! res = {gvars.get_data_resolution()}"

df_shallow, df_deep = create_pd_from_gs_runs(gvars=gvars, gs_runs=gs_all_runs)
log.info(f"df_shallow = {df_shallow['parent_run_name'].unique()}")
log.info(f"#df_shallow = {df_shallow.shape[0]}")

# outlier removal
df_shallow=df_shallow[df_shallow['mean_valid_score'] < 10]
df_deep = df_deep[df_deep['mean_valid_score'] < 10]

#highres executions
reports = {
    'report1_1' : { 'df' : df_shallow, 'arch' : "Unet",      'dataset' : "Shallow", 'filter' : ShallowUnetFilter()}, 
    'report1_2' : { 'df' : df_deep,    'arch' : "Unet",      'dataset' : "Deep", 'filter' : DeepUnetFilter()}, 
    'report2_1' : { 'df' : df_shallow, 'arch' : "Transunet", 'dataset' : "Shallow", 'filter' : ShallowTransunetFilter()}, 
    'report2_2' : { 'df' : df_deep,    'arch' : "Transunet", 'dataset' : "Deep", 'filter' : DeepTransunetFilter()}, 
}

#lowres executions
#reports = {
#    'report1_1' : { 'df' : df_shallow, 'arch' : "Unet",      'dataset' : "Shallow", 'filter' : ShallowUnetFilter()}, 
#    'report2_1' : { 'df' : df_shallow, 'arch' : "Transunet", 'dataset' : "Shallow", 'filter' : ShallowTransunetFilter()}, 
#}

for k,v in reports.items():
    log.info(f"Making report_{v['arch']}_{v['dataset']} ")

    df = v['df']

    report_filename = f"{v['arch']}_{v['dataset']}_{gvars.get_data_resolution()}_report-filter_{gvars.with_filter()}"
    label = f"{v['arch']} Architecture and {v['dataset']} Dataset"

    # unet and transunet split 
    df_filtered=df[df['parent_run_name'].str.contains("_"+v['arch'], case=False)]

    doc = make_gs_report_table_from_df(
        df=df_filtered, 
        title=f"Deep Tomography 2.0 Report: {label}", 
        author=f"Antonio Azambuja",
        arch=v['arch'], 
        dataset=v['dataset'], 
        name=label, 
        filter=v['filter'] if gvars.with_filter() else None,
        verbose=True
    )
    doc.generate_pdf(report_filename, clean_tex=False)



