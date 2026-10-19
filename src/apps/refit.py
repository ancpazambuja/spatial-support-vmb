
"""@package refit_parallel.py
Given a MlFlow's run id, performs the final training with the same
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

from src.apps.command_line_interface import RefitCLI
from src.apps.global_vars import RefitGlobalVars

#from src.preprocessing.local_pipeline import PreprocessingPipe
#from src.processing.train_data import TrainDataInfo, TrainData, Jobs, ProductionOutlierClassification


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

    _, run_name = utl.get_execution_names(global_vars=gvars)
    gvars.set_execution_name(run_name)
    run_id, returned_run_name = mlflow_.init(run_name=run_name)
    gvars.set_execution_id(run_id)
    assert run_name == returned_run_name, f"Internal error! run_name ({run_name}) differs from returned_run_name ({returned_run_name})"
    if gvars.is_verbose():
        log.info(f"NEW JOB! run id: {run_id}")
        log.info(f"         run name: {run_name}")

    experiment_id = MLFlowUtils.get_experiment_id_by_experiment_name(gvars.get_experiment_tracking_name())
    gvars.set_experiment_tracking_id(experiment_id)

    best_run_experiment_id = MLFlowUtils.get_experiment_id_by_experiment_name(gvars.get_best_run_experiment_tracking_name())
    gvars.set_best_run_experiment_tracking_id(best_run_experiment_id)

    return mlflow_





if __name__ == "__main__":

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # reproducibility 

    args = RefitCLI().parse_args()
    print(args)
    gvars = RefitGlobalVars(
        verbose_level=args.verbose_level, # means no verbose
        mlflow_uri=args.experiment_tracking_uri,
        mlflow_experiment=args.experiment_tracking_name,
        mlflow_experiment_id = None, # to be set after mlflow initialization
        data=args.dataset_type,
        data_resolution=args.dataset_resolution,
        data_max_samples=args.dataset_max_samples,
        unembed_type=args.unembed_type,
        conv_filter_size=(args.conv_filter_size_x, args.conv_filter_size_z),
        conv_dilation_size=(args.conv_dilation_size_x, args.conv_dilation_size_z),
        transposed_conv_filter_size=(args.transposed_conv_filter_size_x, args.transposed_conv_filter_size_z),
        max_epochs=args.max_epochs,
        cv=args.cv,
        model_arch=args.model_arch,
        seed=args.seed,
        execution_id=args.execution_id,
        label_id=args.label_id,
        device=args.device, 
        test_perc=args.test_perc,
        refit=args.refit,
        model_path=args.model_path,
        gs_best_run_id=args.gs_best_run_id,
        gs_best_run_experiment_tracking_name=args.gs_best_run_experiment_tracking_name,
        att_num_layers=args.att_num_layers,
        one_step_training=args.one_step_training
    )

    assert gvars.is_refit(), f"Unexpected error. Refit must be set"
    assert gvars.is_one_step_training() or gvars.is_transunet(), f"Unet model has no two step training!"
    
    print_lib_versions(verbose=gvars.is_verbose())
    log.info(f"gvars = {gvars}")
    log.info(f"args = {args}")

    log.printMessage(f"HOLDOUT: TRAIN INDEXES")
    print(gvars.get_train_indexes())

    log.printMessage(f"HOLDOUT: TEST INDEXES")
    print(gvars.get_test_indexes())

# # Preprocessing
# 
# **NOTES:**
# The preprocessing is inplace.
# 
# The last step is a out of place conversion to pytorch tensor.
preprocess_pipe = Preprocessing(verbose=True)

__, __ = preprocess_pipe.fit_transform(data=gvars.get_raw_train_data(), models=gvars.get_raw_train_models())
__, __ = preprocess_pipe.transform(data=gvars.get_raw_test_data(), models=gvars.get_raw_test_models())


#########
# Init environment for mlflow
mlflow_ = init_mlflow_and_set_execution_ids(gvars)


#########
# Understanding the data
#   - selecting samples to view
#   - generating images in order to log them on mlflow (at the end or execution)
NSAMPLES = 3
train_samples = list()
test_samples = list()
for i in range(NSAMPLES):
    train_samples.append(random.randint(0,gvars.get_raw_train_data().get_nmb_of_elems()-1))
    test_samples.append(random.randint(0,gvars.get_raw_test_data().get_nmb_of_elems()-1))

log.info(f"TRAIN SAMPLES = {train_samples}")
log.info(f"TEST SAMPLES = {test_samples}")

train_logger = MLFlowDataAndModelsLogger(verbose=gvars.is_verbose())
train_logger.make_figs(
        indices=train_samples, 
        models=gvars.get_raw_train_models(), 
        data=gvars.get_raw_train_data().get_current_data(),
        colorbar=True, 
        res_str=gvars.get_raw_train_data().get_current_resolution_name()
    )

test_logger = MLFlowDataAndModelsLogger(verbose=gvars.is_verbose())
test_logger.make_figs(
        indices=test_samples, 
        models=gvars.get_raw_test_models(), 
        data=gvars.get_raw_test_data().get_current_data(),
        colorbar=True, 
        res_str=gvars.get_raw_train_data().get_current_resolution_name()
    )


##########
# Grid creation
grid = RefitGrid(global_vars=gvars)
expanded_grid = grid.expand()


##########
# Define number of workers
is_gpu = torch.cuda.is_available()
if is_gpu:
    device_count = torch.cuda.device_count()
else:
    device_count = 1


###########
# Create (or read from) the job's directory for checkpoint purposes
def read_checkpoint_dir():
    raise NotImplemented

if gvars.is_restart():
    tasks = read_checkpoint_dir()
else:
    tasks = BuildTasks(verbose_level=1).build(expanded_grid=expanded_grid, device=gvars.get_device(), n=device_count)
    #create_checkpoint_dir(gvars.get_execution_name(), tasks)

task_processing = TaskProcessingGlobals(
        global_vars=gvars, 
        train_data=gvars.get_raw_train_data(),
        test_data=gvars.get_raw_test_data(),
        mlflow_=mlflow_,
        verbose_level=gvars.get_verbose_level(),
        processing_class=RefitTaskProcessing
    )

###########
# Init parallelism and start workers
with ThreadPool(min(len(tasks),device_count)) as p:
    results = p.map(task_processing, tasks)
    print(f"mse_eval  - mean, std = {results[0][0].mean()}, {results[0][0].std()}") 
    print(f"ssim_eval - mean, std = {results[0][1].mean()},  {results[0][1].std()}") 

mlflow_client, run = MLFlowUtils.create_client_and_get_run(
    mlflow_=mlflow_, 
    experiment_id=gvars.get_experiment_tracking_id(),
    execution_id=gvars.get_execution_id(), 
    execution_name=gvars.get_execution_name()
)

train_title_prefix="TRAIN_" + gvars.get_data_resolution() + "res"
test_title_prefix="TEST_" + gvars.get_data_resolution() + "res" 
train_logger.log_client(mlflow_client=mlflow_client, run_id=run.info.run_id, title_prefix=train_title_prefix)
test_logger.log_client(mlflow_client=mlflow_client, run_id=run.info.run_id, title_prefix=test_title_prefix)

# # Creating machine learning model
# 
# **NOTE**: Using skorch.
# 
# 


#3. Python Script
#You can also use a Python script to define the parameters directly. This can be useful if you need more complex logic.
#
#grid_search_params.py:
#
#param_grid = {
#    "n_estimators": [50, 100, 200],
#    "max_depth": [3, 5, 7],
#    "learning_rate": [0.01, 0.1, 0.2]
#}
#Python code to read the script:
#
#import importlib.util
#
#spec = importlib.util.spec_from_file_location("grid_search_params", "grid_search_params.py")
#grid_search_params = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(grid_search_params)
#
#params = grid_search_params.param_grid
#print(params)


log.info("FINALIZANDO COM EXIT --- RETIRAR ISSO")
os._exit(0)





# GS CV
run_name = f"CLI_TEST_{gvars.get_execution_id()}_{gvars.get_timestamp()}"
with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
    run_id = run.info.run_id
    log.info(f"Initiating running. run_id =  {run.info.run_id} -- run_name = {run.info.run_name}")
    grid_result = grid.fit(x_train, y_train)
    MLFlowUtils.log_results(grid, model_name=None, run_name=run.info.run_name)#, tags={}, log_only_best=False)
    #skmodel.fit(y_train,y_train)
#    #print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

log.info("Summarize results")
# Access the best hyperparameters found by GridSearchCV
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    log.info("%f (%f) with: %r" % (-mean, -stdev, param))


#run_name = f"CLI_TEST_{gvars.get_execution_id()}_{gvars.get_timestamp()}"
#with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
#    run_id = run.info.run_id
#    log.info(f"TRAINING MODEL:")
#    cv_results = cross_val_score(skmodel, x_train, y_train, cv=5, error_score='raise')#, scoring=None)
#    mlflow_.log_metric(metric_name="CV SCORE NEG_MSE MEAN", value=cv_results.mean())
#    mlflow_.log_metric(metric_name="CV SCORE NEG_MSE STD DEV", value=cv_results.std())
#print("cv_results = ", cv_results)
    #model_path = gvars.get_execution_id()+gvars.sep+gvars.get_data_type_str()+gvars.sep+gvars.get_timestamp()
    #mlflow_.log_model(model=skmodel, artifact_path=model_path, 
    #                  inputs=x_test.numpy(), predictions=y_test_hat_skorch,
    #                  conda_env=None
    #                ) # nao estah funcionando para modelos grandes.

#mlflow.evaluate(model=None, data=None, *, model_type=None, targets=None, predictions=None, dataset_path=None,
#                feature_names=None, evaluators=None, evaluator_config=None, custom_metrics=None,
#                extra_metrics=None, custom_artifacts=None, validation_thresholds=None, baseline_model=None,
#                env_manager='local', model_config=None, baseline_config=None, inference_params=None)

#print(dirs.get_results_dirname() + 'skorch_net_highres_model_13x13-kernel__'+utl.get_timestamp()+'.pkl')

#with open(dirs.get_results_dirname() + 'skorch_net_highres_model_13x13-kernel__'+utl.get_timestamp()+'.pkl', 'wb') as f:
#    pickle.dump(skmodel, f)

# saving entire model

def get_weights_and_biases_filename(base):
    return gvars.get_dirs().get_results_dirname()+'WEIGHTS-BIASES_'+base+'.pkl'

def get_history_filename(base):
    return gvars.get_dirs().get_results_dirname()+'HISTORY_'+base+'.json'

def get_optimizer_filename(base):
    return gvars.get_dirs().get_results_dirname()+'OPTIMIZER_'+base+'.pkl'

if gvars.get_data_max_samples() is None: 
    if gvars.is_newer_data():
        data_str='__new-data__'
    else:
        data_str='__old-data__'
    
    filename_base=gvars.get_execution_id()+data_str+utl.get_timestamp()
    
    log.info("Saving full model...")
    log.info(f"\tWeights and biases: {get_weights_and_biases_filename(filename_base)}")
    log.info(f"\tOptimizer: {get_optimizer_filename(filename_base)}")
    log.info(f"\tHistory: {get_history_filename(filename_base)}")
    skmodel.save_params(
            f_params=get_weights_and_biases_filename(filename_base),
            f_optimizer=get_optimizer_filename(filename_base),
            f_history=get_history_filename(filename_base)
        )
    log.ok()











# MODEL EVALUATION

import torch
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter


#import torch
#import torch.nn.functional as F
#
#def ssim_loss(img1, img2):
#    C1 = 0.01 ** 2
#    C2 = 0.03 ** 2
#    
#    mu1 = F.conv2d(img1, weight=torch.ones(1, 1, 3, 3) / 9, padding=1)
#    mu2 = F.conv2d(img2, weight=torch.ones(1, 1, 3, 3) / 9, padding=1)
#    
#    mu1_sq = mu1 ** 2
#    mu2_sq = mu2 ** 2
#    mu1_mu2 = mu1 * mu2
#    
#    sigma1_sq = F.conv2d(img1 ** 2, weight=torch.ones(1, 1, 3, 3) / 9, padding=1) - mu1_sq
#    sigma2_sq = F.conv2d(img2 ** 2, weight=torch.ones(1, 1, 3, 3) / 9, padding=1) - mu2_sq
#    sigma12 = F.conv2d(img1 * img2, weight=torch.ones(1, 1, 3, 3) / 9, padding=1) - mu1_mu2
#    
#    ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
#    ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
#    
#    ssim = ssim_n / ssim_d
#    return 1 - ssim.mean()
#
## Example usage:
#img1 = torch.rand((1, 1, 256, 256))  # Batch of images
#img2 = torch.rand((1, 1, 256, 256))
#
#loss = ssim_loss(img1, img2)
#print(loss.item())



def ssim(image1, image2, data_range=1.0):
    # Convert torch tensors to numpy arrays
    image1_np = image1.detach().cpu().numpy()
    image2_np = image2.detach().cpu().numpy()

    # Constants for SSIM calculation
    K1, K2 = 0.01, 0.03
    L = data_range

    # Compute mean and variance
    mu1 = gaussian_filter(image1_np, sigma=1.5)
    mu2 = gaussian_filter(image2_np, sigma=1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(image1_np ** 2, sigma=1.5) - mu1_sq
    sigma2_sq = gaussian_filter(image2_np ** 2, sigma=1.5) - mu2_sq
    sigma12 = gaussian_filter(image1_np * image2_np, sigma=1.5) - mu1_mu2

    # Calculate SSIM components
    num = (2 * mu1_mu2 + K1) * (2 * sigma12 + K2)
    den = (mu1_sq + mu2_sq + K1) * (sigma1_sq + sigma2_sq + K2)
    ssim_map = num / den

    # Compute the mean SSIM over the entire image
    ssim_value = np.mean(ssim_map)

    return ssim_value




def mse(model, x, y):
    y_hat = model.predict(x)
    mseloss = nn.MSELoss()
    y_cpu = y.to('cpu')
    return  mseloss(torch.from_numpy(y_hat), y_cpu), y_hat, y_cpu



mse_loss_skorch, y_test_hat_skorch, y_test_cpu_skorch = mse(skmodel, x_test, y_test)




# ## Visualizing results

# ### Reshaping tensors



x_test = x_test.to('cpu')

y_test_cpu_skorch = y_test_cpu_skorch.reshape((y_test_cpu_skorch.shape[0], y_test_cpu_skorch.shape[2], y_test_cpu_skorch.shape[3]))
y_test_hat_skorch = y_test_hat_skorch.reshape((y_test_hat_skorch.shape[0], y_test_hat_skorch.shape[2], y_test_hat_skorch.shape[3]))




# ### View



print(test_samples)
#test_samples[0] = 135


# #### Fixed samples



VMIN=-1.7
VMAX=1.7

print(f"min = {np.min(y_test_hat_skorch)}, max = {np.max(y_test_hat_skorch)}, mean = {np.mean(y_test_hat_skorch)}, var = {np.var(y_test_hat_skorch)}")
print(f"min = {np.min(y_test_cpu_skorch.numpy())}, max = {np.max(y_test_cpu_skorch.numpy())}, mean = {np.mean(y_test_cpu_skorch.numpy())}, var = {np.var(y_test_cpu_skorch.numpy())}")

viewer.yy_view(y=y_test_cpu_skorch, y_hat=y_test_hat_skorch, nsamples=2, samples=test_samples,
               vmin=VMIN, vmax=VMAX, colorbar=True, title="SKORCH - HIGH RES")


# Evaluating images
# MSE
mse_loss = nn.MSELoss()

amostras_fixas=range(1,20,2)

individual_losses_skorch = np.zeros(y_test_cpu_skorch.shape[0], dtype=float)
log.info(f"indidual_losses.shape = {individual_losses_skorch.shape}")

for k in range(individual_losses_skorch.shape[0]):
    individual_losses_skorch[k] = mse_loss(y_test_cpu_skorch[k], torch.from_numpy(y_test_hat_skorch[k]))

yy_mse_figs = viewer.yy_view(y=y_test_cpu_skorch, y_hat=y_test_hat_skorch, loss=individual_losses_skorch, samples=amostras_fixas,
                 vmin=VMIN, vmax=VMAX, colorbar=True, title="MSE - HIGH RES")

mse_skorch_mean = individual_losses_skorch.mean()
mse_skorch_loss_scoring = loss_scoring(skmodel, x_test, y_test)

                
# SSIM
individual_indexes_skorch = np.zeros(y_test_cpu_skorch.shape[0], dtype=float)

for k in range(individual_indexes_skorch.shape[0]):
    individual_indexes_skorch[k] = ssim(y_test_cpu_skorch[k], torch.from_numpy(y_test_hat_skorch[k]))

yy_ssim_figs = viewer.yy_view(y=y_test_cpu_skorch, y_hat=y_test_hat_skorch, loss=individual_indexes_skorch, samples=amostras_fixas,
                 vmin=VMIN, vmax=VMAX, colorbar=True, title="SSIM - HIGH RES")
                

ssim_skorch_mean = individual_indexes_skorch.mean()

log.info(f"MSE LOSS - method 1: {mse_loss_skorch}")
log.info(f"MSE LOSS - method 2: {mse_skorch_loss_scoring}")
log.info(f"MSE LOSS  - method 3: {mse_skorch_mean}")
log.info(f"SSIM INDEX:           {ssim_skorch_mean}")

with mlflow.start_run(run_id=run_id):
    mlflow_.log_metric(metric_name="MSE loss evaluation - 1", value=mse_loss_skorch)
    mlflow_.log_metric(metric_name="MSE loss evaluation - 2", value=mse_skorch_loss_scoring)
    mlflow_.log_metric(metric_name="MSE loss evaluation - 3", value=mse_skorch_mean)
    mlflow_.log_metric(metric_name="SSIM index evaluation", value=ssim_skorch_mean)
    mlflow_.log_figure(yy_mse_figs, "MSE_EVAL.png")
    mlflow_.log_figure(yy_ssim_figs, "SSIM_EVAL.png")
    train_logger.log(mlflow_obj=mlflow_, title_prefix="TRAIN_HIGHRES")
    test_logger.log(mlflow_obj=mlflow_, title_prefix="TEST_HIGHRES")
# ## DAQUI PARA BAIXO É RASCUNHO



#viewer.xy_view(data=train_data.get_lowres_data(), nsamples=2, samples=samples, colorbar=True, title="LOW RES")
#viewer.xy_view(data=train_data.get_mediumres_data(), nsamples=2, samples=samples, colorbar=True, title="MEDIUM RES")
#viewer.xy_view(data=train_data.get_highres_data(), nsamples=2, samples=samples, colorbar=True, title="HIGH RES")



