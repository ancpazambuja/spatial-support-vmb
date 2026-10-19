

"""@package task_processing.py
Code for thread processing.
"""


############################
# basic stuff
import os
from datetime import datetime
import time
import random
import sys
from glob import glob
import pickle


import numpy as np
import seaborn as sns

############################
# experiment tracking
import mlflow
from mlflow import MlflowClient


############################
# ml stuff
import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
import torch
import torch.nn as nn
from torch.optim import SGD, Adam #, Adagrad, Adamax, NAdam, Adadelta, RMSprop
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from skorch import dataset
from skorch.callbacks import MlflowLogger, Checkpoint


############################
# internal modules

from src.experiment_tracking.mlflow_utils import MLFlowUtils, MLFlowDataAndModelsLogger
from src.experiment_tracking.mlflow import MLFlow
from src.experiment_tracking.mlflow_skorch_callbacks import ParamLogger

from src.processing.gs_grid import GridSearchElementResult
from src.processing.ml_models_builder import SkorchTransUNetModelBuilder, ScoredSkorchModel
from src.processing.ml_models import TransUNet
from src.processing.ml_models_utils import MLModelsUtils

from src.utl.logging import Log as log

from src.utl.classes import Utils as utl
from src.utl.classes import Graph as graph

from src.visualization.data_viewer import DataAndModelsViewer as viewer

from src.preprocessing.pipeline import Preprocessing

from src.apps.command_line_interface import TrainCLI
from src.apps.global_vars import GlobalVars

def get_result(index=None, gs=None):
    print(f"index = {index}")
    print(gs)
    return GridSearchElementResult(
        index=index,
        mean_train_score = -gs.cv_results_['mean_train_score'][0],
        std_train_score = -gs.cv_results_['std_train_score'][0],
        mean_valid_score = -gs.cv_results_['mean_test_score'][0],
        std_valid_score = -gs.cv_results_['std_test_score'][0],
    )

class Task:
    """
    The task to be processed.
    """
    def __init__(self, id=0, device='cpu', idxs=None, items=None, verbose_level=0):
        self.verbose_level = verbose_level
        self.verbose = True if verbose_level > 0 else False
        self.device = device
        self.id = id # task index inside global grid.

        assert len(idxs) == len(items), f"'idxs' and 'items' must have the same size! #idxs={len(idxs)} and #items={len(items)}"
        self.items = items
        self.global_idxs = idxs

    def get_id(self):
        return self.id

    def get_device(self):
        return self.device

    def set_device(self, device='cpu'):
        self.device = device

    def get_item_global_idx(self, idx):
        assert idx >= 0 and idx < self.get_n_items(), f'Invalid index! (idx = {idx})'
        return self.global_idxs[idx]

    def get_item(self, idx):
        assert idx >= 0 and idx < self.get_n_items(), f'Invalid index! (idx = {idx})'
        return self.items[idx]

    def get_all_items_global_idxs(self):
        return self.global_idxs

    def get_all_items(self):
        return self.items

    def get_n_items(self):
        return len(self.items)

class BuildTasks:
    """
    The task to be processed.
    """
    def __init__(self, verbose_level=None):
        self.verbose_level = verbose_level
        self.verbose = True if verbose_level > 0 else False

    def build(self, expanded_grid=None, device='cpu', n=1):
        '''
        n is the number of workers.
        '''
        tasks = list()
        l = len(expanded_grid)
        log.info(f"l = {l}")
        for i in range(0,min(l,n)): 
            idxs = range(i,l,n) #using rounding robin in order to try to keep work balance
            log.info(f"idxs = {idxs}")
            log.info(f"idxs.len = {len(idxs)}")
            items= list()
            for j in range(0,len(idxs)):
                #print(expanded_grid[j])
                items.append(expanded_grid[idxs[j]])
            #for j in idxs:
            #    items.append(expanded_grid[j])
            tasks.append(Task(id=i, device=device, idxs=idxs, items=items, verbose_level=self.verbose_level))

        # treat for 'gpu:all' case
        if  device == 'cuda:all':
            for i in range(0,len(tasks)):
                tasks[i].set_device(f'cuda:{i}')

        if self.verbose:
            log.info(f"Number of tasks = {len(tasks)}")
            log.info(f"----- TASKS -----")
            for i, t in enumerate(tasks):
                log.info(f"task number = {i}")
                log.info(f"id          = {t.id}")
                log.info(f"global idxs = {t.global_idxs}")
                log.info(f"device      = {t.device}")
                log.line()

        return tasks

class GridSearchProcessing:
    """
    Execute the grid search with cross validation using scilearn framework.
    """
    def __init__(self, estimator=None, grid=None, cv=5, verbose_level=0):
        self.verbose_level = verbose_level
        self.verbose = True if verbose_level > 0 else False

        self.estimator = estimator
        self.grid = grid
        self.cv = cv

        self.x_train = None
        self.y_train = None

        self.gs = GridSearchCV(
                estimator=self.estimator, 
                param_grid=self.grid, 
                cv=self.cv, 
                error_score='raise', 
                return_train_score=True, 
                n_jobs=1, 
                refit=False, 
                verbose=True
            )

    def fit(self, x_train=None, y_train=None):
        self.x_train = x_train
        self.y_train = y_train

        if self.verbose:
            log.info(f"[ START ] Grid search with cv = {self.cv}.")

        results = self.gs.fit(x_train, y_train)

        if self.verbose:
            log.info(f"[ END ] Grid search")

        return self.gs, results




class TaskProcessing:
    def __init__(self, task_processing_globals=None, task=None):
        self.g = task_processing_globals
        self.task = task

        self.model_data_size = None
        self.model = None
        self.cropped_x = None
        self.cropped_y = None
        self.cropped_x_test = None
        self.cropped_y_test = None
        self.device = None
        self.device = torch.device(self.task.get_device()) # construct an object of class torch.device

    def run(self):
        assert str(self.device) == self.task.get_device(), f"Logical internal error"
        utl.set_seed(self.g.gvars.get_seed())

        # find model architecture compatible sizes
        self.model_data_size = self.get_model_data_size()
        if self.g.verbose:
            log.info(f"model_data_size: {self.model_data_size}")

        # set the target device
        if torch.cuda.is_available():
            print (f"-----------------------")
            print (f"cuda current device : {torch.cuda.current_device()}")
            print (f"self.device : {self.device}")
            print (f"-----------------------")
            torch.cuda.device(self.device)
            print (f"[ thread {self.task.get_id()}cuda current device : {torch.cuda.current_device()}")
            torch.cuda.empty_cache()
            self.stream = torch.cuda.Stream(device=self.device)
        #self.set_torch_device(self.task.get_device())
        
        with torch.cuda.stream(self.stream):

            # deal with mlflow client and run
            mlflow_client, run = MLFlowUtils.create_client_and_get_run(
                mlflow_=self.g.mlflow, 
                experiment_id=self.g.gvars.get_experiment_tracking_id(),
                execution_id=self.g.gvars.get_execution_id(), 
                execution_name=self.g.gvars.get_execution_name()
            )

            # create callbacks for trainning
            gs_cbs, _, lr_scheduler, patience = self.create_callbacks(mlflow_client=mlflow_client, run=run)

            # create model
            print (f"-----------------------")
            print (f"[ thread {self.task.get_id()} cuda current device : {torch.cuda.current_device()}")
            self.model = self.create_model(cb=gs_cbs, device=self.device)

            # create a view of the data from self.model_data_size
            self.cropped_x, self.cropped_y, _, _ = self.crop_data(device=None) #self.device) # Testar com True

            if self.g.verbose:
                log.info(f"[ START ] Grid search for task id {self.task.get_id()} with {self.task.get_n_items()} hyper-parameters instances.")

            # Deal with mlflow run setup
            #with mlflow.start_run(run_id=self.g.gvars.get_execution_id(), log_system_metrics=True) as run:
            best_result = GridSearchElementResult()
            for i,t in enumerate(self.task.get_all_items()):
                log.info(f"[ {i} ]")
                gscv = GridSearchProcessing(
                    estimator=self.model, 
                    grid=[t], #task.get_item(i), 
                    verbose_level=self.g.verbose_level,
                    cv=self.g.cv, 
                    )
                gs, results = gscv.fit(x_train=self.cropped_x, y_train=self.cropped_y)
                new_result =  get_result(index=self.task.get_item_global_idx(i),gs=gs)
                updated = best_result.update(r=new_result)
                if self.g.verbose:
                    log.info(f"New best value! global index:{best_result.index}, mean_valid_score={best_result.mean_valid_score} ")
                #print(f"gs.param_grid = {gs.param_grid}")
                
                torch_model = self.model.module_
                model_size=sum(p.numel() for p in torch_model.parameters())
                # log in mlflow the result
                MLFlowUtils.log_results(
                    mlflow_client=mlflow_client, experiment_id=self.g.mlflow.get_experiment_id(), run=run, base_id=self.task.get_item_global_idx(i),
                    gridsearch=gs, model_size=model_size, lr_scheduler=lr_scheduler,
                    patience=patience, seed=self.g.gvars.get_seed(),
                    test_perc=self.g.gvars.get_test_perc(), train_indexes=self.g.gvars.get_train_indexes(), test_indexes=self.g.gvars.get_test_indexes()

                ) #, tags={}, log_only_best=False)
                # save checkpoint
                # save()
                #update best result
                #best_result = ....
        if self.g.verbose:
            log.info(f"[ END ] Grid search finished.")

        return best_result

    def crop_data(self, device=None):

        if self.g.verbose_level > 0: 
            log.info(f"DEVICE = {device}")
    
        # X - expected shape (batch, nchannels, nz, nx)
        nz_cut = self.model_data_size['nz_cut']
        nx_cut = self.model_data_size['nx_cut']
        x_train = self.g.train_data.get_current_data().get_X()[:,:,:nz_cut,:nx_cut]
        x_test = self.g.test_data.get_current_data().get_X()[:,:,:nz_cut,:nx_cut]
        if self.g.verbose_level > 0: 
            log.info(f"x_train.shape = {x_train.shape}")
            log.info(f"x_test.shape = {x_test.shape}")
        
        # y_train - expected shape (batch, nz, nx)
        y_train = self.g.train_data.get_current_data().get_Y()[:,:nz_cut,:nx_cut]
        y_test = self.g.test_data.get_current_data().get_Y()[:,:nz_cut,:nx_cut]
        if self.g.verbose_level > 0: 
            log.info(f"y_train.shape = {y_train.shape}    --- original shape")
            log.info(f"y_test.shape = {y_test.shape}    --- original shape")

        # y_train - reshaped: from (batch, nz, nx) to (batch, 1, nz, nx)
        y_shape = y_train.shape
        y_new_shape = (y_shape[0], 1, y_shape[1], y_shape[2])
        y_train = y_train.reshape(y_new_shape)

        y_test_shape = y_test.shape
        y_test_new_shape = (y_test_shape[0], 1, y_test_shape[1], y_test_shape[2])
        y_test = y_test.reshape(y_test_new_shape)

        assert y_train.shape[1:-1] == y_test.shape[1:-1], f"Internal Error! y_train and y_test must have the same shape (y_train.shape = {y_train.shape}, y_test.shape = {y_test.shape})"


        if self.g.verbose_level > 0: 
            log.info(f"y_train.shape = {y_train.shape}    --- after reshaping")
            log.info(f"y_test.shape = {y_test.shape}    --- after reshaping")
        
        if self.g.verbose_level > 0: 
            log.info(f"x_train,y_train current devices = {x_train.device}, {y_train.device}")
            log.info(f"x_test,y_test current devices = {x_test.device}, {y_test.device}")

        if device is not None:
            x_train = x.to(device)
            y_train = y.to(device)
            if self.g.verbose_level > 0: 
                log.info(f"x_train,y_train after to(device) = {x_train.device}, {y_train.device}")
                log.info(f"x_test,y_test after to(device) = {x_test.device}, {y_test.device}")

        return x_train, y_train, x_test, y_test

    def get_model_data_size(self):
        return utl.get_model_data_size(self.g.train_data, depth=5)

    #def set_torch_device(self, device):
    #    self.device = torch.device(device)
    #    if torch.cuda.is_available():
    #        torch.cuda.empty_cache()

    def create_callbacks(self, mlflow_client=None, run=None, is_one_step_training=True):
        return utl.get_train_callbacks(global_vars=self.g.gvars, mlflow_client=mlflow_client, run=run, is_one_step_training=is_one_step_training)

    def create_model(self, cb=None, device=None):

        if self.g.verbose_level > 0:
            n_channels = self.model_data_size['nchannels']
            log.info(f"N_CHANNELS = {n_channels}")
            log.info(f"CLI - model_arch = {self.g.gvars.get_model_arch()}")
            log.info(f"is_transunet = {self.g.gvars.is_transunet()}")
            log.info(f"task device = {self.task.get_device()}")
            log.info(f"device = {device}")
            log.info(f"device type = {type(device)}")
            log.info(f"cuda current device = {torch.cuda.current_device()}")

        # size of the attention size
        if self.g.gvars.get_data_type() == "shallow":
            attention_img_size=(11,28) #10x28=280, sequence length quando patches.size=(1,1)
        elif self.g.gvars.get_data_type() == "deep": 
            attention_img_size=(21, 25) #21x25 = 525, sequence length quando patches.size=(1,1)
        else:
            raise ValueError

        model = SkorchTransUNetModelBuilder(verbose=self.g.verbose).build(
            seed=self.g.gvars.get_seed(),
            max_epochs=self.g.gvars.get_max_epochs(),
            conv_kernel_size=self.g.gvars.get_conv_filter_size(),
            conv_dilation_size=self.g.gvars.get_conv_dilation_size(),
            transposed_conv_kernel_size=self.g.gvars.get_transposed_conv_filter_size(),
            use_transformers=self.g.gvars.is_transunet(),
            unembed_type=self.g.gvars.get_unembed_type(),

            callbacks=cb,

            device=device,
            batch_size=16,
            #criterion=nn.SSIM(),
            criterion=nn.MSELoss(),
            optimizer=torch.optim.Adam,
            lr=0.0001,
            in_channels=n_channels, out_channels=1, n_filters_init=16,
            pool_kernel_size=(2,2), pool_stride=(2,2),
            batch_normalization=True, dropout=0.2,
            #img_size=(672,800),
            attention_img_size=attention_img_size, #21x25=525 == sequence length quando patches.size=(1,1)
            attention_in_channels=512,  # n_channels of the last encoding feature map
            attention_hidden_size=512, # D, embedding size 
            attention_num_heads=16,
            attention_num_layers=12,
            attention_mlp_dim=4096,
            vis=False,
            )

        if self.g.verbose:
            model.initialize()
            input_size = ( 
                1, 
                self.model_data_size['nchannels'],
                self.model_data_size['nz_cut'], 
                self.model_data_size['nx_cut'],
            )
            print(model.module_.summary(input_size=input_size))

        return model

class RefitTaskProcessing(TaskProcessing):
    def __init__(self, task_processing_globals=None, task=None):
        super().__init__(task_processing_globals=task_processing_globals, task=task)

    def run(self):
        assert str(self.device) == self.task.get_device(), f"Logical internal error"
        utl.set_seed(self.g.gvars.get_seed())

        # find model architecture compatible sizes
        self.model_data_size = self.get_model_data_size()
        if self.g.verbose:
            log.info(f"model_data_size: {self.model_data_size}")
            log.info(f"[ START ] Refit using taks id {self.task.get_id()} with {self.task.get_n_items()} hyper-parameters instances.")

        # set the target device
        if torch.cuda.is_available():
            print (f"-----------------------")
            print (f"cuda current device : {torch.cuda.current_device()}")
            print (f"self.device : {self.device}")
            print (f"-----------------------")
            torch.cuda.device(self.device)
            print (f"[ thread {self.task.get_id()}cuda current device : {torch.cuda.current_device()}")
            torch.cuda.empty_cache()
            self.stream = torch.cuda.Stream(device=self.device)
        #self.set_torch_device(self.task.get_device())
        
        with torch.cuda.stream(self.stream):

            # deal with mlflow client and run
            mlflow_client, run = MLFlowUtils.create_client_and_get_run(
                mlflow_=self.g.mlflow, 
                experiment_id=self.g.gvars.get_experiment_tracking_id(),
                execution_id=self.g.gvars.get_execution_id(), 
                execution_name=self.g.gvars.get_execution_name()
            )

            # create callbacks for trainning
            _, refit_cbs, lr_scheduler, patience = self.create_callbacks(
                mlflow_client=mlflow_client, 
                run=run,
                is_one_step_training=self.g.gvars.is_one_step_training(),
            )
            log.info(f"refit_cbs = {refit_cbs}")

            # create model
            print (f"-----------------------")
            print (f"[ thread {self.task.get_id()} cuda current device : {torch.cuda.current_device()}")
            self.model = self.create_model(cb=refit_cbs, device=self.device, grid=self.task.get_item(0))

            # create a view of the data from self.model_data_size
            self.cropped_x, self.cropped_y, self.cropped_x_test, self.cropped_y_test = self.crop_data(device=None)

            # Refit 
            # Note: model must be parameterized to:
            #   i) splits data into train and validation, 
            #   ii) performs checkpoint and early stopping,
            #   iii) load the best model at the end ('load_best'=True) 
            self.model.fit(self.cropped_x, self.cropped_y) 

            # Evaluate
            # predict (it worked because checkpoint was set with 'load_best=true')
            self.model.module_.eval() # it might not be necessary, because skorch probably does it internally in predict
            y_hat = self.model.predict(self.cropped_x_test)
            # use mse (loss) and ssim (index) metrics
            mae, _  = self.mae(self.cropped_y_test, y_hat)
            mse, _  = self.mse(self.cropped_y_test, y_hat)
            ssim, ssim_qmap = self.ssim(self.cropped_y_test, y_hat)
            if self.g.verbose:
                log.info(f"MSE mean over test set is: {mse.mean()}")
                log.info(f"SSIM mean over test set is: {ssim.mean()}")
            self.log_evaluation_results(
                mlflow_client=mlflow_client, run=run, 
                mae=mae, mse=mse, ssim=ssim, ssim_qmap=ssim_qmap, 
                y=self.cropped_y_test, y_hat=y_hat
            )

            torch_model = self.model.module_
            model_size=sum(p.numel() for p in torch_model.parameters())
            self.log_parameters(mlflow_client=mlflow_client, run=run, 
                params=self.task.get_item(0), 
                model_size=model_size, lr_scheduler=lr_scheduler,
                patience=patience, seed=self.g.gvars.get_seed(),
                test_perc=self.g.gvars.get_test_perc(), train_indexes=self.g.gvars.get_train_indexes(), test_indexes=self.g.gvars.get_test_indexes()
            )

        if self.g.verbose:
            log.info(f"[ END ] Refit finished.")

        return mse, ssim

    #def set_torch_device(self, device):
    #    self.device = torch.device(device)
    #    if torch.cuda.is_available():
    #        torch.cuda.empty_cache()

    def create_model(self, cb=None, device=None, grid=None):

        if self.g.verbose_level > 0:
            n_channels = self.model_data_size['nchannels']
            log.info(f"N_CHANNELS = {n_channels}")
            log.info(f"CLI - model_arch = {self.g.gvars.get_model_arch()}")
            log.info(f"is_transunet = {self.g.gvars.is_transunet()}")
            log.info(f"task device = {self.task.get_device()}")
            log.info(f"device = {device}")
            log.info(f"device type = {type(device)}")
            log.info(f"cuda current device = {torch.cuda.current_device()}")
            log.info(f"type of grid = {type(grid)}")
            log.info(f"grid = {grid}")


        model = ScoredSkorchModel(
            module=TransUNet,
            device=device,
            callbacks=cb,
            seed=self.g.gvars.get_seed(),
            **grid
        )

        if self.g.verbose:
            model.initialize()
            input_size = ( 
                1, 
                self.model_data_size['nchannels'],
                self.model_data_size['nz_cut'], 
                self.model_data_size['nx_cut'],
            )
            print(model.module_.summary(input_size=input_size))

        return model

    def mae(self, y, y_hat):
        return self.evaluate(y, y_hat, metric=nn.L1Loss(), qmap=False)

    def mse(self, y, y_hat):
        return self.evaluate(y, y_hat, metric=nn.MSELoss(), qmap=False)

    def ssim(self, y, y_hat):
        return self.evaluate(y, y_hat, metric=utl.ssim, qmap=True)

    def evaluate(self, y, y_hat, metric=None, qmap=True):
        assert metric is not None, f"Internal error! A metric must be passed."
        assert y.shape == y_hat.shape, f"Internal error! y and y_hat must have the same size. (y.shape = {y.shape}, y_hat.shape = {y.shape})"

        metric_individual = np.zeros((y.shape[0]))
        if qmap:
            quality_map = np.zeros((y.shape[0],y.shape[2],y.shape[3]))
        else:
            quality_map = None


        for k in range(0,y.shape[0]):
            if qmap:
                metric_individual[k], quality_map[k] = metric(y[k,0], torch.from_numpy(y_hat[k,0]))
            else:
                metric_individual[k] = metric(y[k,0], torch.from_numpy(y_hat[k,0]))

        return  metric_individual, quality_map

    def log_evaluation_results(self, mlflow_client=None, run=None, mae=None, mse=None, ssim=None, ssim_qmap=None, y=None, y_hat=None):
        assert y.shape == y_hat.shape, f"y and y_hat must be the same size! y.shape = {y.shape} and y_hat.shape = {y_hat.shape}"
        assert y.shape[0] == len(mae), f"y and mse must be the same size! len(y) = {y.shape[0]} and len(mse) = {len(mae)}"
        assert y.shape[0] == len(mse), f"y and mse must be the same size! len(y) = {y.shape[0]} and len(mse) = {len(mse)}"
        assert y.shape[0] == len(ssim), f"y and ssim must be the same size! len(y) = {y.shape[0]} and len(ssim) = {len(ssim)}"

        run_id = run.info.run_id

        for t,v in enumerate(mae): 
            mlflow_client.log_metric(run_id=run_id, key="mae_values", value=v, step=t)
        mlflow_client.log_metric(run_id, "mae_mean_test_set", mae.mean())
        mlflow_client.log_metric(run_id, "mae_std_test_set", mae.std())
        mlflow_client.log_metric(run_id, "mae_min_test_set", mae.min())
        mlflow_client.log_metric(run_id, "mae_max_test_set", mae.max())

        for t,v in enumerate(mse): 
            mlflow_client.log_metric(run_id=run_id, key="mse_values", value=v, step=t)
        mlflow_client.log_metric(run_id, "mse_mean_test_set", mse.mean())
        mlflow_client.log_metric(run_id, "mse_std_test_set", mse.std())
        mlflow_client.log_metric(run_id, "mse_min_test_set", mse.min())
        mlflow_client.log_metric(run_id, "mse_max_test_set", mse.max())

        for t,v in enumerate(ssim):
            mlflow_client.log_metric(run_id=run_id, key="ssim_values", value=v, step=t)
        mlflow_client.log_metric(run_id, "ssim_mean_test_set", ssim.mean())
        mlflow_client.log_metric(run_id, "ssim_std_test_set", ssim.std())
        mlflow_client.log_metric(run_id, "ssim_min_test_set", ssim.min())
        mlflow_client.log_metric(run_id, "ssim_max_test_set", ssim.max())
        
        y2 = y.reshape((y.shape[0], y.shape[2], y.shape[3]))
        y_hat2 = y_hat.reshape((y_hat.shape[0], y_hat.shape[2], y_hat.shape[3]))

        yy_mae_figs = viewer.yy_view(y=y2, y_hat=y_hat2, loss=mae, samples=list(range(len(mae))),
                 vmin=-1.7, vmax=1.7, colorbar=True, title="MAE - Evaluation")

        yy_mse_figs = viewer.yy_view(y=y2, y_hat=y_hat2, loss=mse, samples=list(range(len(mse))),
                 vmin=-1.7, vmax=1.7, colorbar=True, title="MSE - Evaluation")

        yyq_ssim_figs = viewer.yyq_view(y=y2, y_hat=y_hat2, qmap=ssim_qmap, loss=ssim, samples=list(range(len(ssim))),
                 vmin=-1.7, vmax=1.7, colorbar=True, title="SSIM - Evaluation")

        mlflow_client.log_figure(run_id, yy_mae_figs, "MAE_EVAL.png")
        mlflow_client.log_figure(run_id, yy_mse_figs, "MSE_EVAL.png")
        mlflow_client.log_figure(run_id, yyq_ssim_figs, "SSIM_EVAL.png")

    def log_parameters(self, mlflow_client=None, run=None, params=None, model_size=None,
                       lr_scheduler=None, patience=None, seed=None, 
                       test_perc=None, train_indexes=None, test_indexes=None):
        run_id=run.info.run_id
        mlflow_client.log_param(run_id=run_id, key="model_size", value=model_size)
        mlflow_client.log_param(run_id=run_id, key="lr_scheduler", value=lr_scheduler)
        mlflow_client.log_param(run_id=run_id, key="patience", value=patience)
        mlflow_client.log_param(run_id=run_id, key="_x_seed", value=seed)
        mlflow_client.log_param(run_id=run_id, key="_x_test_perc", value=test_perc)
        mlflow_client.log_param(run_id=run_id, key="_x_train_indexes", value=train_indexes)
        mlflow_client.log_param(run_id=run_id, key="_x_test_indexes", value=test_indexes)
        MLFlowUtils.log_parameters_by_run(mlflow_client=mlflow_client, run=run, params=params)

class GrokkingTaskProcessing(RefitTaskProcessing):
    def __init__(self, task_processing_globals=None, task=None):
        super().__init__(task_processing_globals=task_processing_globals, task=task)

    def create_callbacks(self, mlflow_client=None, run=None):
        return utl.get_train_callbacks(global_vars=self.g.gvars, mlflow_client=mlflow_client, run=run, grokking=True)


    def run(self):
        assert str(self.device) == self.task.get_device(), f"Logical internal error"
        utl.set_seed(self.g.gvars.get_seed())

        # find model architecture compatible sizes
        self.model_data_size = self.get_model_data_size()
        if self.g.verbose:
            log.info(f"model_data_size: {self.model_data_size}")
            log.info(f"[ START ] Refit using taks id {self.task.get_id()} with {self.task.get_n_items()} hyper-parameters instances.")

        # set the target device
        if torch.cuda.is_available():
            print (f"-----------------------")
            print (f"cuda current device : {torch.cuda.current_device()}")
            print (f"self.device : {self.device}")
            print (f"-----------------------")
            torch.cuda.device(self.device)
            print (f"[ thread {self.task.get_id()}cuda current device : {torch.cuda.current_device()}")
            torch.cuda.empty_cache()
            self.stream = torch.cuda.Stream(device=self.device)
        #self.set_torch_device(self.task.get_device())
        
        with torch.cuda.stream(self.stream):

            # deal with mlflow client and run
            mlflow_client, run = MLFlowUtils.create_client_and_get_run(
                mlflow_=self.g.mlflow, 
                experiment_id=self.g.gvars.get_experiment_tracking_id(),
                execution_id=self.g.gvars.get_execution_id(), 
                execution_name=self.g.gvars.get_execution_name()
            )

            # create callbacks for trainning
            _, refit_cbs, lr_scheduler, patience = self.create_callbacks(mlflow_client=mlflow_client, run=run)
            grokking_cbs = refit_cbs
            log.info(f"refit_cbs = {grokking_cbs}")

            # create model
            print (f"-----------------------")
            print (f"[ thread {self.task.get_id()} cuda current device : {torch.cuda.current_device()}")
            self.model = self.create_model(cb=refit_cbs, device=self.device, grid=self.task.get_item(0))

            # create a view of the data from self.model_data_size
            self.cropped_x, self.cropped_y, self.cropped_x_test, self.cropped_y_test = self.crop_data(device=None)

            # Refit 
            # Note: model must be parameterized to:
            #   i) splits data into train and validation, 
            #   ii) performs checkpoint and early stopping,
            #   iii) load the best model at the end ('load_best'=True) 
            self.model.fit(self.cropped_x, self.cropped_y) 

            # Evaluate
            # predict (it worked because checkpoint was set with 'load_best=true')
            self.model.module_.eval() # it might not be necessary, because skorch probably does it internally in predict
            y_hat = self.model.predict(self.cropped_x_test)
            # use mse (loss) and ssim (index) metrics
            mae, _  = self.mae(self.cropped_y_test, y_hat)
            mse, _  = self.mse(self.cropped_y_test, y_hat)
            ssim, ssim_qmap = self.ssim(self.cropped_y_test, y_hat)
            if self.g.verbose:
                log.info(f"MSE mean over test set is: {mse.mean()}")
                log.info(f"SSIM mean over test set is: {ssim.mean()}")
            self.log_evaluation_results(
                mlflow_client=mlflow_client, run=run, 
                mae=mae, mse=mse, ssim=ssim, ssim_qmap=ssim_qmap, 
                y=self.cropped_y_test, y_hat=y_hat
            )

            torch_model = self.model.module_
            model_size=sum(p.numel() for p in torch_model.parameters())
            self.log_parameters(mlflow_client=mlflow_client, run=run, 
                params=self.task.get_item(0), 
                model_size=model_size, lr_scheduler=lr_scheduler,
                patience=patience, seed=self.g.gvars.get_seed(),
                test_perc=self.g.gvars.get_test_perc(), train_indexes=self.g.gvars.get_train_indexes(), test_indexes=self.g.gvars.get_test_indexes()
            )

        if self.g.verbose:
            log.info(f"[ END ] Refit finished.")

        return mse, ssim

    #def set_torch_device(self, device):
    #    self.device = torch.device(device)
    #    if torch.cuda.is_available():
    #        torch.cuda.empty_cache()




class TaskProcessingGlobals:
    """
    Process the task in thread-safe manner. 
    'gvars', 'train_data' and 'expanded_grid' will not be changed internally
    and must not be changed from outside after threads have started
    """
    def __init__(self, global_vars=None, train_data=None, test_data=None, cv=None,  mlflow_=None, verbose_level=None, processing_class=TaskProcessing):

        # global attributes - keep them constant for thread-safety, don't change them
        self.gvars = global_vars
        self.train_data = train_data
        self.test_data= test_data

        # global attributes - already initialized, 
        # but can be used to log results
        self.mlflow = mlflow_

        # private attribute - free to use, can be changed
        self.verbose_level = verbose_level
        self.verbose = True if verbose_level > 0 else False
        self.cv = cv
        self.processing_class = processing_class


        self.model_data_size = None
        self.task = None
        self.model = None

    def __call__(self, task):
        processing_class = self.processing_class 
        task_processing = processing_class(task_processing_globals=self, task=task)
        return task_processing.run()
    




###### ---->>>>  TUDO ISSO ABAIXO FOI PARA A CALLBACK TwoPhaseTraining
## ==========================================
## STEP 1: Train the Pure Bottleneck
## ==========================================
#print("Starting Step 1: Training without skip connections...")
## net is already initialized with module__use_skips=False and max_epochs=50
#net.fit(X, y)
#
#
## ==========================================
## STEP 2 PHASE A: Warmup the Decoder
## ==========================================
#print("Starting Step 2 Phase A: Warming up decoder...")
#
## 1. Turn skips ON and increase total epochs (e.g., 50 + 10 warmup epochs = 60)
#net.set_params(module__use_skips=True, max_epochs=60)
#
## 2. Access the underlying PyTorch module to freeze Encoder and Transformer
#for param in net.module_.encoder.parameters():
#    param.requires_grad = False
#for param in net.module_.transformer.parameters():
#    param.requires_grad = False
#
## Ensure Decoder is unfrozen
#for param in net.module_.decoder.parameters():
#    param.requires_grad = True
#
## 3. Clear AdamW momentum buffers by re-initializing the optimizer
#net.initialize_optimizer()
#
## 4. Train the warmup phase
#net.fit(X, y)
#
#
## ==========================================
## STEP 2 PHASE B: Full Fine-tuning
## ==========================================
#print("Starting Step 2 Phase B: Full Fine-tuning...")
#
## 1. Unfreeze everything
#for param in net.module_.parameters():
#    param.requires_grad = True
#
## 2. Lower learning rate and increase total epochs (e.g., 60 + 40 finetune epochs = 100)
#net.set_params(lr=1e-5, max_epochs=100)
#
## 3. Clear momentum buffers again for the new learning rate
#net.initialize_optimizer()
#
## 4. Final training run
#net.fit(X, y)