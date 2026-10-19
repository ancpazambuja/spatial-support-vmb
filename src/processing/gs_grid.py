

"""@package grdi.py
Grid with hyperparameters to be tryed.
"""


############################
# basic stuff
import os
import numpy as np
import ml_collections

############################
# ml stuff
import sklearn
from sklearn.model_selection import ParameterGrid
import torch
import torch.nn as nn

from skorch import dataset

############################
# transunet
from src.transunet.vit_seg_configs import get_attention_regression_config

############################
# internal modules

from src.utl.logging import Log as log
from src.utl.classes import Utils as utl

from src.experiment_tracking.mlflow_utils import MLFlowUtils, MLFlowDataAndModelsLogger

from src.apps.global_vars import GlobalVars

class GridSearchElementResult:
    """
    The task to be processed.
    """
    def __init__(self, index=-1, mean_train_score=None, std_train_score=None, mean_valid_score=None, std_valid_score=None, verbose_level=0):
        self.verbose_level = verbose_level
        self.verbose = True if verbose_level > 0 else False
        self.index = index
        self.mean_train_score = mean_train_score
        self.std_train_score = std_train_score
        self.mean_valid_score = mean_valid_score
        self.std_valid_score = std_valid_score

    def update_elem(self, r=None):
        updated = False
        if self.index == -1 or r.mean_valid_score < self.mean_valid_score:
            self.index = r.index
            self.mean_train_score = r.mean_train_score
            self.std_train_score = r.std_train_score
            self.mean_valid_score = r.mean_valid_score
            self.std_valid_score = r.std_valid_score
            updated = True
        return updated

    def update(self, r=None, rlist=None):
        assert r is not None or rlist is not None, f"One parameter has to be passed, r or rlist"
        assert r is None or rlist is None, f"Just one parameter must be passed."

        updated = False
        if r is not None:
            updated = self.update_elem(r)
        else:
            for e in rlist:
                u = self.update_elem(e)
                if u:
                    updated = True
        return updated
    

class Grid:
    def __init__(self, global_vars=None, verbose_level=0):
        self.verbose_level = verbose_level
        self.verbose = (verbose_level > 0)
        if global_vars is not None:
            self.grid = self.get_grid(global_vars=global_vars)

    def expand(self):
        param_grid = ParameterGrid(self.grid)
        # Make each entry suitable to scilearn.GridSearchCV
        result = list()
        for i in range(0,len(param_grid)):
            instance = param_grid[i]
            d = dict()
            for k,v in instance.items():
                d[k] = [v]
            result.append(d)
            del d
        return result

    def get_grid(self, global_vars=None):
        # model size
        model_data_size = utl.get_model_data_size(global_vars.get_raw_train_data())
        # convolutional filter
        conv_filter = global_vars.get_conv_filter_size()
        transposed_conv_filter = global_vars.get_transposed_conv_filter_size()
        conv_dilation = global_vars.get_conv_dilation_size()
        # size of the attention size
        if global_vars.get_data_type() == "shallow":
            attention_img_size=(11,28) #10x28=280, sequence length quando patches.size=(1,1)
        elif global_vars.get_data_type() == "deep": 
            attention_img_size=(21, 25) #21x25 = 525, sequence length quando patches.size=(1,1)
        else:
            raise ValueError
        # attention configuration
        att_conf_l2 =  get_attention_regression_config(num_layers=2,  hidden_size=768, num_heads=12, mlp_dim=3072 )
        if global_vars.is_transunet():
            # <<< NOTE >>> For this moment in_channels and hidden_layer must be the same
            # <<< NOTE 2 >>> The parameter hidden_layer below is not been used. The one which is is the module__attention_in_channels.
            #att_conf_l8 =  get_attention_regression_config(num_layers=8,  hidden_size=768, num_heads=12, mlp_dim=3072 )
            att_conf_l12 = get_attention_regression_config(num_layers=12, hidden_size=768, num_heads=12, mlp_dim=3072 )
            att_conf_l24 = get_attention_regression_config(num_layers=24, hidden_size=768, num_heads=12, mlp_dim=3072 )
            #att_conf_list = [att_conf_l2, att_conf_l8, att_conf_l12, att_conf_l24]
            att_conf_list = [att_conf_l2, att_conf_l12, att_conf_l24]
        else:
            att_conf_list = [ att_conf_l2 ] # dummy
        # max epochs
        max_epochs_list = [ global_vars.get_max_epochs() ]

        grouped_param_grid = [
            ##{ 'lr': [0.0001], 'optimizer__weight_decay' : [ 0.0001] }, # 0 means without L2 regularization
            #{ 'lr': [0.0001], 'optimizer__weight_decay' : [0, 0.0001, 0.0005] }, # 0 means without L2 regularization
            { 'lr': [0.0005], 'optimizer__weight_decay' : [0, 0.0005, 0.001] }, # 0 means without L2 regularization
            { 'lr': [0.001], 'optimizer__weight_decay' : [0, 0.001, 0.01] }, # 0 means without L2 regularization
            { 'lr': [0.01], 'optimizer__weight_decay' : [0, 0.01, 0.1] }, # 0 means without L2 regularization
            { 'lr': [0.0001], 'optimizer__weight_decay' : [0, 0.0005] }, # 0 means without L2 regularization
            #{ 'lr': [0.001], 'optimizer__weight_decay' : [0,  0.01] }, # 0 means without L2 regularization
            #{ 'lr': [0.01], 'optimizer__weight_decay' : [0, 0.1] }, # 0 means without L2 regularization
        ]

        # define the grid search parameters
        param_grid_base = {
            'max_epochs': max_epochs_list,
            'batch_size': [16],
            'device': ['cuda'],
            #'optimizer__momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
            #'optimizer__momentum': [0.0], #, 0.2, 0.4, 0.6, 0.8, 0.9],
            'optimizer': [torch.optim.Adam], #, SGD], #, RMSprop, Adagrad, Adadelta, Adam, Adamax, NAdam],
            'iterator_train__shuffle' : [True], # shuffle train and validation batches
            'train_split' : [None], #skorch.dataset.ValidSplit(5) ], 
            'criterion': [nn.MSELoss], #, nn.SSIM()],
            'verbose' : [ True ],
            #'callbacks': [None],
            'module__unembed_type' : [global_vars.get_unembed_type()],
            'module__use_transformers' : [global_vars.is_transunet()],
            'module__in_channels' : [model_data_size['nchannels']],
            'module__out_channels' : [1],
            'module__n_filters_init' : [16],
            'module__conv_kernel_size' : [conv_filter], # (5,5)],# (9,9), (13,13)],
            'module__conv_dilation_size' : [conv_dilation], 
            'module__transposed_conv_kernel_size' : [transposed_conv_filter], 
            'module__pool_kernel_size' : [(2,2)],
            'module__pool_stride' : [(2,2)],
            'module__batch_normalization' : [True],
            'module__dropout' : [0.2],
            'module__attention_img_size' : [attention_img_size],
            #'module__attention_config' : [att_conf_l2, att_conf_l8, att_conf_l12, att_conf_l24],
            'module__attention_config' : att_conf_list,
            'module__attention_in_channels' : [512],
            'module__vis' : [False],
            'module__verbose' : [ True ],
        }
        
        param_grid = list()
        for pg in grouped_param_grid:
            param_grid.append(pg|param_grid_base)

        return param_grid


class GSGridAdamW(Grid):
    def __init__(self, global_vars=None, verbose_level=0):
        super().__init__(verbose_level=verbose_level)
        self.grid = self.get_grid(global_vars=global_vars)

    def get_grid(self, global_vars=None):
        # model size
        model_data_size = utl.get_model_data_size(global_vars.get_raw_train_data())
        # convolutional filter
        conv_filter = global_vars.get_conv_filter_size()
        transposed_conv_filter = global_vars.get_transposed_conv_filter_size()
        conv_dilation = global_vars.get_conv_dilation_size()
        # size of the attention size
        if global_vars.get_data_type() == "shallow":
            attention_img_size=(11,28) #10x28=280, sequence length quando patches.size=(1,1)
        elif global_vars.get_data_type() == "deep": 
            attention_img_size=(21, 25) #21x25 = 525, sequence length quando patches.size=(1,1)
        else:
            raise ValueError
        # attention configuration
        att_conf_l2 =  get_attention_regression_config(num_layers=2,  hidden_size=768, num_heads=12, mlp_dim=3072 )
        if global_vars.is_transunet():
            # <<< NOTE >>> For this moment in_channels and hidden_layer must be the same
            # <<< NOTE 2 >>> The parameter hidden_layer below is not been used. The one which is is the module__attention_in_channels.
            #att_conf_l8 =  get_attention_regression_config(num_layers=8,  hidden_size=768, num_heads=12, mlp_dim=3072 )
            att_conf_l12 = get_attention_regression_config(num_layers=12, hidden_size=768, num_heads=12, mlp_dim=3072 )
            att_conf_l24 = get_attention_regression_config(num_layers=24, hidden_size=768, num_heads=12, mlp_dim=3072 )
            #att_conf_list = [att_conf_l2, att_conf_l8, att_conf_l12, att_conf_l24]
            att_conf_list = [att_conf_l2, att_conf_l12]
        else:
            att_conf_list = [ att_conf_l2 ] # dummy
        # max epochs
        max_epochs_list = [ global_vars.get_max_epochs() ]

        grouped_param_grid = [
            ##{ 'lr': [0.0001], 'optimizer__weight_decay' : [ 0.0001] }, # 0 means without L2 regularization
            #{ 'lr': [0.0001], 'optimizer__weight_decay' : [0, 0.0001, 0.0005] }, # 0 means without L2 regularization

            { 'lr': [0.0005], 'optimizer__weight_decay' : [0.0005, 0.001] }, # 0 means without L2 regularization
            { 'lr': [0.001], 'optimizer__weight_decay' : [0.0001, 0.0005, 0.00005] }, # 0 means without L2 regularization
            { 'lr': [0.01], 'optimizer__weight_decay' : [0.001, 0.01] }, # 0 means without L2 regularization
            { 'lr': [0.0001], 'optimizer__weight_decay' : [0.0005] }, # 0 means without L2 regularization


            #{ 'lr': [0.001], 'optimizer__weight_decay' : [0,  0.01] }, # 0 means without L2 regularization
            #{ 'lr': [0.01], 'optimizer__weight_decay' : [0, 0.1] }, # 0 means without L2 regularization
        ]

        # define the grid search parameters
        param_grid_base = {
            'max_epochs': max_epochs_list,
            'batch_size': [16],
            'device': ['cuda'],
            'optimizer': [torch.optim.AdamW], #, SGD], #, RMSprop, Adagrad, Adadelta, Adam, Adamax, NAdam],
            #'optimizer__momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
            #'optimizer__momentum': [0.0], #, 0.2, 0.4, 0.6, 0.8, 0.9],
            'callbacks__grad_clip__clip_value': [None, 1.0],
            #'callbacks__grad_clip__print_every': [5],
            'iterator_train__shuffle' : [True], # shuffle train and validation batches
            'train_split' : [None], #skorch.dataset.ValidSplit(5) ], 
            'criterion': [nn.MSELoss], #, nn.SSIM()],
            'verbose' : [ True ],
            #'callbacks': [None],
            'module__unembed_type' : [global_vars.get_unembed_type()],
            'module__use_transformers' : [global_vars.is_transunet()],
            'module__in_channels' : [model_data_size['nchannels']],
            'module__out_channels' : [1],
            'module__n_filters_init' : [16],
            'module__conv_kernel_size' : [conv_filter], # (5,5)],# (9,9), (13,13)],
            'module__conv_dilation_size' : [conv_dilation], 
            'module__transposed_conv_kernel_size' : [transposed_conv_filter], 
            'module__pool_kernel_size' : [(2,2)],
            'module__pool_stride' : [(2,2)],
            'module__batch_normalization' : [True],
            'module__dropout' : [0.2],
            'module__attention_img_size' : [attention_img_size],
            #'module__attention_config' : [att_conf_l2, att_conf_l8, att_conf_l12, att_conf_l24],
            'module__attention_config' : att_conf_list,
            'module__attention_in_channels' : [512],
            'module__vis' : [False],
            'module__verbose' : [ True ],
        }
        
        param_grid = list()
        for pg in grouped_param_grid:
            param_grid.append(pg|param_grid_base)

        return param_grid




class RefitGrid(Grid):
    def __init__(self, global_vars=None, verbose_level=0):
        super().__init__(verbose_level=verbose_level)
        self.grid = self.get_grid(global_vars=global_vars)

    def get_grid(self, global_vars=None):

        # model size
        model_data_size = utl.get_model_data_size(global_vars.get_raw_train_data())
        if global_vars.get_data_type() == "shallow":
            attention_img_size=(11,28) #11x28=308, sequence length quando patches.size=(1,1)
        elif global_vars.get_data_type() == "deep": 
            attention_img_size=(21, 25) #21x25 = 525, sequence length quando patches.size=(1,1)
        else:
            raise ValueError

        # define common set of hyper-parameters
        param_grid_base = {
            # RETIRAR ESSAS DUAS LINHAS  E DESCOMENTAR AS OUTRAS DUAS - teste do Adamw
            #'optimizer': torch.optim.AdamW, #, SGD], #, RMSprop, Adagrad, Adadelta, Adam, Adamax, NAdam],
            #'callbacks__grad_clip__clip_value': 1.0,
            #'callbacks__grad_clip__print_every': 5,
            'optimizer': torch.optim.Adam, #, SGD], #, RMSprop, Adagrad, Adadelta, Adam, Adamax, NAdam],
            'callbacks__grad_clip__clip_value': None,
            'max_epochs': global_vars.get_max_epochs(),
            'iterator_train__shuffle' : True, # shuffle train and validation batches
            'train_split' : dataset.ValidSplit(5), 
            'criterion': nn.MSELoss, #, nn.SSIM()],
            'verbose' : True,
            #'callbacks': [None],
            'module__unembed_type' : global_vars.get_unembed_type(),
            'module__use_transformers' : global_vars.is_transunet(),
            'module__in_channels' : model_data_size['nchannels'],
            'module__out_channels' : 1,
            'module__n_filters_init' : 16,
            'module__pool_kernel_size' : (2,2),
            'module__pool_stride' : (2,2),
            'module__batch_normalization' : True,
            'module__dropout' : 0.2,
            'module__attention_img_size' : attention_img_size,
            'module__attention_in_channels' : 512,
            'module__vis' : False,
            'module__verbose' : True,
        }

        _, run = MLFlowUtils.create_client_and_get_run(
            uri=global_vars.get_experiment_tracking_uri(), 
            ## Note: getting the best run id from a previous grid search
            experiment_name=global_vars.get_best_run_experiment_tracking_name(),
            experiment_id=global_vars.get_best_run_experiment_tracking_id(),
            execution_id=global_vars.get_best_run_id()
        )

        run_data_dict = run.data.to_dictionary()

        assert run_data_dict['params'] is not None, f"Error! There is no 'params' attribute associated to the best run user parameter!"
        assert len(run_data_dict['params'].keys()) > 0, f"Error! run_data_dict['params'] has zero attributes!"

        #log.info(f"run_data_dict['params'] = {run_data_dict['params']}")

        batch_size = eval(run_data_dict['params']['batch_size'])

        lr = eval(run_data_dict['params']['lr'])
        #lr = 0.001

        #optimizer__weight_decay = 0.002
        optimizer__weight_decay = eval(run_data_dict['params']['optimizer__weight_decay'])

        module__conv_kernel_size = eval(run_data_dict['params']['module__conv_kernel_size'])
        if global_vars.get_conv_filter_size() == (1,1): # exception to include 1x1 kernel size for simple test
            module__conv_kernel_size = global_vars.get_conv_filter_size()
        else:
            assert module__conv_kernel_size == global_vars.get_conv_filter_size(), f"Global_vars.conv_filter_size differs from best_run.conv_kernel_size. best_run.conv_kernel_size = { module__conv_kernel_size}"

        if global_vars.get_transposed_conv_filter_size() is None:
            if "module__transposed_conv_kernel_size" in run_data_dict['params']: 
                module__transposed_conv_kernel_size = eval(run_data_dict['params']['module__transposed_conv_kernel_size'])
            else:
                module__transposed_conv_kernel_size = (2,2)
        else:
            if global_vars.get_transposed_conv_filter_size()[0] == -1:
                module__transposed_conv_kernel_size = tuple(x + 1 for x in module__conv_kernel_size)
            else:
                module__transposed_conv_kernel_size = global_vars.get_transposed_conv_filter_size()

        print(f"MODULE_TRANSPOSED_CONV_KERNEL_SIZE = {module__transposed_conv_kernel_size}")

        if "module__conv_dilation_size" in run_data_dict['params']: 
            module__conv_dilation_size = eval(run_data_dict['params']['module__conv_dilation_size'])
            assert module__conv_dilation_size == global_vars.get_conv_dilation_size(), f"Global_vars.conv_dilation_size differs from best_run.conv_dilation_size. best_run.conv_kernel_size = { module__conv_kernel_size}"
        else:
            module__conv_dilation_size = global_vars.get_conv_dilation_size()


        #module__attention_config = run_data_dict['params']['module__attention_config']
        #log.info(f" 1 type of  best_run.module_attention_config = { type(module__attention_config)}")
        #log.info(f" 1 best_run.module_attention_config = { module__attention_config}")
        #module__attention_config = MLFlowUtils.string_to_dict(module__attention_config)
        #log.info(f" 2 type of  best_run.module_attention_config = { type(module__attention_config)}")
        #log.info(f" 2 best_run.module_attention_config = { module__attention_config}")
        
        module__attention_config = ml_collections.ConfigDict(MLFlowUtils.string_to_dict(run_data_dict['params']['module__attention_config']))
        module__attention_config.patches = ml_collections.ConfigDict({'size': (1, 1)})

        if global_vars.get_att_num_layers() is not None and global_vars.get_att_num_layers() != -1:
            module__attention_config["transformer"]["num_layers"] = global_vars.get_att_num_layers()


        pg ={
            'batch_size': batch_size,
            'lr': lr, 
            'optimizer__weight_decay' : optimizer__weight_decay,
            'module__conv_kernel_size' :  module__conv_kernel_size,
            'module__conv_dilation_size' :  module__conv_dilation_size,
            'module__transposed_conv_kernel_size' :  module__transposed_conv_kernel_size,
            'module__attention_config' :  module__attention_config,
        }
        param_grid = pg|param_grid_base

        return param_grid

    def expand(self):
        return [self.grid]





class GrokkingGrid(RefitGrid):
    def __init__(self, global_vars=None, verbose_level=0):
        super().__init__(global_vars=global_vars, verbose_level=verbose_level)

    def get_grid(self, global_vars=None):
        grokking_param_grid = super().get_grid(global_vars=global_vars)
        grokking_param_grid['optimizer'] = torch.optim.AdamW
        grokking_param_grid['optimizer__weight_decay'] = 0.01
        grokking_param_grid['max_epochs'] = 10000000000  #ignoring max_epochs from command line
        return grokking_param_grid