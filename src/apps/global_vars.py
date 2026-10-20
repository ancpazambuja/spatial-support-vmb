"""@package global_vars.py
Module with global variables.
"""

from pathlib import Path

import random
import numpy as np
import torch


from src.utl.logging import Log as log
from src.utl.classes import Base
from src.utl.classes import Utils as utl
from src.utl.classes import ProjectDirs, BuildDirectories
from src.inference.inference_dirs import InferenceDirs 

from src.data.data_builder import BuildDeepTomographyDataAndModels
from src.data.data import ModelIdx

from src.experiment_tracking.mlflow_utils import MLFlowUtils

#from src.utl.classes import Utils as utl
#from src.utl.classes import Graph as graph

class BaseGlobalVars(Base):
    """!
    Contains globals variables.
    """

    def __init__(
        self,
        verbose_level=None, # means no verbose
        mlflow_uri=None,
        mlflow_experiment=None,
        mlflow_experiment_id=None,
        sep="__",
        name='BaseGlobalVars'
    ):
        super().__init__(name=name)
        self.verbose_level = verbose_level
        self.verbose = (verbose_level > 0)
        self.mlflow_uri = mlflow_uri
        self.mlflow_experiment = mlflow_experiment
        self.mlflow_experiment_id = mlflow_experiment_id
        self.sep = sep
        self.name = name

    def is_verbose(self):
        return self.verbose

    def get_verbose_level(self):
        return self.verbose_level
    
    def get_experiment_tracking_uri(self):
        return self.mlflow_uri

    def get_experiment_tracking_name(self):
        return self.mlflow_experiment
    
    def get_experiment_tracking_id(self):
        return self.mlflow_experiment_id

    def set_experiment_tracking_id(self, id):
        self.mlflow_experiment_id = id
    

class LoadDataBaseGlobalVars(BaseGlobalVars):
    """!
    Contains globals variables.
    """


    def __init__(
        self,
        *args, 
        checkpoint_basedir=None,
        data="deep", #"shallow"
        data_resolution='high', #'med', 'low'
        model_basedir=None,   # evaluation-only
        seed=-1,
        data_max_samples=-1,
        test_perc=0.1,
        label_id=None,
        execution_id=None,
        **kwargs,
    ):
        kwargs['name'] = 'LoadDataBaseGlobalVars'
        super().__init__(*args, **kwargs)
#        super().__init__(name=name, verbose_level=verbose_level, mlflow_uri=mlflow_uri, 
#            mlflow_experiment=mlflow_experiment, mlflow_experiment_id=mlflow_experiment_id,
#            sep=sep
#            )

        self.checkpoint_basedir = checkpoint_basedir
        self.data = data
        self.data_basedir = None
        self.data_resolution = data_resolution
        self.data_max_samples = data_max_samples
        self.model_basedir = model_basedir
        self.seed = seed
        self.execution_id = execution_id
        self.execution_name = None
        self.label_id = label_id
        self.test_perc = test_perc
        self.timestamp = utl.get_timestamp() 
        self.restart = None

        if self.execution_id is not None:
            assert self.checkpoint_basedir is not None, f"If it is restart, attribute checkpoint_dir must be provided."
            self.restart = True
        else:
            self.restart = False

        if self.data == 'shallow':
            self.data_basedir = "./data/"
            models_mask=[False] * ModelIdx.MAX_MODEL_IDX
            models_mask[ ModelIdx.TRUE_MODEL_IDX] = True
            models_mask[ ModelIdx.HIGHRES_MODEL_IDX] = True
    # NEWER
        else:
            self.data_basedir = "./data/"
            models_mask=[True] * ModelIdx.MAX_MODEL_IDX

        ##############################
        # managing reproducibility 
        ##############################
        self.seed = None if seed < 0 else seed
        if self.seed is not None:
            utl.set_seed(self.seed)

        self.dirs = BuildDirectories(data=self.get_data_type(), basedir=self.get_data_basedir()).build()

        ##############################
        # create train and test data
        ##############################
        
        data_and_models_builder = BuildDeepTomographyDataAndModels(
            dirs=self.dirs, models_mask=models_mask, seed=self.seed,
            test_perc=self.test_perc, max_samples=self.data_max_samples, verbose=True)
        self.train_data, self.test_data, self.train_models, self.test_models, self.train_indexes, self.test_indexes = data_and_models_builder.build()

        self.newer_data = data_and_models_builder.is_newer_data()

        # setting data resolution
        self.train_data.set_resolution(self.data_resolution)
        if self.test_data is not None:
            self.test_data.set_resolution(self.data_resolution)

    def print(self):
        if (self.verbose):
            log.info(f"Project info: {self.dirs}")
            log.info(f"Is newer data? {self.is_newer_data}")
            log.info(f"Experiment tracking URI: {self.mlflow_uri}")
            log.info(f"Experiment tracking name: {self.mlflow_experiment}")
            log.info(f"Experiment tracking id: {self.mlflow_experiment_id}")
            log.info(f"Seed: {self.seed}")
            log.info(f"Max samples: {self.data_max_samples}")

    def is_newer_data(self):
        return self.newer_data
    
    def is_restart(self):
        return self.restart
    
    def get_data_type(self):
        return self.data
    
    def get_data_resolution(self):
        return self.data_resolution
    
    def get_data_max_samples(self):
        return self.data_max_samples
    
    def get_checkpoint_basedir(self):
        return self.checkpoint_basedir
    
    def get_data_basedir(self):
        return self.data_basedir
    
    def get_model_basedir(self):
        return self.model_basedir
    
    def get_dirs(self):
        return self.dirs
    
    def get_raw_train_data(self):
        return self.train_data

    def get_raw_train_models(self):
        return self.train_models

    def get_raw_test_data(self):
        return self.test_data

    def get_raw_test_models(self):
        return self.test_models

    def get_train_indexes(self):
        return self.train_indexes

    def get_test_indexes(self):
        return self.test_indexes
    
    def get_execution_id(self):
        return self.execution_id
    
    def set_execution_id(self, id):
        self.execution_id = id
    
    def get_execution_name(self):
        return self.execution_name
    
    def set_execution_name(self, name):
        self.execution_name = name
    
    def get_timestamp(self):
        return self.timestamp
    
    def get_test_perc(self):
        return self.test_perc
    
    def get_label_id(self):
        return self.label_id
    
    def get_model_arch(self):
        return self.model_arch
    
    def get_seed(self):
        return self.seed
    
    
class GlobalVars(LoadDataBaseGlobalVars):
    """!
    Contains globals variables.
    """


    def __init__(
        self,
        *args,
        unembed_type='paper',
        conv_filter_size=(13,13),
        conv_dilation_size=(1,1),
        transposed_conv_filter_size=(2,2),
        max_epochs=130,
        cv=5,
        model_arch='unet',
        device='cuda:all',
        **kwargs
    ):
        kwargs['name'] = 'GlobalVars'
        super().__init__(*args, **kwargs)
#        super().__init__(name=name, verbose_level=verbose_level, mlflow_uri=mlflow_uri, 
#            mlflow_experiment=mlflow_experiment, mlflow_experiment_id=mlflow_experiment_id,
#            sep=sep, checkpoint_basedir=checkpoint_basedir, data=data, data_resolution=data_resolution,
#            model_basedir=model_basedir, seed=seed, data_max_samples=data_max_samples,
#            test_perc=test_perc, label_id=label_id, execution_id=execution_id
#            )

        self.unembed_type = unembed_type
        self.conv_filter_size = conv_filter_size
        self.conv_dilation_size = conv_dilation_size
        self.transposed_conv_filter_size = transposed_conv_filter_size
        self.max_epochs = max_epochs
        self.cv = cv
        self.model_arch = model_arch
        self.device = device

    def get_conv_filter_size(self):
        return self.conv_filter_size

    def get_conv_dilation_size(self):
        return self.conv_dilation_size

    def get_transposed_conv_filter_size(self):
        return self.transposed_conv_filter_size

    def get_unembed_type(self):
        return self.unembed_type

    def get_max_epochs(self):
        return self.max_epochs

    def get_device(self):
        return self.device

    def get_execution_ids(self):
        return self.execution_id.split(',')
    
    def get_cv(self):
        return self.cv
    
    def get_model_arch(self):
        return self.model_arch
    
    def is_unet(self):
        return self.get_model_arch() == 'unet'

    def is_transunet(self):
        return self.get_model_arch() == 'transunet'

    def is_atrous_unet(self):
        return self.get_model_arch() == 'atrous_unet'

    def is_fourier_unet(self):
        return self.get_model_arch() == 'fourier_unet'

    
class RefitGlobalVars(GlobalVars):
    """!
    Implements the command line interface for refit -- final training -- and evaluation.
    """
    def __init__(self,
            *args, 
            model_path=None, refit=None, att_num_layers=None, gs_best_run_id=None, 
            gs_best_run_experiment_tracking_name=None, one_step_training=True, is_encdec_only=False,
            **kwargs
        ):
        kwargs['name'] = 'RefitGlobalVars'
        super().__init__(*args, **kwargs)
        self.refit = refit
        self.model_path = model_path
        self.att_num_layers = att_num_layers
        self.one_step_training = one_step_training
        self.is_encdec_only = is_encdec_only
        self.gs_best_run_id = gs_best_run_id
        self.gs_best_run_experiment_tracking_name = gs_best_run_experiment_tracking_name
        self.gs_best_run_experiment_tracking_id = None #MLFlowUtils.get_experiment_id_by_experiment_name(self.gs_best_run_experiment_tracking_name)
        assert self.refit is not None, f"refit parameter must be set!"
        assert self.model_path is not None, f"model_path parameter must be set!"

    def get_best_run_experiment_tracking_name(self):
        return self.gs_best_run_experiment_tracking_name
    
    def get_best_run_experiment_tracking_id(self):
        return self.gs_best_run_experiment_tracking_id
    
    def set_best_run_experiment_tracking_id(self, id):
        self.gs_best_run_experiment_tracking_id = id

    def get_best_run_id(self):
        return self.gs_best_run_id
    
    def get_model_path(self):
        return self.model_path
    
    def get_att_num_layers(self):
        return self.att_num_layers
    
    def is_refit(self):
        return self.refit

    def is_one_step_training(self):
        return self.one_step_training


class AddParametersInGSGlobalVars(BaseGlobalVars):
    """!
    Implements auxiliary funcionality.
    """
    def __init__(self,
            *args, 
            with_filter=False,
            data_resolution="high",
            **kwargs
        ):
        kwargs['name'] = 'AddParametersInGS'
        super().__init__(*args, **kwargs)
        self.filter = with_filter
        self.data_resolution = data_resolution

    def with_filter(self):
        return self.filter
    
    def get_data_resolution(self):
        return self.data_resolution


class AnalyseEvaluationsExamplesGlobalVars(LoadDataBaseGlobalVars):
    """!
    Implements auxiliary funcionality.
    """
    def __init__(self,
            *args, 
            mlflow_experiment_prefix=None,
            model_path=None,
            list_of_best_models=None,
            list_of_worst_models=None,
            list_of_hardest_examples=None,
            list_of_easiest_examples=None,
            **kwargs
        ):
        kwargs['name'] = 'AnalyseEvaluationsExamplesGlobalVars'
        print(kwargs)
        super().__init__(*args, **kwargs)
        self.mlflow_experiment_prefix = mlflow_experiment_prefix
        self.model_path = model_path
        self.list_of_best_models = [item.strip() for item in list_of_best_models.split(",")]
        self.list_of_worst_models = [item.strip() for item in list_of_worst_models.split(",")]
        self.list_of_hardest_examples = [item.strip() for item in list_of_hardest_examples.split(",")]
        self.list_of_easiest_examples = [item.strip() for item in list_of_easiest_examples.split(",")]

    def get_experiment_prefix(self):
        return self.mlflow_experiment_prefix

    def get_model_path(self):
        return self.model_path

    def get_best_models(self):
        return self.list_of_best_models

    def get_worst_models(self):
        return self.list_of_worst_models

    def get_hardest_examples(self):
        return self.list_of_hardest_examples

    def get_easiest_examples(self):
        return self.list_of_easiest_examples


class AnalyseEvaluationsGlobalVars(AddParametersInGSGlobalVars):
    """!
    Implements auxiliary funcionality.
    """
    def __init__(self,
            *args, 
            mlflow_experiment_prefix=None,
            **kwargs
        ):
        kwargs['name'] = 'AnalyseEvaluationsGlobalVars'
        super().__init__(*args, **kwargs)
        self.mlflow_experiment_prefix = mlflow_experiment_prefix

    def get_experiment_prefix(self):
        return self.mlflow_experiment_prefix

    
class QualityControlGlobalVars(BaseGlobalVars):
    """!
    Implements the command line interface for model quality control.
    """
    def __init__(self,
            *args, 
            model_path=None, run_id=None, 
            **kwargs
        ):
        kwargs['name'] = 'QualityControlGlobalVars'
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.execution_id = execution_id
        self.model_name = run_id # is used if run_id in None
        assert self.model_path is not None, f"model_path parameter must be set!"
        assert self.execution_id is not None or self.model_name is not None, f"model_path or run_id must be set!"

    def get_model_path(self):
        return self.model_path
    
    def get_execution_id(self):
        return self.execution_id

class StatisticalEvaluationGlobalVars(BaseGlobalVars):
    """!
    Implements the command line interface for the estatiscal evaluation (Friedman and )
    of the trained models (REFIT__).
    """
    def __init__(self,
            *args, 
            model_path=None, seed=None, run_ids_file=None, 
            **kwargs
        ):
        kwargs['name'] = 'QualityControlGlobalVars'
        super().__init__(*args, **kwargs)
        self.model_path = model_path 
        self.seed = seed #the used seed - CAN'T COMPARE MODELS FROM DIFFERENTS SEEDS
        self.run_ids_file = run_ids_file #file with models' run ids
        self.run_ids_list = None

        assert self.model_path is not None, f"model_path parameter must be set!"
        assert self.seed is not None, f"seed parameter must be set!"
        assert self.run_ids_file is not None, f"run_ids_file parameter must be set!"

        self.run_ids_list = self.read_run_ids(self.run_ids_file)

    def get_seed(self):
        return self.seed
    
    def get_run_ids_file(self):
        return self.run_ids_file
    
    def get_run_ids_list(self):
        return self.run_ids_list

    def get_model_path(self):
        return self.model_path

    def get_model_name(self):
        return self.model_name

    def get_model_fullname(self):
        return self.get_model_path() + "/" + self.get_model_name()
    
    

class DT2InferenceGlobalVars(BaseGlobalVars):
    """!
    Implements the command line interface for the DT2 inference complete flow.
    """
    def __init__(
        self,
        *args, 
#        inference_dir_name=None,
        inference_proj_name="marmousi", #"deep", "shallow", "marmousi"
        refit_run_id=None, 
        refit_run_experiment_tracking_name=None,
        label_id=None,
        weights_name_low_res=None,
        weights_name_med_res=None,
        weights_name_high_res=None,
        device='cuda:all',
        **kwargs
    ):
        kwargs['name'] = 'DT2InferenceGlobalVars'
        super().__init__(*args, **kwargs)

#        self.inference_dir_name = inference_dir_name
        self.inference_proj_name = inference_proj_name
        self.refit_run_id = refit_run_id
        self.refit_run_experiment_tracking_name = refit_run_experiment_tracking_name
        self.weights_name_low_res = weights_name_low_res,
        self.weights_name_med_res = weights_name_med_res,
        self.weights_name_high_res = weights_name_high_res,
        self.label_id = label_id
        self.device = device
        self.timestamp = utl.get_timestamp() 

        assert self.nchannels == 15, f"nchannels unexpected. Expected 15. Actual {self.nchannels})"

        # secundary attributes
        self.seed = None # must be read from refit_run
        self.refit_run_experiment_tracking_id = None ##MLFlowUtils.get_experiment_id_by_experiment_name(self.refit_run_experiment_tracking_name)
        self.data_basedir = None
        self.n_models = None
        self.dirs = None
        self.execution_id = None
        self.execution_name = None
        self.train_indexes = None
        self.test_indexes = None
        self.refit_run = None
        self.nn_weights = None
        self.input_datasets = None
        self.nx = None
        self.nz = None
        self.nchannels = 15


        if self.inference_proj_name == 'shallow':
            #self.inference_dir_name = "/gemini/tg/ntbs/doutorado/5semestre/estudo.orientado/deeptomograph_data/data/DadosRede"
            self.inference_dir_name = "/gemini/tg/ntbs/doutorado/6semestre.2025.1/dt2/inference/" + self.inference_proj_name + "/"
            self.nz = 351
            self.nx = 901
            self.n_models = 1000
        # NEWER
        elif self.inference_proj_name == 'deep':
            #self.inference_dir_name = "/gemini/tg/ntbs/doutorado/5semestre/estudo.orientado/deeptomograph_data/data_new_and_bigger/"
            self.inference_dir_name = "/gemini/tg/ntbs/doutorado/6semestre.2025.1/dt2/inference/" + self.inference_proj_name + "/"
            self.nz = 680
            self.nx = 800
            self.n_models = 1000
        # MARMOUSI  
        elif self.inference_proj_name == 'marmousi':
            self.inference_dir_name = "/gemini/tg/ntbs/doutorado/6semestre.2025.1/dt2/inference/" + self.inference_proj_name + "/"
            #self.inference_dir_name = "/gemini/tg/ntbs/doutorado/6semestre.2025.1/dt2/inference/marmousi/"
            self.nz = 351
            self.nx = 901
            self.n_models = 1
        else:
            raise ValueError(f"Attribute 'inference_dir_name' must assume the following values: deep, shallow and marmousi. Actual value: {self.data}")

        ##############################
        # Directory structure 
        ##############################
        self.dirs = InferenceDirs(proj_dirname=self.inference_dir_name, proj_name=self.inference_proj_name, 
            n_models=self.n_models, nz=self.nz, nx=self.nx, 
            nn_weights_name=self.model_name,
            verbose=False)

        filename = self.dirs.get_input_model_filename() 
        assert Path(filename).exists(), f"File does not exist! <{filename}>"


        ##############################
        # managing reproducibility and 
        # avoidind data leakage
        ##############################

        # getting refit_run 
        self.refit_run = "a fazer"#..... 
        self.seed = 0# get from refit_run
        utl.set_seed(self.seed)
        self.train_indexes = ...
        self.test_indexes = ...


        ##############################
        # building datasets
        ##############################
        self.input_datasets = InferenceInputDatasetsBuilder(
            dirs=self.dirs, 
            nx=self.nx, nz=self.nz, nchannels=self.nchannels, 
            verbose=self.verbose
        ).build()

        self.nn_weights = InferenceNNWeightsBuilder(filenames=build_nn_weights_filenames()).build()

    def build_nn_weights_filenames(self):
        filenames = list()
        filenames.append(self.dirs.get_nn_weights_fullname(iter=3))
        filenames.append(self.dirs.get_nn_weights_fullname(iter=2))
        filenames.append(self.dirs.get_nn_weights_fullname(iter=1))
        return filenames

    def print(self):
        if (self.verbose):
            log.info(f"Project info: {self.dirs}")
            log.info(f"Experiment tracking URI: {self.mlflow_uri}")
            log.info(f"Experiment tracking name: {self.mlflow_experiment}")
            log.info(f"Experiment tracking id: {self.mlflow_experiment_id}")
            log.info(f"Seed: {self.seed}")

    def get_inference_proj_name(self):
        return self.inference_proj_name
    
    def get_inference_dir_name(self):
        return self.inference_dir_name
    
    def get_model_name(self):
        return self.model_name
    
    def get_dirs(self):
        return self.dirs
    
    def get_device(self):
        return self.device

    def get_execution_id(self):
        return self.execution_id
    
    def get_execution_ids(self):
        return self.execution_id.split(',')
    
    def set_execution_id(self, id):
        self.execution_id = id
    
    def get_execution_name(self):
        return self.execution_name
    
    def set_execution_name(self, name):
        self.execution_name = name
    
    def get_timestamp(self):
        return self.timestamp
    
    def get_label_id(self):
        return self.label_id
    
    def get_seed(self):
        return self.seed
    
    def get_model_name(self):
        return self.model_name

    def get_execution_id(self):
        return self.execution_id
    
    def set_execution_id(self, id):
        self.execution_id = id
    
    def get_execution_name(self):
        return self.execution_name
    
    def set_execution_name(self, name):
        self.execution_name = name

    def get_input_datasets(self):
        return self.input_datasets

    def get_nn_weights(self):
        return self.nn_weights


class AddTrainTestIndexesInGlobalVars(BaseGlobalVars):
    """!
    Implements auxiliary funcionality.
    """
    def __init__(self,
            *args, 
            type='gs', # 'gs' or 'refit'
            **kwargs
        ):
        kwargs['name'] = 'AddParametersInGS'
        super().__init__(*args, **kwargs)
        self.type = type

    def get_type(self):
        return self.type







    


