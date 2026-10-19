"""@packagne inference_dirs.py

**Project Directory Definition**

Here, the definition of the directory structure for the run inference.

The specific directory structure must already exist in the filesystem having all the input files inside it.

The structure contains places for pre-trained ML models, input data and output data.
"""

import subprocess

import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
from glob import glob
from datetime import datetime

#!pip install matplotlib --upgrade
import segyio

from src.utl.logging import Log as log
from src.utl.classes import ShellCommand

from src.inference.rtm import RTMExe

class DT2_DEFS:
    RES_ORIGINAL = 0
    RES_HIGH = 1
    RES_MED = 2
    RES_LOW = 3
    RES_MAX = 4

class InferenceDirs:
    '''
    root________ input _______ model
            |             |___ seismic
            |
            |____ output _____ model
            |             |___ temp
            |
            |____ nn
            |
            |____ ground_truth _____ model
            |
            |____ evaluation_results
    '''
    def __init__(self, proj_dirname=None, proj_name=None, n_models=1, nz=351, nx=901, 
                 nn_weights_name_low_res=None, nn_weights_name_med_res=None, nn_weights_name_high_res=None, 
                 verbose=True):
        self.verbose=verbose
        self.proj_dirname = proj_dirname # the root of the project
        self.proj_name = proj_name 
        self.n_models = n_models

        self.input_dir = "input/"
        self.output_dir = "output/"
        self.nn_dir = "nn/"
        self.ground_truth_dir = "ground_truth/"
        self.results_dir = "evaluation_results/"

        self.seismic_dirname = "seismic"
        self.model_dirname = "model"

        # filenames of the nn model's saved weights
        self.nn_weights_name_low_res = nn_weights_name_low_res
        self.nn_weights_name_med_res = nn_weights_name_med_res
        self.nn_weights_name_high_res = nn_weights_name_high_res

        # files extensions
        self.weights_ext = "."
        self.seismic_in_ext = ".sum"
        self.in_ext = ".npy"
        self.out_ext = ".npy"
        self.results_ext = ".png"

        self.nn_weights_prefix = "weights_" # nn inference output
        self.seismic_input_prefix = "tiros"  # real seismic data

        self.results_prefix = "eval" 

        dim = str(nz)+"x"+str(nx)

        # marmousi_rtm_351_901_s1_1.npy - must add resolution and model id.
        self.migration_prefix = f"rtm_{dim}_s"  # rtm command line output
        # marmousi_model_351_901_s1_1.npy - must add resolution and model id.
        self.inference_model_prefix = f"model_{dim}_s" # nn inference output
        # marmousi_dv_351_901_s1_1.npy - must add resolution and model id.
        self.inference_dv_prefix = f"dv_{dim}_s" # nn inference output

        if self.verbose: 
            self.__print_filenames()

    def __print_filenames(self):
        log.info(f"INPUT FILENAMES:")
        for i in range(0,self.n_models):
            log.info(f"[ {i} ]")
            log.info(f"SEISMIC:        {self.get_input_seismic_filename(data_idx=i)}")
            log.info(f"MODEL:          {self.get_input_model_filename(data_idx=i)}")
            log.info(f"GROUND TRUTH:   {self.get_ground_truth_model_filename(data_idx=i)}")
            log.info(f"EVAL MSE:       {self.get_evaluation_mse_filename(data_idx=i)}")
            log.info(f"EVAL MAE:       {self.get_evaluation_mae_filename(data_idx=i)}")
            log.info(f"EVAL SSIM:      {self.get_evaluation_ssim_filename(data_idx=i)}")

        log.info(f"-----------------------------------------------------------------")

        log.info(f"OUTPUT FILENAMES:")
        for m in range(0,self.n_models):
            for i in range(0,DT2_DEFS.RES_MAX):
                mig = self.get_output_migration_filename(res_idx=i, data_idx=m)
                model = self.get_output_inference_model_filename(res_idx=i, data_idx=m)
                dv = self.get_output_inference_dv_filename(res_idx=i, data_idx=m)
                log.info(f"[ {i} ]")
                log.info(f"RTM:               {mig}")
                log.info(f"INFERENCE - MODEL: {model}")
                log.info(f"INFERENCE - DV:    {dv}")
    
    def __make_evaluation_filename(self, metric, data_idx):
        sep = "_"
        dir = self.get_evaluation_results_dir()
        filename = self.proj_name + sep + self.results_prefix + sep + metric + sep + str(data_idx+1) + self.results_ext
        return dir + filename

    def __make_ground_truth_model_filename(self, prefix, artifact_name, data_idx):
        sep = "_"
        dir = self.get_ground_truth_dir() + artifact_name
        filename = self.proj_name + sep + prefix + "0" + sep + str(data_idx+1) + self.out_ext
        return dir + "/" + filename

    def __make_input_seismic_filename(self, prefix, artifact_name, data_idx):
        sep = "_"
        dir = self.get_input_dir() + artifact_name
        filename = self.proj_name + sep + prefix + sep + str(data_idx) + self.seismic_in_ext
        return dir + "/" + filename

    def __make_input_model_filename(self, prefix, artifact_name, res_idx, data_idx):
        sep = "_"
        dir = self.get_input_dir() + artifact_name
        filename = self.proj_name + sep + prefix + str(res_idx) + sep + str(data_idx) + self.out_ext
        return dir + "/" + filename

    def __make_out_filename(self, prefix, artifact_name, res_idx, data_idx):
        sep = "_"
        dir = self.get_output_dir() + artifact_name
        filename = self.proj_name + sep + prefix + str(res_idx) + sep + str(data_idx+1) + self.out_ext
        return dir + "/" + filename

    def __make_out_inference_model_filename(self, res_idx, data_idx): 
        return self.__make_out_filename(self.inference_model_prefix, self.model_dirname, res_idx, data_idx)

    def __make_out_inference_dv_filename(self, res_idx, data_idx):
        return self.__make_out_filename(self.inference_dv_prefix, self.model_dirname, res_idx, data_idx)

    def __make_out_mig_filename(self, res_idx, data_idx):
        return self.__make_out_filename(self.migration_prefix, self.seismic_dirname, res_idx, data_idx)

    def __make_index(self, res_idx, data_idx):
        assert data_idx < self.n_models, f"data_idx (={data_idx}) must be less than {self.n_models}"
        assert res_idx < DT2_DEFS.RES_MAX, f"res_idx (={res_idx}) must be less than {DT2_DEFS.RES_MAX}"
        return data_idx * DT2_DEFS.RES_MAX + res_idx

    def get_input_seismic_filename(self, data_idx=0):
        return self.__make_input_seismic_filename( self.seismic_input_prefix, self.seismic_dirname, data_idx+1)

    def get_input_model_filename(self, res_idx=0, data_idx=0):
        return self.__make_input_model_filename( self.inference_model_prefix, self.model_dirname, res_idx, data_idx+1)

    def get_evaluation_mse_filename(self, data_idx=0):
        return self.__make_evaluation_filename( "mse", data_idx)

    def get_evaluation_mae_filename(self, data_idx=0):
        return self.__make_evaluation_filename( "mae", data_idx)

    def get_evaluation_ssim_filename(self, data_idx=0):
        return self.__make_evaluation_filename( "ssim", data_idx)

    def get_ground_truth_model_filename(self, data_idx=0):
        return self.__make_ground_truth_model_filename( self.inference_model_prefix, self.model_dirname, data_idx)

    def get_output_migration_filename(self, res_idx=0, data_idx=0):
        return self.__make_out_mig_filename(res_idx, data_idx)

    def get_output_inference_model_filename(self, res_idx=0, data_idx=0):
        return self.__make_out_inference_model_filename(res_idx, data_idx) 

    def get_output_inference_dv_filename(self, res_idx=0, data_idx=0):
        return self.__make_out_inference_dv_filename(res_idx, data_idx)

    def get_nmb_of_datasets(self):
        return self.n_models

    def get_nmb_of_resolutions(self):
        return DT2_DEFS.RES_MAX

    def get_proj_dirname(self):
        return self.proj_dirname

    def get_proj_name(self):
        return self.proj_name

    def get_ground_truth_dir(self):
        return self.get_proj_dirname() + self.ground_truth_dir

    def get_input_dir(self):
        return self.get_proj_dirname() + self.input_dir 

    def get_output_dir(self):
        return self.get_proj_dirname() + self.output_dir

    def get_evaluation_results_dir(self):
        return self.get_proj_dirname() + self.results_dir

    def get_input_model_dir(self):
        return self.get_input_dir() + self.model_dirname

    def get_input_seismic_dir(self):
        return self.get_input_dir() + self.seismic_dirname

    def get_nn_dir(self):
        return self.nn_dir

    def get_nn_weights_filename(self, iter=None):
        errorMsg = f"'iter' must be 1, 2 or 3. Actual value: {iter}"
        filename = None
        assert iter > 0 and iter < 4, errorMsg
        if iter == 3:
            filename = self.nn_weights_name_low_res
        elif iter == 2:
            filename = self.nn_weights_name_med_res
        elif iter == 1:
            filename = self.nn_weights_name_high_res
        else:
            raise ValueError(errorMsg)
        return filename

    def get_nn_weights_fullname(self, iter):
        return self.get_nn_dir() + self.get_nn_weights_filename(iter=iter)

