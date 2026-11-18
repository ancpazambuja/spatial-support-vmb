
"""@package inference_data.py
Contains classes related to the data needed for Building process of data and models from persisted files.

STATUS: WORK IN PROGRESS / TODO
-------------------------------
This module is an initiated but INCOMPLETE draft. It is NOT part of the
reproducible pipeline of the paper yet.

Intended purpose:
    `DeepTomographyFlowDatasets` will hold the data of the complete
    deep-tomography inference, covering the three iterations of the pipeline
    (RTM migration + neural network velocity-update inference per iteration).

Known gaps:
    - `get_input_it_1()`, `get_output_it_1()` and the remaining iteration
      accessors are not implemented (Ellipsis placeholders).

Do not rely on this module until this notice is removed.
"""
import numpy as np
from  glob import glob

from src.utl.logging import Log as log
from src.utl.serialization import Load, NPLoad
from src.inference.inference_dirs import InferenceDirs

class InferenceInputDatasets:
    def __init__(self, seismic_data=None, input_velocity_model=None, ground_truth=None, verbose=False):
        self.verbose = verbose
        self.seismic_data = seismic_data
        self.input_velocity_model = input_velocity_model
        self.ground_truth = ground_truth
    
    def get_input_seismic(self):
        return self.seismic_data
    
    def get_input_velocity_model(self):
        return self.input_velocity_model

    def get_ground_truth(self):
        return self.ground_truth
    

class DeepTomographyFlowDatasets:
    def __init__(self, seismic_data = None, input_velocity_model = None, 
                 train_indexes = None, test_indexes = None, verbose=False):
        self.verbose = verbose
        self.seismic_data = seismic_data
        self.input_velocity_model = input_velocity_model
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes
    
    def get_input_seismic(self):
        return self.seismic_data
    
    def get_input_it_1(self):
        return ...  # TODO: not implemented yet
    
    def get_output_it_1(self):
        return ...  # TODO: not implemented yet
    
    def get_input_it_2(self):
        return self.train_indexes
    




    




