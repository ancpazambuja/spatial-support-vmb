
"""@package local_data_builder.py
Building process of data and models from persisted files.
"""
import numpy as np
from  glob import glob

from src.utl.logging import Log as log
from src.utl.classes import ProjectDirs
from src.utl.serialization import Load, NPLoad
from src.inference.inference_dirs import InferenceDirs
from src.inference.data.inference_data import InferenceInputDatasets


class InferenceInputDatasetsBuilder:
    def __init__(self, dirs:InferenceDirs=None, nx=None, nz=None, nchannels=None, verbose=False):
        self.dirs = dirs
        self.nx = nx
        self.nz = nz
        self.nchannels = nchannels
        self.verbose = verbose

    def builder(self):
        #read seismic
        seismic_filename = self.dirs.get_input_seismic_filename()
        input_seismic = self.__read_migrated_data_from_file(
                            filename=seismic_filename, 
                            nchannels=self.nchannels, 
                            nx=self.nx, 
                            nz=self.nz, 
                            transpose=True
                        )

        #read velocity
        velmodel_filename = self.dirs.get_input_model_filename()
        input_velocity = self.__read_velocity_data_from_file(
                            filename=velmodel_filename, 
                            nchannels=self.nchannels, 
                            nx=self.nx, 
                            nz=self.nz, 
                            transpose=True
                        )
        
        #read ground_truth
        ground_truth_filename = self.dirs.get_input_model_filename()
        ground_truth = self.__read_velocity_data_from_file(
                            filename=ground_truth_filename, 
                            nchannels=self.nchannels, 
                            nx=self.nx, 
                            nz=self.nz, 
                            transpose=True
                        )
        # input_seismic = self.__read_migrated_data_from_file()elocity model
        return InferenceInputDatasets(
            seismic_data=input_seismic, 
            input_velocity_model=input_velocity, 
            ground_truth=ground_truth, 
            verbose=self.verbose)
    
    #read the velocity model 
    def __read_velocity_data_from_file(filename=None, transpose=True):
        # Load binary file
        v = np.load(filename)
        #print(f"Shape from stream of bytes: {v.shape}")
        #v = v.reshape((m.get_nchannels(), m.get_nx(), m.get_nz())) # nmodels, nx, nz, nchannels
        print(f"Original shape: {v.shape}")
        if transpose:
            v = np.transpose(v, axes=(2,0,1))
            print(f"Transposed shape: {v.shape}") 
        return v
    
    def __read_migrated_data_from_file(filename=None, nchannels=None, nx=None, nz=None, transpose=True):
        # Load binary file
        with open(filename, 'rb') as f:
            v = np.fromfile(f, dtype=np.float32)
            print(f"Shape from stream of bytes: {v.shape}")
            v = v.reshape((nchannels, nx, nz))
            print(f"Original shape: {v.shape}")
            if transpose:
                v = np.transpose(v, axes=(0,2,1))
                print(f"Transposed shape: {v.shape}") 
            return v
    


class InferenceNNWeightsBuilder:
    def __init__(self, dirs:InferenceDirs=None, mlflow_run=None, nx=None, nz=None, nchannels=None, verbose=False):
        self.dirs = dirs
        self.mlflow_run = mlflow_run
        self.nx = nx
        self.nz = nz
        self.nchannels = nchannels
        self.verbose = verbose

    def builder(self):
        #read seismic
        seismic_filename = self.dirs.get_input_seismic_filename()
        input_seismic = self.__read_migrated_data_from_file(
                            filename=seismic_filename, 
                            nchannels=self.nchannels, 
                            nx=self.nx, 
                            nz=self.nz, 
                            transpose=True
                        )

        #read velocity
        velmodel_filename = self.dirs.get_input_model_filename()
        input_seismic = self.__read_velocity_data_from_file(
                            filename=velmodel_filename, 
                            nchannels=self.nchannels, 
                            nx=self.nx, 
                            nz=self.nz, 
                            transpose=True
                        )
        # get train and test indexes from run
        train_indexes = self.mlflow_run...
        test_indexes = self.mlflow_run...

        # input_seismic = self.__read_migrated_data_from_file()elocity model
        return InferenceInputDatasets(seismic_data=input_seismic, input_velocity_model=velmodel_filename, 
                 train_indexes = train_indexes, test_indexes = test_indexes, verbose=self.verbose)
    
    #read the velocity model 
    def __read_velocity_data_from_file(filename=None, transpose=True):
        # Load binary file
        v = np.load(filename)
        #print(f"Shape from stream of bytes: {v.shape}")
        #v = v.reshape((m.get_nchannels(), m.get_nx(), m.get_nz())) # nmodels, nx, nz, nchannels
        print(f"Original shape: {v.shape}")
        if transpose:
            v = np.transpose(v, axes=(2,0,1))
            print(f"Transposed shape: {v.shape}") 
        return v
    
    def __read_migrated_data_from_file(filename=None, nchannels=None, nx=None, nz=None, transpose=True):
        # Load binary file
        with open(filename, 'rb') as f:
            v = np.fromfile(f, dtype=np.float32)
            print(f"Shape from stream of bytes: {v.shape}")
            v = v.reshape((nchannels, nx, nz))
            print(f"Original shape: {v.shape}")
            if transpose:
                v = np.transpose(v, axes=(0,2,1))
                print(f"Transposed shape: {v.shape}") 
            return v
    






