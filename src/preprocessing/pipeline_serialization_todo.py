"""@package local_pipeline_serialization.py
This module contains routines to serialize the pipeline.

"""

import joblib

from src.preprocessing.local_pipeline import PreprocessingPipe
from src.local_logging import Log as log


"""@name XXX
@{
"""


class PipelineSerializationBase:
    PIPE_NAME = "pipeline_preprocess.pkl"
    ONEHOT_PIPE_NAME = "pipeline_onehot_preprocess.pkl"

    def __init__(self, pipe: PreprocessingPipe = None, models_dir=None, verbose=False):
        self.models_dir = models_dir
        self.verbose = verbose
        self.pipe = pipe

    def get_pipe_filename(self):
        return self.models_dir + PipelineSerializationBase.PIPE_NAME

    def get_onehot_pipe_filename(self):
        return self.models_dir + PipelineSerializationBase.ONEHOT_PIPE_NAME


class PipelineReader(PipelineSerializationBase):
    def __init__(self, pipe: PreprocessingPipe = None, models_dir=None, verbose=False):
        super().__init__(pipe=pipe, models_dir=models_dir, verbose=verbose)

    def __load(self, filename):
        if self.verbose:
            log.info(f"Loading pipeline from <{filename}>")
        return joblib.load(filename)

    def load_pipe(self):
        filename = self.get_pipe_filename()
        return self.__load(filename)

    def load_onehot_pipe(self):
        filename = self.get_onehot_pipe_filename()
        return self.__load(filename)


class PipelineWriter(PipelineSerializationBase):
    def __init__(self, pipe: PreprocessingPipe = None, models_dir=None, verbose=False):
        super().__init__(pipe=pipe, models_dir=models_dir, verbose=verbose)

    def __dump(self, pipeline, filename):
        if self.verbose:
            log.info(f"Dumping pipeline to <{filename}>")
        joblib.dump(pipeline, filename)

    def dump_pipe(self):
        filename = self.get_pipe_filename()
        self.__dump(self.pipe.pipeline, filename)

    def dump_onehot_pipe(self):
        filename = self.get_onehot_pipe_filename()
        self.__dump(self.pipe.onehot_pipeline, filename)


"""
@}
"""
