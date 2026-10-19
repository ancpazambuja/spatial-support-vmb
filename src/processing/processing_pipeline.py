"""@package local_processing_pipeline.py
This module contains the Pipeline classes for the processing phase.
"""

#from denseclus import DenseClus
#from hdbscan import HDBSCAN

from sklearn.pipeline import Pipeline

from sklearn.cluster import DBSCAN 
from sklearn.cluster import OPTICS
from sklearn.cluster import HDBSCAN

from src.local_logging import Log as log


class ProcessingPipe:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.pipeline = None

    def fit(self, X):
        self.pipeline.fit(X)

    def predict(self, X):
        return self.pipeline.predict(X)


class DBSCANProcessingPipe(ProcessingPipe):
    def __init__(
        self, eps=0.1, min_samples=5, n_jobs=None, algorithm="auto", verbose=False
    ):
        super().__init__(verbose=verbose)
        self.eps = eps
        self.min_samples = min_samples
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.pipeline = self.__get_pipe()

    def __get_pipe(self):
        return Pipeline(
            [
                (
                    "clustering",
                    DBSCAN(
                        eps=self.eps, min_samples=self.min_samples, n_jobs=self.n_jobs, algorithm=self.algorithm
                    ),
                ),
            ]
        )


class OPTICSProcessingPipe(ProcessingPipe):
    def __init__(
        self,
        xi=0.05,
        min_cluster_size=0.1,
        min_samples=5,
        algorithm="kd_tree",
        verbose=False,
    ):
        super().__init__(verbose=verbose)
        self.min_samples = min_samples
        self.xi = xi
        self.min_cluster_size = min_cluster_size
        self.algorithm = algorithm
        self.pipeline = self.__get_pipe()

    def __get_pipe(self):
        return Pipeline(
            [
                (
                    "clustering",
                    OPTICS(
                        min_samples=self.min_samples,
                        xi=self.xi,
                        min_cluster_size=self.min_cluster_size,
                        algorithm=self.algorithm,
                    ),
                ),
            ]
        )


#class DENSECLUSProcessingPipe(ProcessingPipe):
#    def __init__(
#        self,
#        random_state=None,
#        n_neighbors=30,
#        min_samples=15,
#        min_cluster_size=100,
#        n_components=None,
#        cluster_selection_method="eom",
#        umap_combine_method="intersection",
#        prediction_data=True,
#        verbose=False,
#    ):
#        super().__init__(verbose=verbose)
#        self.random_state = random_state
#        self.n_neighbors = n_neighbors
#        self.min_samples = min_samples
#        self.min_cluster_size = min_cluster_size
#        self.n_components = n_components
#        self.cluster_selection_method = cluster_selection_method
#        self.umap_combine_method = umap_combine_method
#        self.prediction_data = prediction_data
#        self.pipeline = self.__get_pipe()
#
#    def __get_pipe(self):
#        return Pipeline(
#            [
#                (
#                    "clustering",
#                    DenseClus(
#                        random_state=self.random_state,
#                        n_neighbors=self.n_neighbors,
#                        min_samples=self.min_samples,
#                        min_cluster_size=self.min_cluster_size,
#                        n_components=self.n_components,
#                        cluster_selection_method=self.cluster_selection_method,
#                        umap_combine_method=self.umap_combine_method,
#                        prediction_data=self.prediction_data,
#                        verbose=self.verbose,
#                    ),
#                ),
#            ]
#        )
#
#
class HDBSCANProcessingPipe(ProcessingPipe):
    def __init__(
        self, min_samples=15, min_cluster_size=100, n_jobs=None, algorithm="auto", verbose=False
    ):
        super().__init__(verbose=verbose)
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.pipeline = self.__get_pipe()

    def __get_pipe(self):
        return Pipeline(
            [
                (
                    "clustering",
                    HDBSCAN(
                        min_samples=self.min_samples,
                        min_cluster_size=self.min_cluster_size,
                        n_jobs=self.n_jobs, 
                        algorithm=self.algorithm
                    ),
                ),
            ]
        )
