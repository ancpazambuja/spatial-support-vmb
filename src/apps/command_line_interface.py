"""@package commando_line_interface.py
Module with the command line interface.
"""
import argparse

from src.utl.classes import Base
from src.utl.logging import Log as log


class CLI(Base):
    """!
    Base class for command line interface.
    """
    def __init__(self, name="CLI"):
        super().__init__(name=name)
        self.cl = argparse.ArgumentParser()
    
        self.cl.add_argument('--experiment_tracking_uri', type=str,
                            #default="http://bw23b07n01.ep.petrobras.com.br:31947/", help='mlflow uri')
                            default="http://gr14sm02.ep.petrobras.com.br:31947/", help='mlflow uri')
        self.cl.add_argument('--experiment_tracking_name', type=str,
                            default="deep_tomography_2.0_fallback", help='mlflow experiment name')
        self.cl.add_argument('--verbose_level', type=int,
                            default=0, help='verbose level in [0, 3] name')

    def parse_args(self):
        return self.cl.parse_args()


class LoadDataCLI(CLI):
    """!
    Implements the command line interface for base class with load data functionality.
    """
    def __init__(self,name="TrainCLI"):
        super().__init__(name=name)
        cl = self.cl
        cl.add_argument('--checkpoint_basedir', type=str,
                            default=None, help='Path where the checkpoint dir will be created.')
        cl.add_argument('--dataset_max_samples', type=int,
                            default=-1, help='Limits the number of dataset samples. -1 means all data')
        cl.add_argument('--dataset_type', type=str,
                            default=None, help='dataset type. (deep, shallow)')
        cl.add_argument('--dataset_resolution', type=str,
                            default='high', help='dataset resolution. (high, mid, low)')
        cl.add_argument('--seed', type=int,
                            default=-1, help='random seed in [0,inf]. -1 means non-deterministic')
        cl.add_argument('--test_perc', type=float,
                            default=0.1, help='test set holdout percentage')
        cl.add_argument('--execution_id', type=str,
                            default=None, help='execution id (for mlflow logging purposes). Used in case of restarting the job.')
        cl.add_argument('--label_id', type=str,
                            default="undefined", help='main label of the execution (for mlflow logging purposes)')
        

class TrainCLI(LoadDataCLI):
    """!
    Implements the command line interface for training.
    """
    def __init__(self,name="TrainCLI"):
        super().__init__(name=name)
        cl = self.cl
        cl.add_argument('--conv_filter_size_x', type=int,
                            default=13, help='convolution filter size in x direction')
        cl.add_argument('--conv_filter_size_z', type=int,
                            default=13, help='convolution filter size in z direction (depth)')
        cl.add_argument('--conv_dilation_size_x', type=int,
                            default=1, help='convolution filter dilation in x direction')
        cl.add_argument('--conv_dilation_size_z', type=int,
                            default=1, help='convolution filter dilation in z direction')
        cl.add_argument('--transposed_conv_filter_size_x', type=int,
                            default=2, help='transposed convolution filter size in x direction')
        cl.add_argument('--transposed_conv_filter_size_z', type=int,
                            default=2, help='transposed convolution filter size in z direction')
        cl.add_argument('--model_arch', type=str,
                            default="undefined", help='model architecture to be used. (unet, transunet, atrous_unet, fourier_unet)')
        cl.add_argument('--unembed_type', type=str,
                        default="paper", help="if use_transformers is true, indicates the unembed_type ('linear', 'paper', 'none')")
        cl.add_argument('--max_epochs', type=int,
                            default=130, help='maximum number of training epochs')
        cl.add_argument('--device', type=str,
                        default="cuda:0", help="which device the training or evaluation will use ('cpu', 'cuda:all', 'cuda:[0-num_gpus-1).")
        cl.add_argument('--cv', type=int,
                            default=5, help='number of cross-validation subsets')
        
        

class RefitCLI(TrainCLI):
    """!
    Implements the command line interface for refit -- final training -- and evaluation.
    """
    def __init__(self,name="RefitCLI"):
        super().__init__(name=name)
        cl = self.cl
        cl.add_argument('--model_path', type=str,
            default=None, help='path to save the trained model after refit')
        cl.add_argument('--att_num_layers', type=int,
            default=None, help="Transformer's number of layers.")
        cl.add_argument('--refit', action=argparse.BooleanOptionalAction,
            help='indicates whether this execution is refit or not. Must be passed for double check.')
        cl.add_argument('--gs_best_run_id', type=str,
            default=None, help='Run id. Usually, the id of the best run of a gridsearch.')
        cl.add_argument('--gs_best_run_experiment_tracking_name', type=str,
            default=None, help='Name of the experiment which contains the gs_best_run_id.')
        cl.add_argument('--one_step_training', action=argparse.BooleanOptionalAction,
            default=True, help=' whether the training is in two phase or not.')
        cl.add_argument('--encdec_only', action=argparse.BooleanOptionalAction,
            default=False, help=' whether the architecture uses skip connections (U-Net) or not (Encoder-decoder).')

class EvalCLI(TrainCLI):
    """!
    Implements the command line interface for evaluation.
    """
    def __init__(self,name="EvalCLI"):
        super().__init__(name=name)
        cl = self.cl
        cl.add_argument('--model_path', type=str,
                            default=None, help='path of the trained model for evaluation')

class AddParametersInGSCLI(CLI):
    """!
    Implements the command line interface for evaluation.
    """
    def __init__(self,name="AddParametersInGSCLI"):
        super().__init__(name=name)
        cl = self.cl
        cl.add_argument('--filter', action=argparse.BooleanOptionalAction,
            default=False, help=' whether to apply filters or not.')
        cl.add_argument('--dataset_resolution', type=str,
                            default='high', help='dataset resolution. (high, mid, low)')



class AddTrainTestIndexesInCLI(CLI):
    """!
    Implements the command line interface for the auxiliar tools that add/update/change parameters
    values in mlflow runs.
    """
    def __init__(self,name="AddParametersInGSCLI"):
        super().__init__(name=name)
        cl = self.cl
        cl.add_argument('--type', type=str,
                            default='gs', help='Type of mlflow runs which will be updated. (gs or refit)')


class AnalyseEvaluationsCLI(AddParametersInGSCLI):
    """!
    Implements the command line interface for evaluation.
    """
    def __init__(self,name="AnalyseEvaluationsCLI"):
        super().__init__(name=name)
        cl = self.cl
        cl.add_argument('--experiment_prefix', type=str,
                            default=None, help='Prefix of the target experiments.')
        
class AnalyseEvaluationsExamplesCLI(LoadDataCLI):
    """!
    Implements the command line interface for showing the evaluation results related to examples.
    """
    def __init__(self,name="AnalyseEvaluationsExamplesCLI"):
        super().__init__(name=name)
        cl = self.cl
        cl.add_argument('--experiment_prefix', type=str,
                            default=None, help='Prefix of the target experiments.')
        cl.add_argument('--list_of_best_models', type=str,
                            default=None, help='List of best models.')
        cl.add_argument('--list_of_worst_models', type=str,
                            default=None, help='List of worst models.')
        cl.add_argument('--list_of_hardest_examples', type=str,
                            default=None, help='List of examples id for the hardest examples.')
        cl.add_argument('--list_of_easiest_examples', type=str,
                            default=None, help='List of examples id for the easiest examples.')
        cl.add_argument('--model_path', type=str,
                            default=None, help='Path of the trained models.')
        

class DT2InferenceCLI(CLI):
    """!
    Implements the command line interface for evaluation.
    """
    def __init__(self,name="DT2InferenceCLI"):
        super().__init__(name=name)
        cl = self.cl
        #cl.add_argument('--inference_dir_name', type=str,
        #    default=None, help='Path to the inference project.')
        cl.add_argument('--inference_proj_name', type=str,
            default=None, help='Name of the project inside inference_dir_name.')
        cl.add_argument('--refit_run_id', type=str,
            default=None, help='MLflow run id associated of the neural net final training.')
        cl.add_argument('--refit_run_experiment_tracking_name', type=str,
            default=None, help='Name of the experiment which contains the refit_run_id.')
        cl.add_argument('--label_id', type=str,
            default="undefined", help='Main label of the execution (for mlflow logging purposes)')
#        cl.add_argument('--nz', type=int,
#            default=None, help='Number of samples in depth of the input dataset.')
#        cl.add_argument('--nx', type=int,
#            default=None, help='Number of samples in X direction of the input dataset.')
        cl.add_argument('--weights_name_low_res', type=int,
            default=None, help='Name of the trained neural net weights of the first iteration.')
        cl.add_argument('--weights_name_med_res', type=int,
            default=None, help='Name of the trained neural net weights of the second iteration.')
        cl.add_argument('--weights_name_high_res', type=int,
            default=None, help='Name of the trained neural net weights of the third iteration.')
        cl.add_argument('--device', type=str,
                        default="cuda:0", help="which device the training or evaluation will use ('cpu', 'cuda:all', 'cuda:[0-num_gpus-1).")
        




