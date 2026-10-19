"""@package local_classes.py
Module with general utility classes.
"""

import pprint
import subprocess
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
import random
import torch
from glob import glob
from datetime import datetime

#!pip install matplotlib --upgrade
import segyio
from skorch.callbacks import MlflowLogger
from scipy.ndimage import gaussian_filter

from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from sklearn.model_selection import KFold

from src.experiment_tracking.mlflow_skorch_callbacks import ParamLogger
from src.processing.ml_models_utils import MLModelsUtils
from src.utl.logging import Log as log


class Graph:
    """!
    Utility class for graphics.
    """
    #definindo parametros globais
    PLOT_SX = 6 # tamanho do display em X
    PLOT_SY = 6 # tamanho do display em Y
    PLOT_FC_COLOR = 'w'  # cor de fundo branca


    #visualiza uma matriz 2d de rgb
    def drawImageRGB (img, title=''):
        plt.figure(figsize=[Graph.PLOT_SX, Graph.PLOT_SY], facecolor=Graph.PLOT_FC_COLOR)
        plt.title(title)
        plt.imshow(img)
        plt.show()

    #visualiza um atributo associado a uma escala de cores
    def drawImageAttr2(img, title='', vmin=None, vmax=None, cmap="gray", colorbar=False, interpolation=None, norm=None, save=False, extent=None, aspect=None):
        plt.figure(figsize=[Graph.PLOT_SX,Graph.PLOT_SY], facecolor=Graph.PLOT_FC_COLOR)
        plt.title(title)
        plt.imshow(img,cmap,vmin=vmin,vmax=vmax,interpolation=interpolation, norm=norm, extent=extent, aspect=aspect)
        if colorbar:
            plt.colorbar()
        if save:
            plt.savefig(fname="lixo.jpg")
        else:
            plt.show()



    #visualiza um atributo associado a uma escala de cores
    def drawImageAttr (img, title='', vmin=None, vmax=None, cmap="gray", colorbar=False, interpolation=None, norm=None, save=False):
        plt.figure(figsize=[Graph.PLOT_SX,Graph.PLOT_SY], facecolor=Graph.PLOT_FC_COLOR)
        plt.title(title)
        plt.imshow(img,cmap,vmin=vmin,vmax=vmax,interpolation=interpolation, norm=norm)
        if colorbar:
            plt.colorbar()
        if save:
            plt.savefig(fname="lixo.jpg")
        else:
            plt.show()


    #visualiza uma matriz 2d de rgb em um eixo de subplot
    def drawImageRGB_grid(eixo, img, xlim=None, title=None, interpolation=None, aspect=None):
        eixo.set_title(title)
        eixo.imshow(img,interpolation=interpolation,aspect=aspect)
        eixo.set_xlim(xlim)


    #visualiza um atributo associado a uma escala de cores
    def drawImageAttr_grid (eixo, img, vmin=None, vmax=None, title='', cmap="gray", colorbar=False, interpolation=None, norm=None, aspect=None):
        eixo.set_title(title)
        img = eixo.imshow(img, cmap, vmin=vmin, vmax=vmax, interpolation=interpolation, norm=norm, aspect=aspect)
        if colorbar:
            plt.colorbar(img, ax=eixo)

    # Desenha o histograma de um array com controle total na geracao dos bins.
    def hist (v, bins, title="histogram"):
        plt.figure(figsize=(Graph.PLOT_SX*2,Graph.PLOT_SY))
        plt.title(title)
        n,b,p=plt.hist(v,bins)
        plt.xlim( [ bins.min(), bins.max()])
        plt.show()
        return n,b,p


    # Desenha o histograma de um array, dentro de um grid, com controle total na geracao dos bins.
    def hist_grid (eixo, v, bins, title="histogram"):
        h, b = np.histogram(v,bins)           #calcula histogram
        centers = np.zeros(h.shape[0])        # calcula o centro de cada bin
        for i,val in enumerate(bins[:-1]):
            centers[i] = (val + bins[i+1]) / 2
        eixo.set_title(title)
        #eixo.plot(centers,h)                  #desenha a funcao
        eixo.fill_between(centers,h)          #preenche a parte debaixo da curva
        eixo.set_xlim( [centers.min(), centers.max()] )
        return h,bins


    # Desenha o histograma de uma array. Os bins são gerados automaticamente a partir de v.min e v.max.
    def hist2 (v, nbins, lim, title="histogram"):
        plt.figure(figsize=(Graph.PLOT_SX*2,Graph.PLOT_SY))
        plt.title(title)

        #para forçar os limites exatos dos bins
        #n,bins,patches=plt.hist(v,nbins,range=lim)

        #para deixar os limites dos bins serem definidos a partir de v.min e v.max
        n,bins,patches=plt.hist(v,nbins)
        plt.xlim(lim)
        plt.show()
        return n,bins,patches


    def drawFunction_grid(eixo, x,ys, title=None, cores=None, fill=False, lw=None, xscale=True):
        
        multiplo = False 
        if ys.ndim > 1:
            multiplo = True
            fill = False    # se forem mais linhas, nao desenhar com fill
        
        if cores is not None:
            if multiplo:
                assert(len(cores) == ys.shape[0])
            else:
                assert(len(cores) == 1)
        else:
            if multiplo:
                cores = ["black"] * ys.shape[0]
            else:
                cores = [ "black" ]

        if not multiplo:
            linhas = np.reshape(ys, (1, ys.shape[0]))
        else:
            linhas = ys

        eixo.set_title(title)

        for l,c in list(zip(linhas,cores)):
            if fill:
                eixo.fill_between(x,l)
            else:
                eixo.plot(x, l, c, lw=lw)

        if xscale:
            eixo.set_xlim(x.min(), x.max())


    def drawFunction(x,ys, title=None, cores=None, fill=False, lw=None):

        multiplo = False 
        if ys.ndim > 1:
            multiplo = True
            fill = False    # se forem mais linhas, nao desenhar com fill

        if cores is not None:
            if multiplo:
                assert(len(cores) == ys.shape[0])
            else:
                assert(len(cores) == 1)
        else:
            if multiplo:
                cores = ["black"] * ys.shape[0]
            else:
                cores = [ "black" ]

        if not multiplo:
            linhas = np.reshape(ys, (1, ys.shape[0]))
        else:
            linhas = ys

        plt.title(title)

        for l,c in list(zip(linhas,cores)):
            if fill:
                plt.fill_between(x,l)
            else:
                plt.plot(x, l, c, lw=lw)

        plt.xlim(x.min(), x.max())
        plt.show()
        


    #Normaliza um numpy array outplace
    def normalize(v, vmin=None, vmax=None):
        min, max = Graph.getMinMax(v)
        if vmin is not None:
            min=vmin
        if vmax is not None:
            max=vmax
        return (v - min) / ( max - min )

    def getMinMax(v):
        return np.amin(v), np.amax(v) 


    # cria uma nova imagem com a luminância
    def lum(img):
        return np.dot(img,[0.2126,0.7152,0.0722])


    def printGPUInfo():
        iscuda       = torch.cuda.is_available()
        print(f"[ POSSUI GPGPUs? ] {iscuda}")
        if iscuda:
            device_count = torch.cuda.device_count()
            device_id    = torch.cuda.current_device()
            device_name  = torch.cuda.get_device_name(id)
            print(f"[ QUANTAS GPGPUs? ] {device_count}")
            print(f"[ ID DA GPGPU CORRENTE ] {device_id}")
            print(f"[ MODELO GPGPU ] {device_name}")


    def getTensorDevice():
        ret = 'cpu'
        iscuda       = torch.cuda.is_available()
        if iscuda:
            ret = 'cuda'
        return ret



class Utils:
    """! Utility class.
    """
    def __make_pretty(styler):
        styler.set_caption("Statistics")
        # styler.format(rain_condition)
        # styler.format_index(lambda v: v.strftime("%A"))
        # styler.background_gradient(axis=None, vmin=1, vmax=5, cmap="YlGnBu")
        styler.background_gradient(axis=None, vmin=0, vmax=10000, cmap="gray")
        return styler

    def view_statistics(describe):
        return describe.T.loc[:].style.pipe(Utils.__make_pretty)
    
    def print_job_info(outliers, job_names):
        '''! Prints job information. 
        '''
        jobs = outliers.jobs.get_jobs_table()
        outliers_table = outliers.outliers
        outliers_statistics = outliers.statistics
        njobs = len(job_names)
        for i, jname in enumerate(job_names):
            
            log.info(f"<< {i}/{njobs} -- Job Name: {jname}  >>")
            is_outlier = outliers_table[jname]['status']
            log.print_raw(f"\tis outlier? { is_outlier }")
            if is_outlier: 
                nodes = list()
                full_nodes = outliers_table[jname]["nodes"]
                for k,v in full_nodes.items():
                    if v["status"]:
                        nodes.append((k,v, {'n_tasks_outliers' : len(outliers_statistics[jname]['nodes'][k]['times'])})) 
                log.print_raw(f"\tnnodes: {len(nodes)} / {len(full_nodes)} -- {nodes}")


    def ssim(image1, image2):
        # Convert torch tensors to numpy arrays
        image1_np = image1.detach().cpu().numpy()
        image2_np = image2.detach().cpu().numpy()
    
        # Determine the data range from the images
        data_range = max(image1_np.max() - image1_np.min(), image2_np.max() - image2_np.min())
    
        # Constants for SSIM calculation
        K1, K2 = 0.01, 0.03
        C1 = (K1 * data_range) ** 2
        C2 = (K2 * data_range) ** 2
    
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
        num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = num / den
    
        # Compute the mean SSIM over the entire image
        ssim_value = np.mean(ssim_map)
    
        return ssim_value, ssim_map

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def create_stable_cv(cv=None, seed=0):
        # Instancie o KFold fixando a seed explicitamente
        stable_cv = KFold(n_splits=cv, shuffle=True, random_state=seed)
        return stable_cv

    def set_seed(seed):
#        random.seed(seed) 
#        np.random.seed(seed)
#        torch.manual_seed(seed)
#        torch.cuda.manual_seed(seed)
#        torch.backends.cudnn.benchmark = False
#        #torch.use_deterministic_algorithms(True)  # soh funciona com warn_only=True

        # 1. Python e variáveis de ambiente
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # Necessário se usar deterministic_algorithms
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # 2. PyTorch (CPU e GPU)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 3. Travar algoritmos de hardware
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 4. Avisar sobre operações não-determinísticas inevitáveis
        #torch.use_deterministic_algorithms(True, warn_only=True)
        torch.use_deterministic_algorithms(True, warn_only=False)


    def two_split(nsamples=None, test_perc=0.2, seed=None, fixed_seed=False, verbose=False):
        assert nsamples is not None and nsamples > 0, f"[ ERROR ] nsamples must be defined and greater then 0."
        assert fixed_seed == (seed is None), f"[ ERROR ] fixed_seed and seed have to be opposite. (fixed_seed={fixed_seed}, seed={seed})"
        
        if verbose:
            log.printMessage(f"Spliting samples into train and test!")
            log.info(f"Total nmb of samples: {nsamples}")
            log.info(f"Percentage of test samples: {test_perc*100}%")
        
        indexes = np.arange(nsamples)
        if fixed_seed: 
            np.random.seed(10)
        elif seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indexes)
        n = indexes.shape[0]
        assert n == nsamples, f"[ ERRO ] Internal error!"
        ntest = int(np.maximum(np.ceil(test_perc*n), 1))
        ntest = int(np.minimum(nsamples-1, ntest))
        ntrain = n-ntest
        
        set1_indexes = indexes[:ntrain]
        set2_indexes = indexes[ntrain:]
        
        if verbose:
            log.debug(f"set1_indexes.shape[0]+set2_indexes.shape[0] == nsamples        -----  {set1_indexes.shape[0]}, {set2_indexes.shape[0]}, {nsamples}")
        assert set1_indexes.shape[0]+set2_indexes.shape[0] == nsamples, f"[ ERROR ] Internal error. Set1 and set2 must sum 100% of the original data!"
        assert len(list(set(set1_indexes).intersection(set2_indexes))) == 0, f"[ ERROR ] Data leaked! Test data leaked to train data!"
        
        return set1_indexes, set2_indexes

    def get_execution_names(global_vars=None):
        return Utils.get_execution_names_2(
            execution_name=global_vars.get_execution_name(),
            ts=global_vars.get_timestamp(),
            ns=global_vars.get_data_max_samples(),
            data = global_vars.get_data_type(),
            label = global_vars.get_label_id(),
            arch = global_vars.get_model_arch(),
            data_res = global_vars.get_data_resolution()
        )

    def get_execution_names_2(execution_name=None, ts=None, ns=None, data=None, label=None, arch=None, data_res=None):
        if execution_name is None:
            #nsamples
            if ns == -1:
                ns = "full"
            else:
                ns = str(ns)
            gs_execution_name, refit_execution_name = Utils.get_execution_names_aux(ts, ns, data, label, arch, data_res) 
        else:
            gs_execution_name = execution_name
            refit_execution_name = f'REFIT__[none]'

        return gs_execution_name, refit_execution_name

    def get_execution_name_suffix(ts=None, ns=None, data=None, label=None, arch=None, data_res=None):
        return f'{label}_{data}-data_{data_res}-res_{ns}-samples_{arch}-arch_{ts}'

    def get_refit_execution_name(ts=None, ns=None, data=None, label=None, arch=None, data_res=None):
        suffix = Utils.get_execution_name_suffix(ts=ts, ns=ns, data=data, label=label, arch=arch, data_res=data_res) 
        return f'REFIT__{suffix}'

    def get_gs_execution_name(ts=None, ns=None, data=None, label=None, arch=None, data_res=None):
        suffix = Utils.get_execution_name_suffix(ts=ts, ns=ns, data=data, label=label, arch=arch, data_res=data_res) 
        return f'GS__{suffix}'


    def get_execution_names_aux(ts=None, ns=None, data=None, label=None, arch=None, data_res=None):
        gs_execution_name = Utils.get_gs_execution_name(ts=ts, ns=ns, data=data, label=label, arch=arch, data_res=data_res) 
        refit_execution_name = Utils.get_refit_execution_name(ts=ts, ns=ns, data=data, label=label, arch=arch, data_res=data_res) 
        return gs_execution_name, refit_execution_name
        
    def get_train_callbacks(global_vars=None, mlflow_client=None, run=None, is_one_step_training=True, is_encoder_decoder_only=False, grokking=False):

        if mlflow_client is None:
            log.info(f"mflow_client IS None")
        else:
           log.info(f"mflow_client is NOT  None")
        
        refit_checkpoint_callback_fn_prefix = global_vars.get_execution_name()
        log.info(f"refit_checkpoint_callback_fn_prefix = {refit_checkpoint_callback_fn_prefix}")

        # t_0 and t_mult below were set to generate this sequence of cycle lenghts when max_epochs is 500:
        # 34, 68, 136, 272 (total of 510 epochs, so the last 10 will not run) 
        if grokking:
            t_0 = 500
            t_mult = 2
        else:
            t_0 = max(int(0.068*global_vars.get_max_epochs()),1) # 1 is the minimal accepted value
            t_mult = 2

        #lr_scheduler_cb, lr_sched_name = None, None
        lr_scheduler_cb, lr_sched_name = MLModelsUtils.get_skorch_lr_scheduler_cosine_annealing_warm_restarts_callback(T_0=t_0, T_mult=t_mult)
        #lr_scheduler_cb, lr_sched_name = MLModelsUtils.get_skorch_lr_scheduler_cosine_annealing_callback(T_max=global_vars.get_max_epochs())
        #lr_scheduler_cb, lr_sched_name = MLModelsUtils.get_skorch_lr_scheduler_exponential_callback(gamma=0.985)
        #lr_scheduler_cb, lr_sched_name = MLModelsUtils.get_skorch_lr_scheduler_lambda_callback(decay=0.0002)

        lr_scheduler_debuger_cb = MLModelsUtils.get_skorch_print_lr_scheduler_callback()

        if mlflow_client is None: # fluent api
            gs_mlflow_logger_cb = MlflowLogger(create_artifact=False, log_on_epoch_end=True)
        else: # client api
            gs_mlflow_logger_cb = MlflowLogger(run, mlflow_client, create_artifact=False, log_on_epoch_end=True)

        refit_mlflow_logger_cb = MlflowLogger(run, mlflow_client, create_artifact=False, log_on_epoch_end=True)
        refit_checkpoint_cb = MLModelsUtils.get_skorch_checkpoint_callback(
                            fn_prefix=f"{refit_checkpoint_callback_fn_prefix}__",
                            dirname=global_vars.get_dirs().get_results_dirname(),
                            load_best=True)

        if grokking:
            patience = -1
            refit_early_stopping_cb = None
        else:
            patience = 300
            refit_early_stopping_cb = ('early_stopping', MLModelsUtils.get_skorch_early_stopping_callback(patience=patience))


        grad_norm_callback = MLModelsUtils.get_skorch_gradient_clipping_callback(clip_value=1.0)
        #grad_norm_callback = MLModelsUtils.get_skorch_monitor_gradient_clipping_callback(clip_value=1.0)

        ### OBS ###
        # Testes empiricos mostraram que o encoder-decoder treina sem overfitting ateh aproximadamente
        # 250 epocas com kernel size = 9 (e com lr_schedule).
        refit_two_step_training_cb = ( 
            'two_step_training',
            MLModelsUtils.get_skorch_two_step_training_callback(
                #parametros validos para max_epochs = 1000
                nmb_of_epochs=global_vars.get_max_epochs(), step1_perc=0.125, warmup_perc=0.50, finetune_lr=1e-3
            )
        )

        ### OBS ###
        # Testes empiricos mostraram que o encoder-decoder treina sem overfitting ateh aproximadamente
        # 250 epocas com kernel size = 9 (e com lr_schedule).
        refit_encoder_decoder_only_cb = ( 
            'encoder_decoder_only',
            MLModelsUtils.get_skorch_disable_skip_connections_callback()
        )

        refit_param_logger_cb = ParamLogger()
        #loadInitState_cb = MLModelsUtils.get_skorch_loadInitState_callback(checkpoint_cb)
       
        gs_cb_list = [
            ('mlflow_logger', gs_mlflow_logger_cb),#, param_logger_cb
            ('lr_scheduler', lr_scheduler_cb),
            ('lr_scheduler_debbuger', lr_scheduler_debuger_cb),
            ('grad_clip',grad_norm_callback),
            #checkpoint_cb, early_stopping_cb, mlflow_logger_cb#, param_logger_cb
        ]

        refit_cb_list = [
            ('check_point', refit_checkpoint_cb), 
            ('mlflow_logger', refit_mlflow_logger_cb), 
            ('lr_scheduler', lr_scheduler_cb),
            ('lr_scheduler_debbuger', lr_scheduler_debuger_cb),
            ('grad_clip',grad_norm_callback),
            #refit_param_logger_cb
        ]

        if not grokking:
            refit_cb_list.append(refit_early_stopping_cb)

        if is_encoder_decoder_only:
            log.debug(f"Setting ENCODER_DECODER ONLY callback")
            refit_cb_list.append(refit_encoder_decoder_only_cb)
        else:
            log.debug(f"NOT setting ENCODER_DECODER ONLY callback")
        
        if not is_one_step_training:
            log.debug(f"Setting TWO_PHASE_TRAINING callback")
            refit_cb_list.append(refit_two_step_training_cb)
        else:
            log.debug(f"NOT setting two phase training callbak")

        return gs_cb_list, refit_cb_list, lr_sched_name, patience
   

    def get_model_data_size(data=None, depth=5):
        shape_tmp = data.get_current_data().get_X().shape
        _nchannels = shape_tmp[1]
        _nz_cut, _nx_cut = Utils.find_size(shape_tmp[2], shape_tmp[3], depth=depth)
        return { 
                'nchannels' : _nchannels, 
                'nz_cut' : _nz_cut, 
                'nx_cut' : _nx_cut 
            }

    def find_size(n1, n2, depth=5):
        def is_valid(N, nsteps):
            nn=N
            for i in range(nsteps):
                if (nn%2 != 0): return False
                nn = nn//2
            return True
        def find(n, depth):
            nn = None
            for i in range(n, 2**(depth-1), -1 ):
                if is_valid(i,depth):
                    nn = i
                    break
            return nn
        #depth = 5
        nn1 = find(n1,depth)
        nn2 = find(n2,depth)
        return nn1, nn2

    def get_timestamp():
        return datetime.now().isoformat().split(".")[0].replace(":", ".")

    def get_dataset_resolution_from_name(run_name=None):
        dt_res = None
        if run_name.find('high-res') >= 0:
            dt_type = 'highres'
        elif run_name.find('med-res') >= 0:
            dt_type = 'medres'
        elif run_name.find('low-res') >= 0:
            dt_type = 'lowres'
        else:
            raise ValueError(f"Invalid dataset resolution!")
        return dt_type
    
    def get_dataset_type_from_name(run_name=None):
        dt_type = None
        if run_name.find('deep') >= 0:
            dt_type = 'deep'
        elif run_name.find('shallow') >= 0:
            dt_type = 'shallow'
        else:
            raise ValueError(f"Invalid dataset type!")
        return dt_type


class Sismo:
#    ILINE_IDX = (segyio.TraceField.INLINE_3D-1)    / 4
#    XLINE_IDX = (segyio.TraceField.CROSSLINE_3D-1) / 4

    def __find_norm(v):
        n = np.abs( np.min(v) )
        n = np.maximum(n, np.max(v) )
        return n

    def __wiggle_normalize(v,norm,perc):
        return v / (2.*norm*perc) # [-0.5,0.5]
    

    def load_post_stack_segy(fname):
        with segyio.open(fname, ignore_geometry=True) as file:
            n_traces = file.tracecount
            sample_rate = segyio.tools.dt(file) / 1000
            n_samples = file.samples.size
            time = file.samples
            
            header = file.header
            _inline = file.attributes(segyio.TraceField.INLINE_3D)
            _xline  = file.attributes(segyio.TraceField.CROSSLINE_3D)
            inline = np.array(_inline[:])
            xline = np.array(_xline[:])
            min_iline = np.min(inline)
            max_iline = np.max(inline)
            nilines = int(max_iline-min_iline+1)
            min_xline = np.min(xline)
            max_xline = np.max(xline)
            nxlines = int(max_xline-min_xline+1)
            
            amplitude = file.trace.raw[:]
            amplitude = np.array(amplitude).reshape((nilines,nxlines,n_samples))
            
            inline.reshape((nilines, nxlines)) 
            xline.reshape((nilines, nxlines)) 
            
            #Premissa de que o dado estah cheio
            assert nilines * nxlines == n_traces, "[ ERROR ] Dado não está cheio!"
            
        #      min_xlin
        #      max_xlin
            return {
                    'nt':n_traces, 
                    'na':n_samples,
                    'dt':sample_rate, 
                    't' : time, 
                    'v' : amplitude,
                    'il' : inline,
                    'xl' : xline,
                    'iline_info' : { 'min' : min_iline, 'max' : max_iline, 'n' : nilines },
                    'xline_info' : { 'min' : min_xline, 'max' : max_xline, 'n' : nxlines },
                    }

    def draw_wiggle_grid(eixo, traces, x, title=None, max_global=False):
        if (len(traces.shape) == 1):
            painel = np.reshape( traces, (1,traces.shape[0]) )
        else:
            painel = traces

        if max_global:
            norm_factor = Sismo.__find_norm(painel.ravel())  
            
        for i,tr in enumerate(painel):
            if not max_global:
                norm_factor = Sismo.__find_norm(tr)  
            transf = matplotlib.transforms.Affine2D().scale(1./(2.*norm_factor), 1.).translate(i,0.) +  eixo.transData
            #clip = matplotlib.transforms.Bbox( [ [-perc/2., 0.], [perc/2., dado['t'][-1]] ]).transformed(translation)
            #print(clip)
            #eixo.plot(tr, dado['t'],'k-', **{'transform' : translation })
            #eixo.fill_betweenx(dado['t'], i, tr, transform=translation, color='k' )
            #eixo.fill_betweenx(dado['t'], i, tr, transform=translation, color='k' )

            #eixo.fill_betweenx(dado['t'], 0, tr, transform=translation, where=( np.logical_and( tr<0, tr>-(norm_factor*perc)) ),color='k', edgecolor='k' )
            #eixo.fill_betweenx(dado['t'], 0, tr, transform=translation, where=( np.logical_and( tr>0, tr<(norm_factor*perc)) ),color='k' )
            #eixo.fill_betweenx(dado['t'], 0, tr, transform=transf, where=( tr<0 ),color='b', edgecolor='k' )
            eixo.plot(tr, x,'k-', transform=transf, lw=0.5 )
            eixo.fill_betweenx(x, 0, tr, transform=transf, where=( tr>0 ),color='k' )
            
        if title:
            eixo.set_title(title)
        eixo.set_ylim(eixo.get_ylim()[::-1])
        eixo.set_xlim([-0.5, i+0.5])
        
    def draw_variable_density_grid(eixo, traces, vmin=None, vmax=None, title=None, cmap='seismic', colorbar=True, interpolation='bilinear', aspect=None, norm=None, save=None):
        if (len(traces.shape) == 1):
            painel = np.reshape( traces, (1,traces.shape[0]) )
        else:
            painel = traces
        norm_factor = Sismo.__find_norm(painel.ravel())  
        #transf = matplotlib.transforms.Affine2D().scale(1./(2.*norm_factor), 1.).translate(i,0.) +  eixo.transData
        Utils.drawImageAttr_grid(eixo, painel/norm_factor, vmin=vmin, vmax=vmax, title=title, cmap=cmap, colorbar=colorbar, interpolation=interpolation, norm=norm, aspect=aspect, save=save)
        #eixo.set_ylim(eixo.get_ylim()[::-1])
        
    def print_segy_info(dado):
        print("[ NTRACOS ] ", dado['nt'])
        print("[ NAMSTS ] ", dado['na'])
        print("[ INTERVALO DE AMOSTRAGEM (ms) ] ", dado['dt'])
        print("[ INTERVALO DE TEMPO (ms) ] [", np.min(dado['t']), ", ", np.max(dado['t']), "]")
        print("[ ILINE INFO ] ", dado['iline_info']['min'], dado['iline_info']['max'], dado['iline_info']['n'])
        print("[ XILINE INFO ] ", dado['xline_info']['min'], dado['xline_info']['max'], dado['xline_info']['n'])
        print("[ SHAPE OF VOL ]", dado['v'].shape)
        print("[ tempos ]", dado['t'][-1])

    
#########################################################################
## CLASSE BASE COM ATRIBUTOS BÁSICOS
class Base:

    def __init__(self, name="Base class name"):
        """documentação... a fazer."""
        self.name = name

        
    def __str__(self):
        ret = ""
        ret = ret + pprint.pformat(vars(self))
        ret = ret + "\n"
        return ret

    
    def show(self, header=True):
        if header:
            print("########################")

        print(self.__str__())
        
        if header:        
            print("########################")

#    def __str__(self):
#        self.show(header=True)
#        return ""
        

class ProjectDirs(Base):
    """!
    Represents pathnames and filenames of the main user artifacts,
    both created by the program or read by the program.
    """
    SEP = "/"
    def __init__(
        self,
        basedir=None,
        x_dirname=None,  
        x1_filename_prefix = None,
        x2_filename_prefix = None,
        x3_filename_prefix = None,
        
        y_dirname=None,
        y1_filename_prefix = None,
        y2_filename_prefix = None,
        y3_filename_prefix = None,
        
        models_dirname=None,
        true_models_filename_prefix = None,
        highres_models_filename_prefix = None,
        mediumres_models_filename_prefix = None,
        lowres_models_filename_prefix = None,
        water_mask_filename_prefix=None,
        
        yy_dirname = None,
        verbose=False,
    ):
        super().__init__(name="ProjectDirs")
        """!
        @param x_dirname ...
        @param y_dirname ...
        """
        self.basedir = basedir
        self.x_dirname = x_dirname 
        self.x1_filename_prefix = x1_filename_prefix
        self.x2_filename_prefix = x2_filename_prefix
        self.x3_filename_prefix = x3_filename_prefix
        
        self.y_dirname = y_dirname 
        self.y1_filename_prefix = y1_filename_prefix
        self.y2_filename_prefix = y2_filename_prefix
        self.y3_filename_prefix = y3_filename_prefix
        
        self.models_dirname = models_dirname
        self.true_models_filename_prefix = true_models_filename_prefix
        self.highres_models_filename_prefix = highres_models_filename_prefix
        self.mediumres_models_filename_prefix = mediumres_models_filename_prefix
        self.lowres_models_filename_prefix = lowres_models_filename_prefix
        self.water_mask_filename_prefix = water_mask_filename_prefix
        
        self.yy_dirname = yy_dirname
        
        self.yy_subdirsname = ("test", "train", "validation", "figs")
        
        self.verbose = verbose

    def get_basedir(self):
        return self.basedir
    
    def get_X_dirname(self):
        return self.x_dirname
    
    def get_X_highres_prefix(self):
        return self.get_X_dirname() + self.x1_filename_prefix
    
    def get_X_mediumres_prefix(self):
        return self.get_X_dirname() + self.x2_filename_prefix
    
    def get_X_lowres_prefix(self):
        return self.get_X_dirname() + self.x3_filename_prefix
    
    def get_Y_dirname(self):
        return self.y_dirname
    
    def get_Y_highres_prefix(self):
        return self.get_Y_dirname() + self.y1_filename_prefix
    
    def get_Y_mediumres_prefix(self):
        return self.get_Y_dirname() + self.y2_filename_prefix
    
    def get_Y_lowres_prefix(self):
        return self.get_Y_dirname() + self.y3_filename_prefix
    
    def get_models_dirname(self):
        return self.models_dirname
    
    def get_true_models_prefix(self):
        return self.get_models_dirname() + self.true_models_filename_prefix
    
    def get_highres_models_prefix(self):
        return self.get_models_dirname() + self.highres_models_filename_prefix
    
    def get_mediumres_models_prefix(self):
        return self.get_models_dirname() + self.mediumres_models_filename_prefix
    
    def get_lowres_models_prefix(self):
        return self.get_models_dirname() + self.lowres_models_filename_prefix
    
    def get_water_mask_prefix(self):
        return self.get_models_dirname() + self.water_mask_filename_prefix
    
    def get_results_dirname(self):
        return self.yy_dirname
    
    def __str__(self):
        ret = "\n"
        #ret = ret + "[ INFO ] Trainning info:\n" + str(self.__dict__)
        ret = ret + super().__str__() + "\n"
#        ret = ret + f"   Project path               = {self.get_basedir()}" + "\n"
#        ret = ret + f"   X path                     = {self.get_X_dirname()}" + "\n"
#        ret = ret + f"   Y path                     = {self.get_Y_dirname()}" + "\n"
#        ret = ret + f"   X high resolution prefix   =  {self.get_X_highres_prefix()}" + "\n"
#        ret = ret + f"   X medium resolution prefix =  {self.get_X_mediumres_prefix()}" + "\n"
#        ret = ret + f"   X low resolution prefix    =  {self.get_X_lowres_prefix()}" + "\n"
#        ret = ret + "\n"
#        ret = ret + f"   Models outputdir           = {self.get_models_checkpoint_dirname()}" + "\n"
#        ret = ret + f"   Predictions on test data   = {self.get_results_dirname()}" + "\n"
        return ret







class DataInfo(Base):
    
    def __init__(self, name="Parametos dos Dados de Entrada", nsamples_trainning=10):
        super().__init__(name="DataInfo")
        self.offset  = 350
        self.nsamples_trainning = nsamples_trainning
        self.x_shape = (351,901,15) #h,w,ncanais
        self.y_shape = (351,901)    #h,w
        #self.x_shape = (352,352,14) #h,w,ncanais
        #self.y_shape = (352,352)    #h,w
    
    
    def get_nx(self):
        return self.x_shape[1]

    def get_nz(self):
        return self.x_shape[0]

    def get_nlag(self):
        return self.x_shape[2]

    def get_offset(self):
        return self.offset

    def get_nmb_of_trainning_samples(self): 
        return self.nsamples_trainning
    
    def set_nmb_of_trainning_samples(self, n):
        self.nsamples_trainning = n
    
    def get_x_shape(self):
        return self.x_shape
    
    def get_x_channels(self):
        return self.x_shape[2]
    
    def get_y_shape(self):
        return self.y_shape


class BuildDirectories(Base):

    #DIR_DEEPER_DATA = "/gemini/tg/ntbs/doutorado/4semestre/inf2475/trabalho.final/deeptomograph_data/data_new_and_bigger/"
    #DIR_SHALLOWER_DATA = "/gemini/tg/ntbs/doutorado/4semestre/inf2475/trabalho.final/deeptomograph_data/data/"
    DIR_DEEPER_DATA = "/gemini/tg/ntbs/doutorado/5semestre/estudo.orientado/deeptomograph_data/data_new_and_bigger/"
    DIR_SHALLOWER_DATA = "/gemini/tg/ntbs/doutorado/5semestre/estudo.orientado/deeptomograph_data/data/"

    def __init__(self, data=None, basedir=None):
        super().__init__(name="BuildDirectories")
        self.basedir = basedir
        self.data = data
    
    def build(self):
        if self.data is None:
            if self.basedir == BuildDirectories.DIR_DEEPER_DATA:
                self.data = "deep" 
            elif self.basedir == BuildDirectories.DIR_SHALLOWER_DATA:
                self.data = "shallow" 
            else:
                raise ValueError(f"Parameter 'data_dir' must assume the following values: \n{BuildDirectories.DIR_DEEPER_DATA} \nand \n{BuildDirectories.DIR_SHALLOWER_DATA}. \nActual value: {self.data_dir}")

        if self.data == "deep": 
            d = self.buildDirsForNewerData()
        elif self.data == "shallow":
            d = self.buildDirsForOlderData()
        else:
            raise ValueError(f"Parameter 'data' must assume the following values: deep and shallow (not marmousi). Actual value: {self.data}")
        return d
        
    def buildDirsForOlderData(self):
        return ProjectDirs(
            
            basedir=self.basedir,
            x_dirname=self.basedir + "DadosRede/",
            x1_filename_prefix = "inputs1_[0-9]*.npy",
            x2_filename_prefix = "inputs2_[0-9]*.npy",
            x3_filename_prefix = "inputs3_[0-9]*.npy",
            
            y_dirname=self.basedir + "DadosRede/",
            y1_filename_prefix = "outputs1_[0-9]*.npy", #highres
            y2_filename_prefix = "outputs2_[0-9]*.npy", #medmres
            y3_filename_prefix = "outputs3_[0-9]*.npy", #lowres
            
            models_dirname = self.basedir + "nothing/",
            true_models_filename_prefix = "model_[0-9]*.vp",
            highres_models_filename_prefix = "model_suave_[0-9]*.vp",
            mediumres_models_filename_prefix = "model_suave2_[0-9]*.vp",
            lowres_models_filename_prefix = "model_suave3_[0-9]*.vp",
            water_mask_filename_prefix = "nothing",
            
            yy_dirname = self.basedir + "deeptomograph_results/predictions/",
            verbose=True)
    
    def buildDirsForNewerData(self):
        return ProjectDirs(
            basedir=self.basedir,
            x_dirname=self.basedir + "./",
            x1_filename_prefix = "input_suave_[0-9]*.npy",
            x2_filename_prefix = "input_suave2_[0-9]*.npy",
            x3_filename_prefix = "input_suave3_[0-9]*.npy",
            
            y_dirname=self.basedir + "./",
            y1_filename_prefix = "nothing",
            y2_filename_prefix = "nothing",
            y3_filename_prefix = "nothing",
            
            models_dirname = self.basedir + "./",
            true_models_filename_prefix = "model_[0-9]*.npy",
            highres_models_filename_prefix = "model_suave_[0-9]*.npy",
            mediumres_models_filename_prefix = "model_suave2_[0-9]*.npy",
            lowres_models_filename_prefix = "model_suave3_[0-9]*.npy",
            water_mask_filename_prefix = "mask_agua_[0-9]*.npy",
            
            yy_dirname = self.basedir + "deeptomograph_results/predictions/",
            verbose=True)


##############################


class ShellCommand:
    def __init__(self, base_command):
        self.base_command = base_command
        self.args = []

    def set_args(self, *args):
        self.args = list(args)

    def run(self):
        command = [self.base_command] + self.args
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.CalledProcessError as e:
            return {
                "stdout": e.stdout,
                "stderr": e.stderr,
                "returncode": e.returncode,
                "error": str(e)
            }