
"""@package local_data_builder.py
Building process of data and models from persisted files.
"""
import numpy as np
from  glob import glob

from src.utl.logging import Log as log
from src.utl.classes import ProjectDirs, Utils as utl
from src.utl.serialization import Load, NPLoad
from src.data.data import Models2D, DeepTomographyModels, SupervisedData, Data, ModelIdx, DataIdx

class BaseBuilder:
    def __init__(self, dirs:ProjectDirs=None, z_len_newer_data=680, max_samples=None, verbose=False):
        self.dirs = dirs
        self.verbose = verbose
        self.z_len_newer_data = z_len_newer_data
        self.max_samples = max_samples
    
    def _get_filelist(self, str_aux, indexes_list=None):
        filelist = glob(str_aux)
        filelist.sort() 
        
        if self.verbose:
            if indexes_list is not None: 
                log.debug(f"len(indexes_list) = {len(indexes_list)}")

        if indexes_list is None:
            return filelist
        else:
            ######################################################
            #### ATENCAO: A LINHA ABAIXO ESTÁ COM ERRO. MAS....
            # Quando self.max_samples é -1, perde-se a última 
            # posicao dos dados de treino e teste.
            # Mas eu não posso corrigir isso agora, pois isso 
            # causaria um descompasso entre as execucoes que jah 
            # ocorreram (com o erro) e as analises e avaliacoes
            # que foram executadas depois. Alem de novas execucoes
            # que ficariam com datasets diferentes.
            ######################################################
            final_list = [ filelist[i] for i in indexes_list[:self.max_samples] ] 
            log.debug(f"len(final_list) = {len(final_list)}")
            return final_list 
    
    def _get_X_highres_filenames(self):
        return self._get_filelist(self.dirs.get_X_highres_prefix())
    
    def is_newer_data(self):
        filenames = self._get_X_highres_filenames()
        x_aux = NPLoad().load(filenames[0])
        
        if self.verbose:
            log.info(f"FINDING OUT DATA FORMAT")
            log.info(f"\tFilename: {filenames[0]}")
            log.info(f"\tFile shape: {x_aux.shape}")
            log.info(f"\tTarget z_len (newer dataset): {self.z_len_newer_data}")
            
        return x_aux.shape[0] == self.z_len_newer_data
    
    def train_test_split(self, test_perc=0.2, seed=None):
        filenames = self._get_X_highres_filenames()
        nsamples = len(filenames)
        if test_perc == 0.:
            return np.arange(nsamples), None
        else: 
            return utl.two_split(nsamples=nsamples, test_perc=test_perc, seed=seed, verbose=self.verbose)


class BuildModels:
    """
    Builds an instance of Models2D.
    """
    def __init__(self, m_filelist=None, loader:Load=NPLoad(), verbose=False):
        self.m_filelist = m_filelist
        self.loader = loader 
        self.verbose = verbose

    def build(self):
        log.line()
        log.info(f"Reading models...")
        log.info(f"\t\tNumber of models: {len(self.m_filelist)}")
        
        #finding number model shape
        m =  self.loader.load(self.m_filelist[0])
        log.info(f"\t\t2D shape: {m.shape}")
        models = np.zeros( (len(self.m_filelist),) + m.shape, dtype=float)
        
        for i, f in enumerate(self.m_filelist): 
            models[i] = self.loader.load(f)
        
        return Models2D(models=models, verbose=self.verbose) 


class BuildDeepTomographyModels(BaseBuilder):
    """
    Builds an instance of Deep Tomography models.
    """
    def __init__(self, dirs:ProjectDirs, models_mask=None, z_len_newer_data=None,  max_samples=None, verbose=False):
        super().__init__(dirs=dirs, z_len_newer_data=z_len_newer_data, max_samples=max_samples, verbose=verbose)
        if models_mask is None:
            self.models_mask = [True]*ModelIdx.MAX_MODEL_IDX
        else:
            assert len(models_mask) == ModelIdx.MAX_MODEL_IDX, f"Size of models mask must be {ModelIdx.MAX_MODEL_IDX}."
            assert models_mask[ModelIdx.TRUE_MODEL_IDX], f"True model must be loaded!"
            self.models_mask = models_mask

    
    def build(self, indexes_list=None):
        #assert indexes_list is not None, f"[ ERROR ] Indexes list must not be None!"
        if indexes_list is None:
            return None


        models_prefix = list()
        models_prefix.append(self.dirs.get_true_models_prefix())
        models_prefix.append(self.dirs.get_highres_models_prefix())
        models_prefix.append(self.dirs.get_mediumres_models_prefix())
        models_prefix.append(self.dirs.get_lowres_models_prefix())
        models_prefix.append(self.dirs.get_water_mask_prefix())

        log.info(f"BUILDING TRUE MODEL FROM <{self.dirs.get_models_dirname()}>")

        m = list()
        for mask, prefix in zip(self.models_mask, models_prefix):
            if mask: 
                m_list = self._get_filelist(prefix, indexes_list)
                print(f"Models len = {len(m_list)}")
                print(f"File: {prefix}")
                m_aux = BuildModels(
                        m_filelist=m_list, 
                        verbose=self.verbose
                    ).build()
            else:
                m_aux = None
            m.append(m_aux)
        
        return DeepTomographyModels(models=m, verbose=self.verbose)

class BuildSupervisedData:
    """
    Builds an instance of SupervisedData.
    """
    def __init__(self, x_filelist=None, y_filelist=None, z_padding=0, loader:Load=NPLoad(), verbose=False):
        self.x_filelist = x_filelist
        self.y_filelist = y_filelist
        self.loader = loader
        self.z_padding = z_padding  # if we wanted to add more samples to make the size suitable for u-net's encoder-decoder.
        assert len(self.x_filelist) == len(self.y_filelist), f"[ ERRO ] X and Y must be same size!"

    def build(self):
        
        log.line()
        log.info(f"Reading supervised data...")
        log.info(f"\t\tNumber of supervised data pairs: {len(self.x_filelist)}")

        IDX_Z = 0
        
        x_aux = self.loader.load(self.x_filelist[0])
        y_aux = self.loader.load(self.y_filelist[0])

        x_shape = x_aux.shape
        x_shape = (x_shape[0] + self.z_padding,) + x_shape[1:]

        y_shape = y_aux.shape
        y_shape = (y_shape[0] + self.z_padding,) + y_shape[1:]

        X = np.zeros( (len(self.x_filelist),) + x_shape, dtype=float)
        Y = np.zeros( (len(self.y_filelist),) + y_shape, dtype=float)

        for i, xy_files in enumerate(zip(self.x_filelist,self.y_filelist)): 
            x_aux = self.loader.load(xy_files[0])
            y_aux = self.loader.load(xy_files[1])
            X[i,:x_shape[IDX_Z]-self.z_padding,:,:] = x_aux
            Y[i,:y_shape[IDX_Z]-self.z_padding,:] = y_aux

        # padding
        # repeat the last z
        if self.z_padding > 0:
            x_o = x_shape[IDX_Z]-self.z_padding
            y_o = y_shape[IDX_Z]-self.z_padding
            X[:,x_o:,:,:] = X[:,x_o-1:x_o,:,:]
            Y[:,y_o:,:]   = Y[:,y_o-1:x_o,:]
        
        return SupervisedData(X=X, Y=Y) 


class BuildDeepTomographyData(BaseBuilder):
    """
    Builds an instance of RawData.
    """
    #def __init__(self, dirs=None:ProjectDirs, verbose=False):
    def __init__(self, dirs:ProjectDirs=None, z_len_newer_data=None, max_samples=None,  verbose=False):
        super().__init__(dirs=dirs, z_len_newer_data=z_len_newer_data, max_samples=max_samples, verbose=verbose)

    def build(self, indexes_list=None):
        
        log.info(f"BUILDING RAWDATA FROM <{self.dirs.get_basedir()}>")

        if indexes_list is None:
            return None

        x_list = list()
        y_list = list()
        
        x_list.append(self._get_filelist(self.dirs.get_X_highres_prefix(), indexes_list))
        y_list.append(self._get_filelist(self.dirs.get_Y_highres_prefix(), indexes_list))
        
        x_list.append(self._get_filelist(self.dirs.get_X_mediumres_prefix(), indexes_list))
        y_list.append(self._get_filelist(self.dirs.get_Y_mediumres_prefix(), indexes_list))
        
        x_list.append(self._get_filelist(self.dirs.get_X_lowres_prefix(), indexes_list))
        y_list.append(self._get_filelist(self.dirs.get_Y_lowres_prefix(), indexes_list))
        
        ############################### 
        data = list()
        for x_filelist, y_filelist in zip(x_list, y_list):
            data.append(
                BuildSupervisedData(
                    x_filelist=x_filelist, 
                    y_filelist=y_filelist,
                    loader=NPLoad(),
                    z_padding=1,
                    verbose=self.verbose
                ).build()
            )
        
        return Data(data=data, verbose=self.verbose)


class BuildSupervisedDeeperData:
    """
    Builds an instance of SupervisedData.
    """
    #def __init__(self, dirs=None:ProjectDirs, verbose=False):
    def __init__(self, x_filelist=None, Y=None, loader:Load=NPLoad(), verbose=False):
        self.x_filelist = x_filelist
        self.Y = Y
        self.loader = loader
        self.verbose = verbose
        assert len(self.x_filelist) == Y.shape[0], f"[ ERRO ] X and Y must be same size!"

    def build(self):
        
        log.line()
        log.info(f"Reading supervised data...")
        log.info(f"\t\tNumber of supervised data pairs: {len(self.x_filelist)}")
        
        x_aux = self.loader.load(self.x_filelist[0])
        X = np.zeros( (len(self.x_filelist),) + x_aux.shape, dtype=float)
        for i, f in enumerate(self.x_filelist): 
            X[i] = self.loader.load(f)
        
        return SupervisedData(X=X, Y=self.Y) 


class BuildDeepTomographyDeeperData(BaseBuilder):
    """
    Builds an instance of RawData.
    """
    #def __init__(self, dirs=None:ProjectDirs, verbose=False):
    def __init__(self, dirs=None, data_mask=None, models=None, z_len_newer_data=None, max_samples=None, verbose=False):
        super().__init__(dirs=dirs, z_len_newer_data=z_len_newer_data, max_samples=max_samples, verbose=verbose)
        self.models = models
        if data_mask is None:
            self.data_mask = [True] * DataIdx.MAX_DATA_IDX
        else:
            assert len(data_mask) == DataIdx.MAX_DATA_IDX, f"Size of data mask must be {DataIdx.MAX_DATA_IDX}."
            self.data_mask = data_mask


    def build(self, indexes_list=None):

        if indexes_list is None:
            return None
        #assert indexes_list is not None or len(indexes_list) > 0, f"[ ERROR ] Indexes list must not be None or not have zero size!"
        
        log.info(f"BUILDING RAWDATA FROM <{self.dirs.get_basedir()}>")

        x_highres_list = self._get_filelist(self.dirs.get_X_highres_prefix(), indexes_list)
        x_mediumres_list = self._get_filelist(self.dirs.get_X_mediumres_prefix(), indexes_list)
        x_lowres_list = self._get_filelist(self.dirs.get_X_lowres_prefix(), indexes_list)

        ############################### 
        if self.data_mask[DataIdx.HIGHRES_DATA_IDX]:
            log.info(f"\tHIGHRES...")
            highres = BuildSupervisedDeeperData(
                x_filelist=x_highres_list, 
                Y=self.models.get_true_models().get_models() - self.models.get_highres_models().get_models(),
                loader=NPLoad(),
                verbose=self.verbose
                ).build()
        else: 
            log.printMessage(f"\tHIGHRES WILL NOT BE LOADED!")
            highres = None
        
        ############################### 
        if self.data_mask[DataIdx.MEDIUMRES_DATA_IDX]:
            log.info(f"\tMEDIUMRES...")
            mediumres = BuildSupervisedDeeperData(
                x_filelist=x_mediumres_list, 
                Y=self.models.get_highres_models().get_models() - self.models.get_mediumres_models().get_models(),
                loader=NPLoad(),
                verbose=self.verbose
                ).build()
        else:
            log.printMessage(f"\tMEDIUMRES WILL NOT BE LOADED!")
            mediumres = None
        
        ############################### 
        if self.data_mask[DataIdx.LOWRES_DATA_IDX]:
            log.info(f"\tLOWRES...")
            lowres = BuildSupervisedDeeperData(
                x_filelist=x_lowres_list, 
                Y=self.models.get_mediumres_models().get_models() - self.models.get_lowres_models().get_models(),
                loader=NPLoad(),
                verbose=self.verbose
                ).build()
        else:
            log.printMessage(f"\tLOWRES WILL NOT BE LOADED!")
            lowres = None
        
        #assert highres.get_nmb_of_elems() == mediumres.get_nmb_of_elems(), f"[ ERRO ] Highres data and mediumres data must have the same size!"
        #assert highres.get_nmb_of_elems() == lowres.get_nmb_of_elems(),    f"[ ERRO ] Highres data and lowres data must have the same size!"
        
        data = list() 
        data.append(highres)
        data.append(mediumres)
        data.append(lowres)

        return Data(data=data, verbose=self.verbose) 


class BuildDeepTomographyDataAndModels(BaseBuilder):
    """
    Builds an instance of Data (raw data) and DeepTomographyModels.
    """
    def __init__(self, dirs=None, models_mask=None, test_perc=0.2, z_len_newer_data=680, max_samples=None, seed=None, verbose=False):
        super().__init__(dirs=dirs, z_len_newer_data=z_len_newer_data, max_samples=max_samples, verbose=verbose)
        self.models = None
        self.data = None
        self.test_perc = test_perc
        self.seed = seed
        self.models_mask = models_mask
        if models_mask is not None:
            self.data_mask = (models_mask[ModelIdx.HIGHRES_MODEL_IDX], 
                              models_mask[ModelIdx.MEDIUMRES_MODEL_IDX],
                              models_mask[ModelIdx.LOWRES_MODEL_IDX],
                            )
        else:
            self.data_mask = None
    
    def build(self):
        v = self.verbose
        
        if self.is_newer_data():
            
            if v:
                log.printMessage(f"[ NEW DATA ] Building Deep Tomography models...")
            model_builder = BuildDeepTomographyModels(dirs=self.dirs, models_mask=self.models_mask, z_len_newer_data=self.z_len_newer_data, max_samples=self.max_samples, verbose=self.verbose)
            if v:
                log.line()
                log.line()
            train_indexes, test_indexes = model_builder.train_test_split(test_perc=self.test_perc,seed=self.seed)
            train_models = model_builder.build(indexes_list=train_indexes)
            if v:
                log.line()
                log.line()
            test_models = model_builder.build(indexes_list=test_indexes)
            assert test_models is not None or self.test_perc == 0., f"[ INTERNAL ERROR]"
            
            if v:
                log.printMessage(f"[ NEW DATA ] Building Deep Tomography data...")
            
            if v:
                log.line()
                log.line()
            data_builder = BuildDeepTomographyDeeperData(dirs=self.dirs, data_mask=self.data_mask, models=train_models, z_len_newer_data=self.z_len_newer_data, verbose=self.verbose, max_samples=self.max_samples)
            train_data = data_builder.build(indexes_list=train_indexes)
            
            if v:
                log.line()
                log.line()
            data_builder = BuildDeepTomographyDeeperData(dirs=self.dirs, data_mask=self.data_mask, models=test_models, z_len_newer_data=self.z_len_newer_data, verbose=self.verbose, max_samples=self.max_samples)
            test_data = data_builder.build(indexes_list=test_indexes)
            assert test_data is not None or self.test_perc == 0., f"[ INTERNAL ERROR]"
            
        else:
            
            if v:
                log.printMessage(f"[ OLD DATA ] No velocity models to build!")
            
            train_models = None
            test_models = None
            
            if v:
                log.printMessage(f"[ OLD DATA ] Building Deep Tomography data...")
                
            data_builder = BuildDeepTomographyData(dirs=self.dirs, z_len_newer_data=self.z_len_newer_data, verbose=self.verbose, max_samples=self.max_samples)
            train_indexes, test_indexes = data_builder.train_test_split(test_perc=self.test_perc, seed=self.seed)
            
            if v:
                log.line()
                log.line()
            train_data = data_builder.build(indexes_list=train_indexes)
            if v:
                log.line()
                log.line()
            test_data = data_builder.build(indexes_list=test_indexes)
            assert test_data is not None or self.test_perc == 0., f"[ INTERNAL ERROR]"


        if v:
            log.printMessage(f"[ END OF BUILDING PROCESS ]")
                
        return train_data, test_data, train_models, test_models, train_indexes, test_indexes


