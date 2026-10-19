"""@package rtm_proxy.py
Module responsible for running the rtm from a shell command line. 
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
from src.utl.classes import ShellCommand, Base


class RTMExe(Base):
    '''
    Represents the RTM command to run.
    '''

    def __init__(self, exe_filename=None, par_filename=None, input_model_filename=None,
        input_seismic_data_file=None, output_dir=None):
        super().__init__(name="RTMExe")
        self.exe_filename = exe_filename
        self.par_filename = par_filename
        self.input_model_filename = input_model_filename
        self.input_seismic_data_file = input_seismic_data_file
        self.output_dir = output_dir

        self.args = None
        self.result = None

        
    def make_args(self):
        args = list()
        args.append("-p")
        args.append(self.par_filename)
        args.append("1")
        return args

    def run(self, args=None):
        self.args = self.make_args(self.exe_filename)
        process = ShellCommand()

        


class RTMProjExe(Base):
    '''
    Represents the RTM command to run in a project view.
    '''
    def __init__(self, rtm_basedir=None, par_filename=None, resolution=None):
        super().__init__(name="RTMExe")
        self.rtm_basedir = rtm_basedir
        self.exe_filename = exe_filename
        self.par_filename = par_filename
        self.input_model_filename = input_model_filename
        self.input_seismic_data_file = input_seismic_data_file
        self.output_dir = output_dir

        self.args = None
        self.result = None

        
    def make_args(self):
        args = list()
        args.append("-p")
        args.append(self.par_filename)
        args.append("1")
        return args

    def run(self, args=None):
        self.args = self.make_args(self.exe_filename)
        process = ShellCommand()

        
