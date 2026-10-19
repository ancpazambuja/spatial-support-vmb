"""@package local_classes.py
Module with general utility classes.
"""

import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
from datetime import datetime

from src.utl.logging import Log as log


class Section:
    def __init__(self, title=None, text=None, figures=None, subsections=None, verbose=False):
        #super().__init__()
        self.verbose=verbose
        self.title=title
        self.text=text
        self.figures=figures
        self.subsections=None
        self.add_subsections(subsections)
    
    def set_title(self, title=None):
        self.title = title

    def set_text(self, text=None):
        self.text = text

    def set_figures(self, figures=None):
        self.figures = figures

    def add_subsection(self, subsection=None):
        if subsection is not None:
            if self.subsections is None:
                self.subsections = list()
            self.subsections.append(subsection)

    def add_subsections(self, subsections=None):
        if subsections is not None:
            for s in subsections:
                self.add_subsection(s)


