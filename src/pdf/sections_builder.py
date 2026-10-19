"""@package local_classes.py
Module with general utility classes.
"""

import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
from datetime import datetime

from src.utl.logging import Log as log
from src.pdf.section import Section


class SectionsBuilder:
    def build_from_list(l=None):
        """
        Build a list of Section's objects recursively from a 'l'. 
        'l' is a list of dictionaries of form:
        [
            {
                'title' : "title_example",
                'text' : "text_example",
                'figures' : [ 
                                { 'filename' : "./fig1.png", 'caption' : "caption_example"},
                                { 'filename' : "./fig2.png", 'caption' : "caption_example"},
                            ],
                'subsections' : [ 
                                    {
                                        'title' : "....",
                                        'text' : "....", 
                                        'figures' : list({},{}),
                                        'subsections' : [
                                                            { 'title', etc.},
                                                            ...,
                                                            { 'title', etc.}
                                                        ]
                                    },
                                    ...,
                                    {

                                    }
                                ]
            },
            ...,
            {

            }
        ]
        """
        assert l is None or isinstance(l, list), f"Parameter 'l' must be None or of type 'list' -- {type(l)}"

        if l is None or len(l) == 0:
            secs = None
        else:
            secs = list()

            for v in l:
                s = Section( title=v.get('title'), text=v.get('text'), figures=v.get('figures'))
                s.add_subsections(subsections=SectionsBuilder.build_from_list(v.get('subsections')))
                secs.append(s)

        return secs



