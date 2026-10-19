"""@package local_classes.py
Module with general utility classes.
"""
import os
import pprint

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from datetime import datetime

from pylatex import Document, Section, Subsection, Subsubsection, Figure, Command, Package, Ref, Label
from pylatex.utils import NoEscape

from src.utl.logging import Log as log
from src.utl.classes import Graph as graph
from src.visualization.gs_viewer import GSViewer as gsviewer

def df2latex(
    df=None, caption='No caption', sort_column=None, columns=None, 
    columns_labels=None, n=None, label=None):

    df = df.sort_values(by=sort_column)
    # The code below didn't work due (I suppose) to special charatecters and NoEscape issues.
    #sorted_string = sorted_lines[columns_to_report].to_latex(index=False, escape=False)
    df = df[columns].head(n)

    latex_string = ''

    if label is not None:
        latex_string += f"See table '{caption}'." 
        latex_string += r"\ref {" + f"{label}" + r"}."
        latex_string += f"\n"

    latex_string += r"\begin{table}[h!] "
#        latex_string += r"\centering "
    latex_string += r"\begin{tabularx}{\textwidth}{|X|X|X|X|X|X|X|} "
    latex_string += f"\n"
    latex_string += r"\hline "
    latex_string += f"\n"
    for i, c in enumerate(columns_labels):
        _c = c.replace("_", "\_")
        if i == 0:
            latex_string += r" \textbf{" + _c + "} "
        else:
            latex_string += r" & \textbf{" +  _c + r"} "
    latex_string += r" \\ "
    latex_string += f"\n"
    latex_string += r"\hline "
    latex_string += f"\n"
    for index, row in df.iterrows():
        for i, v in enumerate(row.values):
            if i == 0:
                latex_string += f" {v:.5g} "
            else:
                latex_string += f" & {v:.5g} "
        latex_string += r" \\ "
        latex_string += f"\n"
    latex_string += r"\hline "
    latex_string += f"\n"
    latex_string += r"\end{tabularx} " #+ "\n"
    latex_string += f"\n"
    latex_string += r"\caption{" + f"{caption}" + r"} "
    if label is not None:
        latex_string += f"\n"
        latex_string += r"\label{" + f"{label}" + r"} "
    latex_string += f"\n"
    latex_string += r"\end{table}"
    latex_string += f"\n"

    return latex_string

def clean_label(fname):
    fig_label = fname.replace('./', '')
    fig_label = fig_label.replace(' ', '')
    fig_label = fig_label.replace('_', '')
    fig_label = fig_label.replace('.png', '')
    return fig_label

def clean_filename(fname):
    fname = fname.replace(' ', '_')
    return fname

class GSUtils:
    def make_countplot(df=None, x_name=None, save=False, fname=None, dark_background=False, title='Missed Title'):
        fig_label = clean_label(fname)
        fname = clean_filename(fname)

        figure_list = [ {
                'filename' : fname,
                'caption' : f'Countplot for {x_name}.',
                'label' : f"fig:{fig_label}",
            } ] 
        gsviewer.countplot(
            df=df, x_name=x_name, save=True, fname=fname, title=title, dark_background=dark_background
            )
        return figure_list

    def make_jointplot(df=None, x_name=None, y_name=None, save=False, fname=None, dark_background=False,
        title='Missed Title', kind='kde', hue=None):
        fig_label = clean_label(fname)
        fname = clean_filename(fname)

        figure_list = [ {
                'filename' : fname,
                'caption' : f'Joint plot for {x_name}.',
                'label' : f"fig:{fig_label}",
            } ]

        gsviewer.jointplot(
            df=df, x_name=x_name, y_name=y_name, save=True, 
            fname=fname, title=title, kind=kind, hue=hue, dark_background=dark_background
            )
        return figure_list

    def make_boxplot(df=None, x_name=None, y_name=None, save=False, 
                     fname=None, title='Missed Title', count=False,
                     hue=None, palette=None, dark_background=False
                     ):
        fig_label = clean_label(fname)
        fname = clean_filename(fname)
        caption=f'Boxplot for {x_name}'
        if hue is not None:
            caption=caption + f' and hue={hue}'
        caption=caption + '.'

        figure_list = [ {
                'filename' : fname,
                'caption' : caption,
                'label' : f"fig:{fig_label}",
            } ]

        gsviewer.boxplot(
            df=df, x_name=x_name, y_name=y_name, save=True, 
            fname=fname, title=title, count=count, 
            palette=palette, hue=hue, dark_background=dark_background, 
            )
        return figure_list

    def make_violin(df=None, x_name=None, y_name=None, save=False, 
                     fname=None, title='Missed Title', count=False,
                     hue=None, palette=None, dark_background=False,
                     y_order=None, x_label=None, y_label=None, errorbar_callback=None
                     ):
        fig_label = clean_label(fname)
        fname = clean_filename(fname)
        caption=f'Violin for {x_name}'
        if hue is not None:
            caption=caption + f' and hue={hue}'
        caption=caption + '.'

        figure_list = [ {
                'filename' : fname,
                'caption' : caption,
                'label' : f"fig:{fig_label}",
            } ]

        gsviewer.violin(
            #df=df.reset_index().rename(columns={'index': 'arch_name'}), x_name=x_name, y_name=y_name, save=True, 
            df=df, x_name=x_name, y_name=y_name, save=True, errorbar_callback=errorbar_callback,
            fname=fname, title=title, y_order=y_order, x_label=x_label, y_label=y_label
            )
        return figure_list

    def make_scatterplot(df=None, ax=None, x_name=None, y_name=None, save=False,
        fname=None, title='Missed Title', hue=None, hue_norm=None, palette=None, size=None, sizes=None, 
        dark_background=False, median_lines=False):
        fig_label = clean_label(fname)
        fname = clean_filename(fname)

        figure_list = [ {
                'filename' : fname,
                'caption' : f'Scatter plot for {x_name}.',
                'label' : f"fig:{fig_label}",
            } ]

        gsviewer.scatterplot(
            df=df, ax=ax, x_name=x_name, y_name=y_name, save=save, 
            fname=fname, title=title, hue=hue, hue_norm=hue_norm, palette=palette, dark_background=dark_background,
            median_lines=median_lines, size=size, sizes=sizes,
            )
        return figure_list

    def make_heatplot(ax=None, df=None, fname=None, title='Missed Title', 
                      annot=False, save=False, vmin=None, vmax=None, mask=None, **kwargs):
        fig_label = clean_label(fname)
        fname = clean_filename(fname)

        figure_list = [ {
                'filename' : fname,
                'caption' : f'Heatmap.',
                'label' : f"fig:{fig_label}",
            } ]

        gsviewer.heatmap(ax=ax, df=df, title=title, fname=fname, 
            save=save, annot=annot, vmin=vmin, vmax=vmax, mask=mask, **kwargs)

        return figure_list

    def make_lineplot(ax=None, df=None, x_name=None, y_name=None, save=False,
        fname=None, title='Missed Title', hue=None, palette=None, 
        style=None, markers=None, scatter=False, dark_background=False,
        ):
        fig_label = clean_label(fname)
        fname = clean_filename(fname)

        figure_list = [ {
                'filename' : fname,
                'caption' : f'Line plot for {x_name}.',
                'label' : f"fig:{fig_label}",
            } ]

        gsviewer.lineplot(
            ax=ax, df=df, x_name=x_name, y_name=y_name, save=True, markers=markers,
            fname=fname, title=title, hue=hue, palette=palette, scatter=scatter,
            dark_background=dark_background, style=style
            )
        return figure_list


    @staticmethod
    def make_lineplot_errorbar(ax=None, df=None, x_name=None, y_name=None,
                               y_lower_name=None, y_upper_name=None,
                               save=False, fname=None, title='Missed Title',
                               hue=None, palette=None,
                               dark_background=False,
                               line_color='blue',
                               error_bar_color='red',
                               capsize=4,
                               hue_offset=0.3,
                               xtick_labels=None):
    
        fig_label = clean_label(fname)
        fname = clean_filename(fname)
    
        figure_list = [{
            'filename': fname,
            'caption': f'Line plot with CI error bars for {x_name}.',
            'label': f"fig:{fig_label}",
        }]
    
        gsviewer.lineplot_errorbar(
            ax=ax, df=df, x_name=x_name, y_name=y_name,
            y_lower_name=y_lower_name, y_upper_name=y_upper_name,
            save=True, fname=fname, title=title,
            hue=hue, hue_offset=hue_offset, palette=palette,
            dark_background=dark_background,
            line_color=line_color,
            error_bar_color=error_bar_color,
            capsize=capsize,
            xtick_labels=xtick_labels,
        )
    
        return figure_list


    def make_fillplot(df=None, save=False, fname=None, title='Missed Title',
            min_name=None, max_name=None, mean_name=None, palette=None, fill_label=None,
        ):
        fig_label = clean_label(fname)
        fname = clean_filename(fname)

        figure_list = [ {
                'filename' : fname,
                'caption' : f'{title}. Fill plot.',
                'label' : f"fig:{fig_label}",
            } ]

        gsviewer.fillplot(
            df=df, save=save, fname=fname, title=title,
            min_name=min_name, max_name=max_name, mean_name=mean_name,
            palette=palette, fill_label=fill_label,
            )
        return figure_list

    def make_pairplot(df=None, save=False, fname=None, title='Missed Title',
            x_vars=None, y_vars=None, kind=None, palette=None,
            hue=None, diag_kind=None, corner=None, caption=""
        ):
        fig_label = clean_label(fname)
        fname = clean_filename(fname)

        figure_list = [ {
                'filename' : fname,
                'caption' : f'{title}. Pair plot.',
                'label' : f"fig:{fig_label}",
            } ]

        gsviewer.pairplot(
            df=df, save=True, fname=fname, title=title,
            hue=hue, palette=palette, kind=kind, diag_kind=diag_kind,
            x_vars=x_vars, y_vars=y_vars, corner=corner
            )
        return figure_list

    def make_pairgrid(df=None, save=False, fname=None, title='Missed Title',
            x_vars=None, y_vars=None, map=None, palette=None,
            hue=None, corner=None
        ):
        fig_label = clean_label(fname)
        fname = clean_filename(fname)

        figure_list = [ {
                'filename' : fname,
                'caption' : f'{title}. Pair grid.',
                'label' : f"fig:{fig_label}",
            } ]

        gsviewer.pairgrid(
            df=df, save=True, fname=fname, title=title,
            hue=hue, palette=palette, map=map,
            x_vars=x_vars, y_vars=y_vars, corner=corner,
            )
        return figure_list
      
class GSListBase:
    def __init__(self, subsections=None, df=None, arch=None, dataset=None, label=None, verbose=False):
        self.subsections=subsections
        self.verbose=verbose
        self.df=df
        self.arch=arch
        self.dataset=dataset
        self.label=label
    
    def make(self):
        raise NotImplemented

class GSIntroduction(GSListBase):
    def __init__(self, subsections=None, df=None, arch=None, dataset=None, label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, arch=arch, dataset=dataset, label=label, verbose=verbose)

    def make(self):
        ngsruns = self.df['parent_run_name'].nunique()
        nruns = len(self.df)

        if self.verbose:
            log.info(f"[ {self.label} ] Nmb of global runs = {ngsruns}")
            log.info(f"[ {self.label} ] Nmb of elements = {nruns}")
    
        sec = [
            {
                'title' : "Introduction", 
                'text' :  NoEscape(
f"This report analyses the results of the hyperparameters \
tuning (\\textbf{{HT}}) in the context \
of \\textit{{Deep Tomography 2.0}} development.\
\n\n\
\\begin{{table}}[h!] \
\\centering \
\\begin{{tabular}}{{|c|c|}} \
\\hline \
\\textbf{{Network Architecture}} & {self.arch} \\\\ \
\\hline \
\\textbf{{Training Dataset}} & {self.dataset} \\\\ \
\\hline \
\\end{{tabular}} \
\\caption{{Report Scope}} \
\\end{{table}} \
\n\n\
\\begin{{table}}[h!] \
\\centering \
\\begin{{tabular}}{{|c|c|}} \
\\hline \
\\textbf{{HT Strategy}} & K-Fold CV \\\\ \
\\hline \
\\textbf{{K}} & 5 \\\\ \
\\hline \
\\textbf{{Nmb of HT Sets}} & {nruns} \\\\ \
\\hline \
\\textbf{{Nmf of Training Curves}}  & {nruns*5} \\\\ \
\\hline \
\\textbf{{Processor Type}} & 4xNVIDIA A100-SXM4-80GB  \\\\ \
\\hline \
\\end{{tabular}} \
\\caption{{General Information}} \
\\end{{table}} \
\n\n\
Three main sections follow: \
\\begin{{itemize}} \
    \\item global view; \
    \\item fixed kernel size; \
    \\item varying kernel sizes. \
    \\end{{itemize}} \
"),
                'subsections' : self.subsections,
#                'figures' : self.figures,
            }
        ]

        return sec

class GSGlobal(GSListBase):
    def __init__(self, subsections=None, df=None, arch=None, dataset=None, label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, arch=arch, dataset=dataset, label=label, verbose=verbose)

    def make(self):
        sec = [
            {
                'title' : "Global Analysis", 
                'text' :  NoEscape(
f"\
This section shows the attributes altogether in order to \
give us an hyperparameters search's overview. \
\n\n\
Analyzing the behavior of the error function by fixing one \
attribute at a time may give us some insights like: \
\n\n\
\\begin{{itemize}} \
    \\item understand how each hyperparameter individually affects \
    the model's performance; \
    \\item identify non-obvious trends and patterns; \
    \\item identify anomalies or outliers in the data possibly \
    related to certain hyperparameters settings; \
    \\item guide, if necessary, to fine-tuning narrowly on some attribute \
    neigbhorhood. \
\\end{{itemize}} \
"),
                'subsections' : self.subsections,
#                'figures' : self.figures,
            }
        ]
        return sec

class GSFixedKernelSize(GSListBase):
    def __init__(self, subsections=None, df=None, arch=None, dataset=None, label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, arch=arch, dataset=dataset, label=label, verbose=verbose)

    def make(self):
        sec = [
            {
                'title' : "Fixed Kernel Size", 
                'text' : NoEscape(
f"\
This section analyses the behaviour of the grid search while keeping \
fixed the kernel-size and varying other quatities like: \
\\begin{{itemize}} \
    \\item learning rate; \
    \\item optimizer weight decay (regularization);  \
    \\item SGD's batch size; \
    \\item attention's number of layers; \
\\end{{itemize}} \
"),
                'subsections' : self.subsections,
#                'figures' : self.figures,
            }
        ]
        return sec

class GSVaryingKernelSizes(GSListBase):
    def __init__(self, subsections=None, df=None, arch=None, dataset=None, label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, arch=arch, dataset=dataset, label=label, verbose=verbose)

    def make(self):
        sec = [
            {
                'title' : "Varying Kernel Sizes", 
                'text' : f"""xxxxx""",
                'subsections' : self.subsections,
#                'figures' : self.figures,
            }
        ]
        return sec

class GSConclusion(GSListBase):
    def __init__(self, subsections=None, df=None, arch=None, dataset=None, label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, arch=arch, dataset=dataset, label=label, verbose=verbose)

    def make(self):
        sec = [
            {
                'title' : "Conclusion", 
                'text' : f"""xxxxx""",
                'subsections' : self.subsections,
#                'figures' : self.figures,
            }
        ]
        return sec

class GSGlobalPlots(GSListBase):
    def __init__(self, title=None, text=None, subsections=None, df=None,
        x_name=None, y_name=None, label=None, corner=None, verbose=False
        ):
        super().__init__(subsections=subsections, df=df, label=label, verbose=verbose)
        self.title=title
        self.text=text
        self.x_name=x_name
        self.y_name=y_name
        self.corner=corner

    def make(self):
        scatter_filename=f"./{self.label}_global__{self.x_name}__scatterplot.png"
        scatter_figure = GSUtils.make_scatterplot(
            df=self.df, x_name=self.x_name, y_name=self.y_name, 
            hue='conv_kernel_size', palette=gsviewer.PLT_PALETTE_SET2,
            save=True, fname=scatter_filename, title=f'{self.label} - {self.x_name}')

        boxplot_filename=f"./{self.label}_global__{self.x_name}_{self.y_name}__boxplot.png"
        boxplot_figure = GSUtils.make_boxplot(
            df=self.df, x_name=self.x_name, y_name=self.y_name, count=True,
            save=True, fname=boxplot_filename, title=f'{self.label} - {self.x_name}')
        
        boxplot2_filename=f"./{self.label}_global__{self.x_name}_{self.y_name}__boxplot_2.png"
        boxplot2_figure = GSUtils.make_boxplot(
            df=self.df, x_name=self.x_name, y_name=self.y_name, count=True,
            save=True, fname=boxplot2_filename,
            hue='conv_kernel_size', palette=gsviewer.PLT_PALETTE_SET2,
            title=f'{self.label} - {self.x_name}')

        sec = [
            {
                'title' : self.title, 
                'text' : self.text, 
                'subsections' : self.subsections,
                'figures' : scatter_figure + boxplot_figure + boxplot2_figure,
            }
        ]
        return sec


# graficos globais
## boxplot para cada grandeza
## curva para cada grandeza

# fixed kernel-size
## lr
### boxplot para cada ks
### curva para cada ks (melhor lr)
## wd
### boxplot para cada ks
### curva para cada ks (melhor lr)
## num_layers
### boxplot para cada ks
### curva para cada ks (melhor lr)

# fixed kernel-size and fixed num_layers
# REPETIR ACIMA


# views
## UNET
### fixed kernel_size
#### bar 
##### lr 
##### weight_decay (wd)
##### num_layers
#### curve 
##### lr 
##### weight_decay (wd)
##### num_layers
### global 
#### heatmap of lr x wd
#### heatmap of lr x lr x wd
### VARYING kernel_size
#### curve 
##### kernel_size with lr = best lr from grid search
## TransUNET
class GSListBuilder:
    def build_from_df(df=None, arch=None, dataset=None, label=None, verbose=False):

########## SECTION 1
        s12_subsections = list()
        s12_subsections += GSBestModelsAll(df=df, label=label, n=40, verbose=verbose).make()
        s12_subsections += GSBestModelsPerKernelSize(df=df, label=label, verbose=verbose).make()

        s1_subsecs = list()
        s1_subsecs = s1_subsecs + GSParametersOverview(
                df=df, label=label, verbose=verbose, corner=True,
            ).make()
        s1_subsecs = s1_subsecs + GSBestModels(
                df=df, label=label, verbose=verbose, subsections=s12_subsections
            ).make()
        s1_subsecs += GSSequenceLines(df=df, label=label, 
                x_label='LR', y_label='WDecay', z_label='CK Size', fxyz_label='MSE Mean',
                x_name='lr', y_name='weight_decay', z_name='conv_kernel_size', fxyz_name='mean_valid_score',
                verbose=verbose
            ).make()
        s1_subsecs += GSSequenceLines(df=df, label=label, 
                x_label='LR', y_label='WDecay', z_label='CK Size', fxyz_label='MSE Mean',
                x_name='lr', y_name='weight_decay', z_name='conv_kernel_size', fxyz_name='mean_valid_score',
                verbose=verbose, n=60
            ).make()
        s1_subsecs += GSSequenceHeatMap(df=df, label=label, 
                x_label='LR', y_label='WDecay', z_label='CK Size', fxyz_label='MSE Mean',
                x_name='lr', y_name='weight_decay', z_name='conv_kernel_size', fxyz_name='mean_valid_score',
                verbose=verbose
            ).make()
        s1_subsecs += GSSequenceHeatMap(df=df, label=label, 
                x_label='LR', y_label='WDecay', z_label='Att NLayers', fxyz_label='MSE Mean',
                x_name='lr', y_name='weight_decay', z_name='att_num_layers', fxyz_name='mean_valid_score',
                verbose=verbose
            ).make()

        s1 = GSIntroduction(
                df=df, arch=arch, 
                dataset=dataset, label=label, verbose=verbose,
                subsections=s1_subsecs,
            ).make()


########## SECTION 2
        s2_subsecs = list()
        s2_subsecs = s2_subsecs + GSGlobalOverview(df=df, label=label, verbose=verbose).make()
        s2_subsecs = s2_subsecs + GSGlobalLearningRate(df=df, label=label, verbose=verbose).make()
        s2_subsecs = s2_subsecs + GSGlobalWeightDecay(df=df, label=label, verbose=verbose).make()
        s2_subsecs = s2_subsecs + GSGlobalConvolutionKernelSize(df=df, label=label, verbose=verbose).make()
        if arch != 'Unet':
            s2_subsecs = s2_subsecs + GSGlobalAttentionNumberOfLayers(df=df, label=label, verbose=verbose).make()
        s2_subsecs = s2_subsecs + GSGlobalMiniBatchSize(df=df, label=label, verbose=verbose).make()


        s2 = GSGlobal(
                df=df, arch=arch, 
                dataset=dataset, label=label, verbose=verbose,
                subsections= s2_subsecs
            ).make()



########## SECTION 2

#        s212 = GSFKSWeightDecay(df=df, label=label, subsections=None, 
#                verbose=verbose).make()
#
#        s22= GSFKSBatchSize(df=df, label=label, subsections=None, 
#                verbose=verbose).make()
#
#        s23 =  GSFKSAttNumberOfLayers(df=df, label=label, subsections=None, 
#                verbose=verbose).make()

        s3_subsecs = list()
        s3_subsecs = s3_subsecs + GSFKSLearningRate(df=df, label=label, verbose=verbose).make()
        s3_subsecs = s3_subsecs + GSFKSWeightDecay(df=df, label=label, verbose=verbose).make()
        s3_subsecs = s3_subsecs + GSFKSConvolutionKernelSize(df=df, label=label, verbose=verbose).make()
        if arch != 'Unet':
            s3_subsecs = s3_subsecs + GSFKSAttentionNumberOfLayers(df=df, label=label, verbose=verbose).make()
        s3_subsecs = s3_subsecs + GSFKSMiniBatchSize(df=df, label=label, verbose=verbose).make()

        s3 = GSFixedKernelSize(
                df=df, arch=arch, 
                dataset=dataset, label=label, verbose=verbose,
                subsections=s3_subsecs,
            ).make()


        s4 = GSVaryingKernelSizes(
                df=df, arch=arch, 
                dataset=dataset, label=label, verbose=verbose,
            ).make()

        s5 = GSConclusion(
                df=df, arch=arch, 
                dataset=dataset, label=label, verbose=verbose,
            ).make()

        return s1 + s2 + s3 + s4 + s5 

class GSPDFBuilder:
    def __make_title(doc=None, title=None, author=None, verbose=False):

        # Get the current date and time
        #current_datetime = datetime.now()
        #formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M")
        # Define a new LaTeX command for the date and time

        doc.preamble.append(Command('title', title))
        doc.preamble.append(Command('author', author))
        doc.preamble.append(Command('date', NoEscape(r'\today')))
        doc.append(NoEscape(r'\maketitle'))
        # Add the command definition to the document
        #doc.preamble.append(Command('currentdatetime', arguments=formatted_datetime))
        #doc.append(NoEscape(r"The current date and time is: \currentdatetime"))
        # Use the command in the document body

    def __make_table_of_contents(doc=None, verbose=False):
        if verbose:
            log.info("Adding the table of contents!")
        # Add the Table of Contents
        doc.append(Command('tableofcontents'))
        doc.append(Command('newpage'))

    def __add_sections_to_pdf(doc=None, sections=None, depth=0, verbose=False):
        if verbose:
            log.info(f"INSIDE ADD_SECTIONS_TO_PDF")
            log.info(f"sections = {sections}")
        for s in sections:

            if depth==0:
                sec_class = Section
            elif depth==1:
                sec_class = Subsection
            elif depth==2:
                sec_class = Subsubsection
            else:
                raise ValueError("'depth' parameter should be in [0,2]")

            with doc.create(sec_class(s.title)) as sec:
                log.info(f"s['text'] = {s.text} ")
                sec.append(s.text)
                if s.figures is not None:
                    figures = s.figures
                    for f in figures:
                        sec.append(f"\n\nSee figure {f['caption']} ")
                        sec.append(Ref(f['label']))
                        sec.append(".")
                    for f in figures:
                        with sec.create(Figure(position='h!')) as plot:
                            plot.add_image(f['filename'], width=NoEscape(r'0.8\textwidth'))
                            plot.add_caption(f['caption'])
                            plot.append(Label(f['label']))
                if s.subsections is not None:
                    GSPDFBuilder.__add_sections_to_pdf(doc=sec, sections=s.subsections, depth=depth+1)
                    
    def build_from_sections(sections=None, title=None, author=None, verbose=False):
        doc = Document()

        # Add non-default packages
        doc.packages.append(Package('graphicx'))
        doc.packages.append(Package('tabularx'))
        doc.packages.append(Package('hyperref'))
        doc.packages.append(Package('booktabs'))
        doc.packages.append(Package('bookmark'))
        

        # Add a title
        GSPDFBuilder.__make_title(doc=doc, title=title, author=author, verbose=verbose)
        GSPDFBuilder.__make_table_of_contents(doc=doc, verbose=verbose)
        GSPDFBuilder.__add_sections_to_pdf(doc=doc, sections=sections, verbose=verbose)
        return doc

###########################################
###########################################
###########################################

class GSGlobalOverview(GSListBase):
    def __init__(self, subsections=None, df=None, label=None, corner=False, verbose=False):
        super().__init__(subsections=subsections, df=df, label=label, verbose=verbose)
        self.corner = corner

    def make(self):
        x_columns = [
            'lr', 'weight_decay', 'conv_kernel_size',
            'minibatch_size', 'att_num_layers'
        ]
        y_columns = [
             'std_valid_score', 'mean_valid_score'
        ]

        pairplot_filename=f"./{self.label}_global_overview_pairplot.png"
        pairplot_figure = GSUtils.make_pairplot(
            df=self.df, save=True, 
            x_vars=x_columns, y_vars=y_columns, 
            fname=pairplot_filename, title=f'{self.label} - Global Analysis Overview',
            kind='scatter', diag_kind=None, corner=self.corner, 
            hue='conv_kernel_size', palette=gsviewer.PLT_PALETTE_SET2)

        pairgrid_filename=f"./{self.label}_global_overview_pairgrid_boxplot.png"
        pairgrid_figure = GSUtils.make_pairgrid(
            df=self.df, save=True, 
            x_vars=x_columns, y_vars=y_columns, 
            fname=pairgrid_filename, title=f'{self.label} - Global Analysis Overview',
            map=sns.boxplot, corner=self.corner, 
            hue=None, palette=None)
            #hue='conv_kernel_size', palette=gsviewer.PLT_PALETTE_SET2) # SEPARATE BY HUE

        sec = [
            {
                'title' : "Global Analysis Overview", 
                'text' : NoEscape(
f"This section shows all the attributes in a single view. \
"), 
                'subsections' : self.subsections,
                'figures' : pairplot_figure + pairgrid_figure
            }
        ]
        return sec

class GSGlobalLearningRate(GSListBase):
    def __init__(self, subsections=None, df=None, label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, label=label, verbose=verbose)

    def make(self):
        delegated=GSGlobalPlots(
            title='Learning Rate', 
            text=NoEscape(
f"This section shows the behaviour of the model for each learning rate used. \
"), 
            subsections=self.subsections,
            df=self.df, x_name='lr', y_name='mean_valid_score', 
            label=self.label, verbose=self.verbose
        )

        return delegated.make()

class GSGlobalWeightDecay(GSListBase):
    def __init__(self, subsections=None, df=None, label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, label=label, verbose=verbose)

    def make(self):
        delegated=GSGlobalPlots(
            title='Optimizer Weight Decay', 
            text=NoEscape(
f"This section shows the behaviour of the model for each optimizer weight decay used. \
"), 
            subsections=self.subsections,
            df=self.df, x_name='weight_decay', y_name='mean_valid_score', 
            label=self.label, verbose=self.verbose
        )

        return delegated.make()

class GSGlobalConvolutionKernelSize(GSListBase):
    def __init__(self, subsections=None, df=None, label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, label=label, verbose=verbose)

    def make(self):
        delegated=GSGlobalPlots(
            title='Convolution Kernel Size', 
            text=NoEscape(
f"This section shows the behaviour of the model for each convolution kernel size used. \
"), 
            subsections=self.subsections,
            df=self.df, x_name='conv_kernel_size', y_name='mean_valid_score', 
            label=self.label, verbose=self.verbose
        )

        return delegated.make()

class GSGlobalAttentionNumberOfLayers(GSListBase):
    def __init__(self, subsections=None, df=None, label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, label=label, verbose=verbose)

    def make(self):
        delegated=GSGlobalPlots(
            title='Attention Number of Layers', 
            text=NoEscape(
f"This section shows the behaviour of the model for each attention number of layers used. \
"), 
            subsections=self.subsections,
            df=self.df, x_name='att_num_layers', y_name='mean_valid_score', 
            label=self.label, verbose=self.verbose
        )

        return delegated.make()


class GSParametersOverview(GSListBase):
    def __init__(self, subsections=None, df=None, label=None, corner=None, verbose=False):
        super().__init__(subsections=subsections, df=df, label=label, verbose=verbose)
        self.corner = corner

    def make(self):
        pairplot_columns = [
            'lr', 'weight_decay', 'conv_kernel_size',
            'minibatch_size', 'att_num_layers'
        ]

        pairplot_filename=f"./{self.label}_global__pairplot.png"

        title = f"{self.label}" + " - Grid Parameters Overview"

        pairplot_figure = GSUtils.make_pairplot(
            df=self.df, save=True, fname=pairplot_filename, 
            title=f'{title}', kind='scatter', diag_kind=None,
            corner=self.corner, x_vars=pairplot_columns, y_vars=pairplot_columns,
            hue='mean_valid_score', palette="coolwarm"
            )
        
        sec = [
            {
                'title' : 'Grid Parameters Overview', 
                'text' : NoEscape(
f"This section shows the discretization used for the grid search. \
"), 
                'subsections' : self.subsections,
                'figures' : pairplot_figure,
            }
        ]
        return sec

class GSBestModels(GSListBase):
    def __init__(self, subsections=None, df=None, label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, label=label, verbose=verbose)

    def make(self):

        latex_string = f"This section shows a brief summary of the grid search results."

        sec = [
            {
                'title' : 'Results Summary', 
                'text' : NoEscape(latex_string),
                'subsections' : self.subsections,
                'figures' : None,
            }
        ]
        return sec

class GSBestModelsAll(GSListBase):
    def __init__(self, subsections=None, df=None, label=None, n=10, verbose=False):
        super().__init__(subsections=subsections, df=df, label=label, verbose=verbose)
        self.n = n

    def make(self):
        scatterplot_filename=f"./{self.label}_best_models_all__scatterplot.png"

        df_aux = self.df.sort_values(by='mean_valid_score').head(self.n)

        scatterplot_figure = GSUtils.make_scatterplot(
            df=df_aux, save=True, fname=scatterplot_filename, 
            title=f'{self.label} - conv_kernel_size',
            x_name='conv_kernel_size', y_name='mean_valid_score',
            hue='std_valid_score', palette="seismic", dark_background=True
        )

        columns_to_report = [
            'mean_valid_score', 'std_valid_score',
            'conv_kernel_size', 'lr', 'weight_decay',
            'minibatch_size', 'att_num_layers' 
        ]

        columns_labels = [
            'MSE Mean', 'MSE Std',
            'CK Size', 'LR', 'WDecay',
            'MB Size', 'Att NLayers' 
        ]


        latex_string = df2latex(
            df=self.df, caption='Overall Best Models', label='tab:OverallBestModels', sort_column='mean_valid_score',
            columns=columns_to_report, columns_labels=columns_labels, n=self.n
        )

        sec = [
            {
                'title' : 'Overall Best Models', 
                'text' : NoEscape(latex_string),
                #'text' : 'Teste', 
                'subsections' : self.subsections,
                'figures' : scatterplot_figure,
            }
        ]
        return sec

class GSBestModelsPerKernelSize(GSListBase):
    def __init__(self, subsections=None, df=None, label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, label=label, verbose=verbose)

    def make(self):
        lineplot_filename=f"./{self.label}_best_models__lineplot.png"

        df_aux = self.df.loc[self.df.groupby('conv_kernel_size')['mean_valid_score'].idxmin()]

        lineplot_figure = GSUtils.make_lineplot(
            df=df_aux, save=True, fname=lineplot_filename, 
            title=f'{self.label} - conv_kernel_size', scatter=True,
            x_name='conv_kernel_size', y_name='mean_valid_score',
#            hue='std_valid_score', palette="coolwarm",
        )

        columns_to_report = [
            'mean_valid_score', 'std_valid_score',
            'conv_kernel_size', 'lr', 'weight_decay',
            'minibatch_size', 'att_num_layers' 
        ]

        columns_labels = [
            'MSE Mean', 'MSE Std',
            'CK Size', 'LR', 'WDecay',
            'MB Size', 'Att NLayers' 
        ]


        latex_string = df2latex(
            df=df_aux, caption='Best Models Per Conv Kernel Size', sort_column='mean_valid_score',
            columns=columns_to_report, columns_labels=columns_labels, n=None,
            label="tab:BestModelsPerCKSize"
        )

        sec = [
            {
                'title' : 'Best Models Per Conv Kernel Size', 
                'text' : NoEscape(latex_string),
                #'text' : 'Teste', 
                'subsections' : self.subsections,
                'figures' : lineplot_figure,
            }
        ]
        return sec

class GSSequenceLines(GSListBase):
    def __init__(self, subsections=None, df=None, label=None, verbose=False, 
        x_label=None, y_label=None, z_label=None, fxyz_label=None,
        x_name=None, y_name=None, z_name=None, fxyz_name=None, n=None
        ):
        super().__init__(subsections=subsections, df=df, label=label, verbose=verbose)
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
        self.fxyz_label = fxyz_label
        self.x_name = x_name
        self.y_name = y_name
        self.z_name = z_name
        self.fxyz_name = fxyz_name
        self.n = n

    def make(self):
        #getting min and max of the first 40th best.
        df_aux = self.df.sort_values(by=self.fxyz_name).head(40)
        vmin=df_aux['mean_valid_score'].min()
        vmax=df_aux['mean_valid_score'].max()

        title = f"Sequence of Lines"
        df_all = self.df
        if self.n is not None:
            df_all = df_all.sort_values(by=self.fxyz_name).head(self.n)
            title = title + f" -- N={self.n}"

        # Group by LR
        df_aux_all = df_all.loc[df_all.groupby([self.x_name, self.y_name, self.z_name])[self.fxyz_name].idxmin()]
        x_values = df_aux_all[self.x_name].unique()
        y_values = df_aux_all[self.y_name].unique()


        NX = 5
        NY = 3
        fig = plt.figure(figsize=(graph.PLOT_SX*2,graph.PLOT_SY*4), constrained_layout=True)
        #assert len(x_values)*len(y_values) <= NX*NY, f"Size of gridspec isn't enough. nx = {NX}, ny = {NY}"
        gs = fig.add_gridspec(NX,NY)

# Create a figure with three subplots
#fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        #figure_list = list()
        idx=0
        for i,x in enumerate(x_values):
            for j,y in enumerate(y_values):

                df_aux = df_aux_all[ (df_aux_all[self.x_name] == x) & (df_aux_all[self.y_name] == y) ]
                if df_aux is None or df_aux.shape[0] == 0:
                    continue
    
                if df_aux[(df_aux[self.x_name] == x) & (df_aux[self.y_name] == y)][self.fxyz_name].min() > vmin * 1.5:
                    continue

                ii = idx//NY
                jj = idx%NY
                assert idx < NX*NY, f"Array out of bounds!"
                idx = idx + 1

                ax = fig.add_subplot(gs[ii,jj])
    
                if self.n is not None:
                    plot_filename=f"./{self.label}_lineplot_view_{x}_{y}_{self.fxyz_name}_lineplot_sequencial.png"
                else:
                    plot_filename=f"./{self.label}_lineplot_view_{x}_{y}_{self.fxyz_name}_lineplot_sequencial_{self.n}.png"
                plot_figure = GSUtils.make_lineplot( ax=ax, df=df_aux, save=False, 
                        x_name=self.z_name, y_name=self.fxyz_name, markers=True,
                        fname=plot_filename, title=f'{x},{y}', scatter=True,
                        #hue=self.x_name, palette=gsviewer.PLT_PALETTE_SET2,
                        #style=self.y_name
                    )
                ax.set_title(f"{x},{y}")
    
        #plt.tight_layout()
        if self.n is None:
            globalplot_filename=f"./{self.label}_sequence_of_lines_view_{self.x_name}_{self.y_name}_{self.z_name}_{self.fxyz_name}_lineplot.png"
        else:
            globalplot_filename=f"./{self.label}_sequence_of_lines_view_{self.x_name}_{self.y_name}_{self.z_name}_{self.fxyz_name}_lineplot_{self.n}.png"
        fig_label = clean_label(globalplot_filename)
        fname = clean_filename(globalplot_filename)
        plt.savefig(fname)

        figure_list = [ {
                'filename' : fname,
                'caption' : f'Multiple lines.',
                'label' : f"fig:{fig_label}",
            } ]

        sec = [
            {
                'title' : title,
                'text' : NoEscape(
f"This section shows the lineplot for fixed {self.x_label} and {self.y_label} and free  {self.z_label}) \
"), 
                'subsections' : self.subsections,
                'figures' : figure_list
            }
        ]
        return sec

class GSSequenceHeatMap(GSListBase):
    def __init__(self, subsections=None, df=None, label=None, verbose=False, 
        x_label=None, y_label=None, z_label=None, fxyz_label=None,
        x_name=None, y_name=None, z_name=None, fxyz_name=None
        ):
        super().__init__(subsections=subsections, df=df, label=label, verbose=verbose)
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
        self.fxyz_label = fxyz_label
        self.x_name = x_name
        self.y_name = y_name
        self.z_name = z_name
        self.fxyz_name = fxyz_name

    def make(self):

        #getting min and max of the first 40th best.
        df_aux = self.df.sort_values(by=self.fxyz_name).head(40)
        vmin=df_aux['mean_valid_score'].min()
        vmax=df_aux['mean_valid_score'].max()
        # Group by LR
        df_aux_all = self.df.loc[self.df.groupby([self.x_name, self.y_name, self.z_name])[self.fxyz_name].idxmin()]
        z_values = df_aux_all[self.z_name].unique()


        fig = plt.figure(figsize=(graph.PLOT_SX*2,graph.PLOT_SY*4), constrained_layout=True)
        gs = fig.add_gridspec(len(z_values), 1)

# Create a figure with three subplots
#fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        #figure_list = list()
        for i,z in enumerate(z_values):
            df_aux = df_aux_all[df_aux_all[self.z_name] == z]
            df_aux = df_aux.pivot_table(values=self.fxyz_name, index=self.y_name, columns=self.x_name)

            ax = fig.add_subplot(gs[i,0])

            heatplot_filename=f"./{self.label}_heatmap_view_{self.x_name}_{self.y_name}_{self.z_name}_{z}_{self.fxyz_name}_heatmap.png"
            heatplot_figure = GSUtils.make_heatplot( ax=ax, vmin=vmin, vmax=vmax,
                df=df_aux, save=False, fname=heatplot_filename, annot=True,
                title=f'{self.label} Heatmap - {self.x_label}/{self.y_label}/({self.z_label}={z}',
            )
            ax.set_title(f"{self.z_label} {z}")
            #figure_list += heatplot_figure


        #plt.tight_layout()
        heatplot_filename=f"./{self.label}_heatmap_view_{self.x_name}_{self.y_name}_{self.z_name}_{self.fxyz_name}_heatmap.png"
        fig_label = clean_label(heatplot_filename)
        fname = clean_filename(heatplot_filename)
        plt.savefig(fname)

        figure_list = [ {
                'filename' : fname,
                'caption' : f'Heatmap.',
                'label' : f"fig:{fig_label}",
            } ]

        sec = [
            {
                'title' : f"Heatmaps for {self.x_label}/{self.y_label} - {self.z_name}",
                'text' : NoEscape(
f"This section shows the heatmap for {self.x_label} and {self.y_label} and fixed {self.z_label}) \
"), 
                'subsections' : self.subsections,
                'figures' : figure_list
            }
        ]
        return sec

class GS3DView(GSListBase):
    def __init__(self, subsections=None, df=None, sectitle=None, label=None, verbose=False, 
        x_label=None, y_label=None, z_label=None, fxyz_label=None,
        x_name=None, y_name=None, z_name=None, fxyz_name=None
        ):
        super().__init__(subsections=subsections, df=df, label=label, verbose=verbose)
        self.sectitle = sectitle
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
        self.fxyz_label = fxyz_label
        self.x_name = x_name
        self.y_name = y_name
        self.z_name = z_name
        self.fxyz_name = fxyz_name


    def make(self):
        # Group by LR
        #df_aux = self.df.loc[self.df.groupby('lr')['mean_valid_score'].idxmin()]
        df_aux = self.df.loc[self.df.groupby([self.x_name, self.y_name, self.z_name])[self.fxyz_name].idxmin()]

        scatterplot_filename=f"./{self.label}_3d_view_{self.x_name}_{self.y_name}_{self.z_name}_{self.fxyz_name}_scatterplot_3d.png"
        scatterplot_figure = GSUtils.make_scatterplot(
            df=df_aux, save=True, 
            x_name=self.x_name, y_name=self.y_name,
            fname=scatterplot_filename, title=f'{self.label} 3D - {self.x_label}/{self.y_label}/{self.z_label}',
            hue=self.z_name, size=self.fxyz_name, sizes=(0.02, 0.04), palette=gsviewer.PLT_PALETTE_SET2,
            )

        sec = [
            {
                'title' : NoEscape(self.sectitle),
                'text' : NoEscape(
f"This section shows 3D view of f({self.x_label},{self.y_label},{self.z_label})={self.fxyz_label} \
"), 
                'subsections' : self.subsections,
                'figures' : scatterplot_figure
            }
        ]
        return sec

class GSGlobalMiniBatchSize(GSListBase):
    def __init__(self, subsections=None, df=None, label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, label=label, verbose=verbose)

    def make(self):
        delegated=GSGlobalPlots(
            title='Minibatch Size', 
            text=NoEscape(
f"This section shows the behaviour of the model for each minibatch size used. \
"), 
            subsections=self.subsections,
            df=self.df, x_name='minibatch_size', y_name='mean_valid_score', 
            label=self.label, verbose=self.verbose
        )

        return delegated.make()





###########################################
###########################################
###########################################

class GSFKSBase(GSListBase):
    def __init__(self, subsections=None, df=None, sectitle=None, with_hue=True, 
        label=None, x_label=None, x_name=None, y_name=None, verbose=False
        ):
        super().__init__(subsections=subsections, df=df, label=label, verbose=verbose)
        self.sectitle = sectitle
        self.with_hue = with_hue
        self.x_label = x_label
        self.x_name = x_name
        self.y_name = y_name


    def make(self):
        # Group by LR
        #df_aux = self.df.loc[self.df.groupby('lr')['mean_valid_score'].idxmin()]
        df_aux = self.df.loc[self.df.groupby([self.x_name, 'conv_kernel_size'])[self.y_name].idxmin()]
        if self.with_hue:
            hue='conv_kernel_size'
        else:
            hue=None

        lineplot_filename=f"./{self.label}_fixed_kernel_size_{self.x_name}_lineplot.png"
        lineplot_figure = GSUtils.make_lineplot(
            df=df_aux, save=True, 
            x_name=self.x_name, y_name=self.y_name, markers=True,
            fname=lineplot_filename, title=f'{self.label} - Fixed Kernel Size - {self.x_label}',
            hue=hue, palette=gsviewer.PLT_PALETTE_SET2,
            scatter=True
            )

        sec = [
            {
                'title' : NoEscape(self.sectitle),
                'text' : NoEscape(
f"This section shows how {self.x_label} vary when the kernel size is fixed \
"), 
                'subsections' : self.subsections,
                'figures' : lineplot_figure
            }
        ]
        return sec

class GSFKSLearningRate(GSFKSBase):
    def __init__(self, subsections=None, sectitle='Learning Rate', df=None, label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, label=label, sectitle=sectitle,  
            x_label='LR', x_name='lr', y_name='mean_valid_score', verbose=verbose)

class GSFKSWeightDecay(GSFKSBase):
    def __init__(self, subsections=None, sectitle='Optimizer Weight Decay', df=None, label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, label=label, sectitle=sectitle,  
            x_label='WDecay', x_name='weight_decay', y_name='mean_valid_score', verbose=verbose)

class GSFKSConvolutionKernelSize(GSFKSBase):
    def __init__(self, subsections=None, df=None, sectitle='Convolution Kernel Size', label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, label=label, with_hue=False, sectitle=sectitle, 
            x_label='CK Size', x_name='conv_kernel_size', y_name='mean_valid_score', verbose=verbose)

class GSFKSAttentionNumberOfLayers(GSFKSBase):
    def __init__(self, subsections=None, df=None, sectitle='Attention Number Of Layers', label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, label=label, sectitle=sectitle,
            x_label='Att NLayers', x_name='att_num_layers', y_name='mean_valid_score', verbose=verbose)

class GSFKSMiniBatchSize(GSFKSBase):
    def __init__(self, subsections=None, df=None, sectitle='Minibatch Size', label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, label=label, sectitle=sectitle, 
            x_label='MB Size', x_name='minibatch_size', y_name='mean_valid_score', verbose=verbose)


