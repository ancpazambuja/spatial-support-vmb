"""@package local_classes.py
Module with general utility classes.
"""

import pprint
import os
from datetime import datetime
import math
#import plotly.express as px

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np 
import scipy.stats as st
import pandas as pd

from pylatex import Document, Section, Subsection, Subsubsection, Figure, Command, Package, Ref, Label
from pylatex.utils import NoEscape

from functools import partial

from src.utl.logging import Log as log
from src.utl.classes import Graph as graph
from src.visualization.gs_viewer import GSViewer as gsviewer
from src.pdf.gs_sections_builder import GSUtils, GSPDFBuilder
from src.pdf.eval_utils import DfHeatmapBuilder, DfHeatmapExamplesCountBuilder 
from src.pdf.eval_utils import DfLineplotExamplesCountBuilder, DfHeatmapExamplesMetricErrorBuilder
from src.pdf.eval_utils import EvalUtils as eval_utl
#eval_utl.sort_summary, eval_utl.fast_fail, eval_utl.fast_fail_mock, build_paired_test_table, compute_corrected_ci, get_metric_values_from_run



import pandas as pd

def paired_test_table_to_latex(df_results, label="tab:paired_tests", caption="Bonferroni-Adjusted Paired $t$-Test Results against the Champion Architecture"):
    """
    Converts the paired test results DataFrame into a publication-ready LaTeX table.
    Automatically resizes the table to fit the page width.
    """
    if df_results is None or df_results.empty:
        return ""

    lines = []

    if label is not None:
        lines.append(f"\n")
        lines.append(f"See this Table \\ref{{{label}}}.")
    
    # 1. Begin LaTeX table environment
    lines.append("\\begin{table}[htpb]")
    lines.append("  \\centering")
#    lines.append(f"  \\caption{{{caption}}}")
#    lines.append(f"  \\label{{{label}}}")
    
    # 2. THE PAGE-FITTING MAGIC: Wrap tabular in a resizebox
    lines.append("  \\resizebox{\\textwidth}{!}{%")
    
    # l = left aligned, c = center aligned
    lines.append("  \\begin{tabular}{l c c l}")
    lines.append("  \\toprule")
    
    # 3. Table Headers
    lines.append("  \\textbf{Architecture} & \\textbf{Mean Diff.} & \\textbf{Bonferroni 95\\% CI} & \\textbf{Verdict} \\\\")
    lines.append("  \\midrule")
    
    # 4. Iterate through the dataframe rows
    for i, (_, row) in enumerate(df_results.iterrows()):
        # Clean architecture name for LaTeX (escape underscores)
        if i == 0:
            arch = str("$"+row['Architecture']+"$").replace('_', '\\_')
            diff = ci_lower = ci_upper = ci_str = ''
            verdict = f"\\textbf{{BEST}}"
        else:
            # Format numbers to 4 decimal places
            arch = str(row['Architecture']).replace('_', '\\_')
            diff = f"{row['Mean Diff (Champ - Model)']:.4f}"
            ci_lower = f"{row['Adj CI Lower']:.4f}"
            ci_upper = f"{row['Adj CI Upper']:.4f}"
            ci_str = f"[{ci_lower}, {ci_upper}]"
            verdict = str(row['Verdict'])
        
            # Highlight statistical ties
            if "Tie" in verdict:
                arch = f"\\textbf{{{arch}}}"
                verdict = f"\\textbf{{{verdict}}}"
            
        # Append the formatted row
        lines.append(f"  {arch} & {diff} & {ci_str} & {verdict} \\\\")
        
    # 5. Close everything
    lines.append("  \\bottomrule")
    lines.append("  \\end{tabular}%")
    lines.append("  } % Close resizebox")
    lines.append(f"\\caption{{{caption}}}")
    if label is not None:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}\n")
    
    return "\n".join(lines)

#def paired_test_table_to_latex(df_results, label="tab:paired_tests", caption="Bonferroni-Adjusted Paired $t$-Test Results against the Champion Architecture"):
##    '''
##    Converts the paired test results DataFrame into a publication-ready LaTeX table.
##    Requires \usepackage{booktabs} in your LaTeX preamble.
##    '''
#    if df_results is None or df_results.empty:
#        return ""
#
#    lines = []
#    
#    # 1. Begin LaTeX table environment
#    lines.append("\\begin{table}[htpb]")
#    lines.append("  \\centering")
#    lines.append(f"  \\caption{{{caption}}}")
#    lines.append(f"  \\label{{{label}}}")
#    
#    # l = left aligned, c = center aligned
#    lines.append("  \\begin{tabular}{l c c l}")
#    lines.append("  \\toprule")
#    
#    # 2. Table Headers
#    lines.append("  \\textbf{Architecture} & \\textbf{Mean Diff.} & \\textbf{Bonferroni 95\\% CI} & \\textbf{Verdict} \\\\")
#    lines.append("  \\midrule")
#    
#    # 3. Iterate through the dataframe rows
#    for _, row in df_results.iterrows():
#        # Clean architecture name for LaTeX (escape underscores)
#        arch = str(row['Architecture']).replace('_', '\\_')
#        
#        # Format numbers to 4 decimal places
#        diff = f"{row['Mean Diff (Champ - Model)']:.4f}"
#        ci_lower = f"{row['Adj CI Lower']:.4f}"
#        ci_upper = f"{row['Adj CI Upper']:.4f}"
#        ci_str = f"[{ci_lower}, {ci_upper}]"
#        
#        verdict = str(row['Verdict'])
#        
#        # If it is a tie, bold the architecture and the verdict to make it pop
#        if "Tie" in verdict:
#            arch = f"\\textbf{{{arch}}}"
#            verdict = f"\\textbf{{{verdict}}}"
#         
#        # Append the formatted row
#        lines.append(f"  {arch} & {diff} & {ci_str} & {verdict} \\\\")
#        
#    # 4. Close the table
#    lines.append("  \\bottomrule")
#    lines.append("  \\end{tabular}")
#    lines.append("\\end{table}\n")
#    
#    # Join everything with line breaks
#    return "\n".join(lines)




def ml_results_to_latex(df, caption='Model Evaluation Results', label=None):
    """
    Specialized function to convert an ML evaluation DataFrame to a LaTeX table.
    Assumes df index is 'arch_name' and contains columns:
    'mse', 'mse_95ci', 'mae', 'mae_95ci', 'ssim', 'ssim_95ci'
    """
    best_mse = df['mse'].min()
    best_mae = df['mae'].min()
    best_ssim = df['ssim'].max()

    latex_lines = []
    
    if label is not None:
        latex_lines.append(f"See Table \\ref{{{label}}}.")
        
    latex_lines.append(r"\begin{table}[h!]")
    latex_lines.append(r"\centering")
    
    # --- NEW: Resize box to force table to fit page width ---
    # The '%' sign prevents unwanted spaces in LaTeX compilation
    latex_lines.append(r"\resizebox{\textwidth}{!}{%") 
    
    latex_lines.append(r"\begin{tabular}{l c c c}") 
    latex_lines.append(r"\hline")
    
    headers = ["Model", "SSIM", "MSE", "MAE"]
    header_str = " & ".join([f"\\textbf{{{h}}}" for h in headers]) + r" \\"
    latex_lines.append(header_str)
    latex_lines.append(r"\hline")
    
    def format_cell(mean_val, ci_tuple, is_best):
        mean_str = f"{mean_val:.5g}"
        ci_str = f"[{ci_tuple[0]:.5g}, {ci_tuple[1]:.5g}]"
        
        cell_text = f"{mean_str} {ci_str}"
        
        if is_best:
            return f"\\textbf{{{cell_text}}}"
        return cell_text

    for arch_name, row in df.iterrows():
        safe_arch = str(arch_name).replace("_", r"\_")
        
        row_cells = [safe_arch]
        
        row_cells.append(format_cell(
            row['ssim'], row['ssim_95ci'], is_best=(row['ssim'] == best_ssim)
        ))
        row_cells.append(format_cell(
            row['mse'], row['mse_95ci'], is_best=(row['mse'] == best_mse)
        ))
        row_cells.append(format_cell(
            row['mae'], row['mae_95ci'], is_best=(row['mae'] == best_mae)
        ))
        
        latex_lines.append(" & ".join(row_cells) + r" \\")

    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}")
    
    # --- NEW: Close the resizebox ---
    latex_lines.append(r"}")
    
    latex_lines.append(f"\\caption{{{caption}}}")
    
    if label is not None:
        latex_lines.append(f"\\label{{{label}}}")
        
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)

def df2latex(
    df=None, caption='No caption', sort_column=None, columns=None, 
    columns_labels=None, n=None, label=None):

    if sort_column is not None:
        df = df.sort_values(by=sort_column)
    
    df = df[columns].head(n)
    parts = []

    parts.append("\n")
    parts.append("\n")
    parts.append("\n")

    if label is not None:
        parts.append(f"See Table \\ref{{{label}}}.\n")

    # --- Ajuste dinâmico do tabularx ---
    num_cols = len(columns)
    col_format = "|" + "X|" * num_cols  # Gera algo como |X|X|X|
    
    parts.append(r"\begin{table}[h!] ")
    parts.append(r"\begin{tabularx}{\textwidth}{" + col_format + "} ")
    parts.append("\n" + r"\hline" + "\n")

    # Cabeçalho
    header_cells = [r"\textbf{" + c.replace("_", r"\_") + "}" for c in columns_labels]
    parts.append(" & ".join(header_cells) + r" \\ " + "\n")
    parts.append(r"\hline" + "\n")

    # Linhas do DataFrame
    for _, row in df.iterrows():
        row_cells = []
        for v in row.values:
            if isinstance(v, float):
                formatted_v = f"{v:.5g}"
            elif isinstance(v, tuple):
                inner_vals = [f"{x:.5g}" if isinstance(x, float) else str(x) for x in v]
                formatted_v = "[" + ", ".join(inner_vals) + "]"
            else:
                formatted_v = str(v)
            
            if isinstance(v, str):
                formatted_v = formatted_v.replace("_", r"\_")
            
            row_cells.append(f" {formatted_v} ")
        
        parts.append(" & ".join(row_cells) + r" \\ " + "\n")

    parts.append(r"\hline" + "\n")
    parts.append(r"\end{tabularx} " + "\n")
    parts.append(r"\caption{" + f"{caption}" + r"} ")
    
    if label is not None:
        parts.append("\n" + r"\label{" + f"{label}" + r"} ")
    
    parts.append("\n" + r"\end{table}" + "\n")

    return "".join(parts)

def df2latex_old(
    df=None, caption='No caption', sort_column=None, columns=None, 
    columns_labels=None, n=None, label=None):

    if sort_column is not None:
        df = df.sort_values(by=sort_column)
    # The code below didn't work due (I suppose) to special charatecters and NoEscape issues.
    #sorted_string = sorted_lines[columns_to_report].to_latex(index=False, escape=False)
    df = df[columns].head(n)

    latex_string = ''

    if label is not None:
        latex_string += f"See Table \\ref{{{label}}}."
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
            if isinstance(v, float):
                formatted_v = f"{v:.5g}"
            elif isinstance(v, tuple):
                # Iterate through the tuple and format only the floats inside it
                inner_vals = [f"{x:.5g}" if isinstance(x, float) else str(x) for x in v]
                formatted_v = "[" + ", ".join(inner_vals) + "]"
            else:
                # Fallback for integers, lists, or strings
                formatted_v = str(v)
            if isinstance(v, str):
                formatted_v = formatted_v.replace("_", r"\_")

            if i == 0:
                latex_string += f" {formatted_v} " 
            else:
                latex_string += f" & {formatted_v} "
#            fmt=''
#            if isinstance(v,float):
#                fmt='.5g'
#            if i == 0:
#                latex_string += f" {v:{fmt}} " 
#            else:
#                latex_string += f" & {v:{fmt}} "
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

      
class EvalListBase:
    def __init__(self, subsections=None, df=None, matrix=None, dataset=None, label=None, verbose=False):
        self.subsections=subsections
        self.verbose=verbose
        self.df=df
        self.matrix=matrix
        self.dataset=dataset
        self.label=label
    
    def make(self):
        raise NotImplemented

class ContainerSection(EvalListBase):
    def __init__(self, figures=None, subsections=None, df=None, matrix=None, 
                 dataset=None, label=None, 
                 title=None,
                 description=None,
                 verbose=False):
        super().__init__(subsections=subsections, df=df, matrix=matrix, dataset=dataset, label=label, verbose=verbose)
        self.title = title 
        self.description = description
        self.figures = figures

    def make(self):
        sec = [
            {
                'title' : self.title,
                'text' : self.description,
                'subsections' : self.subsections,
                'figures' : self.figures,
            }
        ]
        return sec

class EvalIntroduction(EvalListBase):
    def __init__(self, subsections=None, df=None, matrix=None, dataset=None, label=None, n=10, verbose=False):
        super().__init__(subsections=subsections, df=df, matrix=matrix, dataset=dataset, label=label, verbose=verbose)

    def make(self):
        summary = self.matrix.get_mccv_summary()
        total_time = (self.df['end_time'] - self.df['start_time']).sum()
        ndays = (total_time.total_seconds() / 3600) / 24
        nruns = len(self.df)
        nmodels = len(summary)


        if self.verbose:
            log.info(f"[ {self.label} ] Nmb of runs = {nruns}")
    
        sec = [
            {
                'title' : "Introduction", 
                'text' :  NoEscape(
f"This report analyses the results of the evalution phase \
over the test dataset in the context of \\textit{{Deep Tomography 2.0}} \
development.\
\n\n\
\\begin{{table}}[h!] \
\\centering \
\\begin{{tabular}}{{|c|c|}} \
\\hline \
\\textbf{{Training Dataset}} & {self.dataset} \\\\ \
\\hline \
\\textbf{{Architectures}} & Unet|Transunet|Unet-dilation-convs \\\\ \
\\hline \
\\end{{tabular}} \
\\caption{{Report Scope}} \
\\end{{table}} \
\n\n\
\\begin{{table}}[h!] \
\\centering \
\\begin{{tabular}}{{|c|c|}} \
\\hline \
\\textbf{{Nmf of models}}  & {nmodels} \\\\ \
\\hline \
\\textbf{{Nmf of Training Curves}}  & {nruns} \\\\ \
\\hline \
\\textbf{{Processor Type}} & 4xNVIDIA A100-SXM4-80GB  \\\\ \
\\hline \
\\textbf{{A100-days:}}  & {ndays:5.1f} \\\\ \
\\hline \
\\end{{tabular}} \
\\caption{{General Information}} \
\\end{{table}} \
\n\n\
For visualization, violin plots report the distribution of performance across Monte Carlo \
splits (estimator variability), while red error bars indicate Nadeau–Bengio–adjusted \
95\\% confidence intervals for the mean, corresponding to the quantities tested \
in paired statistical comparisons with Bonferroni adjustment. \
"),
                'subsections' : self.subsections,
#                'figures' : self.figures,
            }
        ]

        return sec

class EvalConclusion(EvalListBase):
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


class EvalKernelSizeLines_old2(EvalListBase):

    def __init__(
        self,
        subsections=None,
        df=None,
        matrix=None,
        alpha=0.05,
        dataset=None,
        label=None,
        metric="mse",
        arch="unet",
        kernel_sizes=(3, 5, 7, 9, 13),
        dilations=(1, 2, 3),
        verbose=False,
    ):
        super().__init__(subsections=subsections, df=df, matrix=matrix, dataset=dataset, label=label, verbose=verbose)

        self.metric = metric
        self.arch = arch
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.alpha = alpha

        # needed for corrected CI
        self.n_train = self.matrix.get_train_dataset_size()
        self.n_test = self.matrix.get_test_dataset_size()

    # --------------------------------------------------
    # Build aggregated dataframe with CI
    # --------------------------------------------------
    def build_dataframe(self):
        df = self.df.reset_index()

        # --- filter architecture ---
        df = df[df["arch_name"].str.contains(self.arch, case=False)]

        # --- enforce desired hyperparameters ---
        df = df[df["conv_kernel_size"].isin(self.kernel_sizes)]
        df = df[df["conv_dilation_size"].isin(self.dilations)]

        # --- define corrected CI function ---
        compute_ci = partial(
            eval_utl.compute_corrected_ci,
            alpha=self.alpha,
            n_train=self.n_train,
            n_test=self.n_test
        )

        # --- aggregate MCCV runs ---
        df_grouped = (
            df.groupby(["conv_kernel_size", "conv_dilation_size"])[self.metric]
            .agg(
                mean="mean",
                std="std",
                ci=compute_ci,   # your corrected CI
            )
            .reset_index()
        )

        return df_grouped

    # --------------------------------------------------
    # Plot using GS-style visualization
    # --------------------------------------------------
    def plot(self, df_grouped):

        sns.set(style="darkgrid", context="paper")

        fig, ax = plt.subplots(figsize=(8, 5))

        for r in self.dilations:
            df_r = df_grouped[df_grouped["conv_dilation_size"] == r]
            df_r = df_r.sort_values("conv_kernel_size")

            x = df_r["conv_kernel_size"].values
            y = df_r["mean"].values

            # CI expected format: (lower, upper)
            ci = np.array(list(df_r["ci"].values))
            y_lower = ci[:, 0]
            y_upper = ci[:, 1]

            ax.plot(
                x,
                y,
                marker="o",
                label=f"r = {r}",
            )

            ax.fill_between(
                x,
                y_lower,
                y_upper,
                alpha=0.2
            )

        ax.set_xlabel("Kernel size (k)")
        ax.set_ylabel(self.metric.upper())

        ax.set_title(
            f"{self.arch.upper()} — {self.metric.upper()} vs Kernel Size\n"
            f"(MCCV + corrected CI, α={self.alpha})"
        )

        ax.legend(title="Dilation (r)")
        ax.grid(True)

        return fig

    # --------------------------------------------------
    # Framework integration (same pattern as others)
    # --------------------------------------------------
    def make(self):

        df_grouped = self.build_dataframe()
        fig = self.plot(df_grouped)

        fname = f"{self.arch}_{self.metric}_kernel_lines.png"
        fname = fname.replace(" ", "_")

        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)

        return [{
            "title": f"{self.arch.upper()} — Kernel Size vs {self.metric.upper()}",
            "figures": [fname],
            "subsections": []
        }]




class EvalListBuilder:
    def build(df=None, matrix=None, alpha=0.05, dataset=None, label=None, verbose=False):

################### SECTION 1 ###################
        s1_subsections = list()
        s1_subsections += EvalBestModelsAll(
            df=df, matrix=matrix, dataset=dataset, alpha=alpha, 
            label=label, n=100, verbose=verbose
        ).make()
        s1_subsections += EvalHeatmapEvaluationMatrix(
            df=df, matrix=matrix, dataset=dataset,
            label=label,verbose=verbose
        ).make()
        s1_subsections += EvalBestModelsAfterFastFail(
            df=df, matrix=matrix, dataset=dataset, alpha=alpha,
            label=label, n=100, verbose=verbose
        ).make()
        s1 = EvalIntroduction(
            df=df, matrix=matrix, dataset=dataset, label=label, 
            subsections=s1_subsections, verbose=verbose
        ).make()

################### SECTION 2 ###################

        s2_subsections = list()
        for arch in ["unet", "transunet"]:
            s2_subsections += EvalKernelSizeLines(
                df=df,
                matrix=matrix,
                dataset=dataset,
                alpha=alpha,
                label=label,
                metric="mse",
                arch=arch,
            ).make()

        description = "This section shows plots related to H2b."

        s2 = ContainerSection(subsections=s2_subsections, 
                              df=df, matrix=matrix, dataset=dataset,
                              title="Dilation Comparison",
                              description=description,
                              verbose=False).make()
################### SECTION 3 ###################
        s3_subsections = list()
        s3_subsections += EvalPerExampleBehaviour(
            df=df, matrix=matrix, dataset=dataset, alpha=alpha,
            label=label, verbose=verbose, all=True
        ).make()
        s3_subsections += EvalPerExampleBehaviour(
            df=df, matrix=matrix, dataset=dataset, alpha=alpha,
            label=label, verbose=verbose, all=False
        ).make()

        description = "This section shows dataset example-driven analysis. It is based mainly on the hardness " \
        "of each example. Given some metric of interest, we individually aggregate it over all selected models " \
        "and all MCCV random splits (test size 10%), calculating the mean, std, min and max and show all together " \
        "as a heatmap. This visualization highlights globally difficult examples and global easy ones."

        s3 = ContainerSection(subsections=s3_subsections, 
                              df=df, matrix=matrix, dataset=dataset,
                              title="Dataset View -- Per Example Analysis",
                              description=description,
                              verbose=False).make()

        return s1 + s2 + s3

#        s12_subsections += EvalBestModelsPerHPSet(df=df, label=label, verbose=verbose).make()
#
#        s1_subsecs = list()
#        s1_subsecs = s1_subsecs + GSParametersOverview(
#                df=df, label=label, verbose=verbose, corner=True,
#            ).make()
#        s1_subsecs = s1_subsecs + GSBestModels(
#                df=df, label=label, verbose=verbose, subsections=s12_subsections
#            ).make()
#        s1_subsecs += GSSequenceLines(df=df, label=label, 
#                x_label='LR', y_label='WDecay', z_label='CK Size', fxyz_label='MSE Mean',
#                x_name='lr', y_name='weight_decay', z_name='conv_kernel_size', fxyz_name='mean_valid_score',
#                verbose=verbose
#            ).make()
#        s1_subsecs += GSSequenceLines(df=df, label=label, 
#                x_label='LR', y_label='WDecay', z_label='CK Size', fxyz_label='MSE Mean',
#                x_name='lr', y_name='weight_decay', z_name='conv_kernel_size', fxyz_name='mean_valid_score',
#                verbose=verbose, n=60
#            ).make()
#        s1_subsecs += GSSequenceHeatMap(df=df, label=label, 
#                x_label='LR', y_label='WDecay', z_label='CK Size', fxyz_label='MSE Mean',
#                x_name='lr', y_name='weight_decay', z_name='conv_kernel_size', fxyz_name='mean_valid_score',
#                verbose=verbose
#            ).make(src/pdf/gs_sections_builder.py)
#        s1_subsecs += GSSequenceHeatMap(df=df, label=label, 
#                x_label='LR', y_label='WDecay', z_label='Att NLayers', fxyz_label='MSE Mean',
#                x_name='lr', y_name='weight_decay', z_name='att_num_layers', fxyz_name='mean_valid_score',
#                verbose=verbose
#            ).make()
##
#        s1 = GSIntroduction(
#                df=df, arch=arch, 
#                dataset=dataset, label=label, verbose=verbose,
#                subsections=s1_subsecs,
#            ).make()
#
#
########### SECTION 2
#        s2_subsecs = list()
#        s2_subsecs = s2_subsecs + GSGlobalOverview(df=df, label=label, verbose=verbose).make()
#        s2_subsecs = s2_subsecs + GSGlobalLearningRate(df=df, label=label, verbose=verbose).make()
#        s2_subsecs = s2_subsecs + GSGlobalWeightDecay(df=df, label=label, verbose=verbose).make()
#        s2_subsecs = s2_subsecs + GSGlobalConvolutionKernelSize(df=df, label=label, verbose=verbose).make()
#        if arch != 'Unet':
#            s2_subsecs = s2_subsecs + GSGlobalAttentionNumberOfLayers(df=df, label=label, verbose=verbose).make()
#        s2_subsecs = s2_subsecs + GS#GlobalMiniBatchSize(df=df, label=label, verbose=verbose).make()
#
#
#        s2 = GSGlobal(
#                df=df, arch=arch, 
#                dataset=dataset, label=label, verbose=verbose,
#                subsections= s2_subsecs
#            ).make()
#
#
#
########### SECTION 2
#
##        s212 = GSFKSWeightDecay(df=df, label=label, subsections=None, 
##                verbose=verbose).make()
##
##        s22= GSFKSBatchSize(df=df, label=label, subsections=None, 
##                verbose=verbose).make()
##
##        s23 =  GSFKSAttNumberOfLayers(df=df, label=label, subsections=None, 
##                verbose=verbose).make()
#
#        s3_subsecs = list()
#        s3_subsecs = s3_subsecs + GSFKSLearningRate(df=df, label=label, verbose=verbose).make()
#        s3_subsecs = s3_subsecs + GSFKSWeightDecay(df=df, label=label, verbose=verbose).make()
#        s3_subsecs = s3_subsecs + GSFKSConvolutionKernelSize(df=df, label=label, verbose=verbose).make()
#        if arch != 'Unet':
#            s3_subsecs = s3_subsecs + GSFKSAttentionNumberOfLayers(df=df, label=label, verbose=verbose).make()
#        s3_subsecs = s3_subsecs + GSFKSMiniBatchSize(df=df, label=label, verbose=verbose).make()
#
#        s3 = GSFixedKernelSize(
#                df=df, arch=arch, 
#                dataset=dataset, label=label, verbose=verbose,
#                subsections=s3_subsecs,
#            ).make()
#
#
#        s4 = GSVaryingKernelSizes(
#                df=df, arch=arch, 
#                dataset=dataset, label=label, verbose=verbose,
#            ).make()
#
#        s5 = GSConclusion(
#                df=df, arch=arch, 
#                dataset=dataset, label=label, verbose=verbose,
#            ).make()
#        return s1 + s2 + s3 + s4 + s5 

###########################################
###########################################
###########################################

class EvalBestModelsAll(EvalListBase):
    def __init__(self, subsections=None, df=None, matrix=None, alpha=0.05, dataset=None, label=None, n=10, verbose=False):
        super().__init__(subsections=subsections, df=df, matrix=matrix, dataset=dataset, label=label, verbose=verbose)
        self.n = n
        self.alpha=alpha
        self.n_test=self.matrix.get_test_dataset_size()
        self.n_train=self.matrix.get_train_dataset_size()

    def make(self):

        # The metric used for discarding
        attr_anchor='mse'
        attr_anchor_reverse=False
        #attr_anchor='mae'
        #attr_anchor_reverse=False
        #attr_anchor='ssim'
        #attr_anchor_reverse=True
        # For table
        summary = self.matrix.get_mccv_summary()
        sorted = eval_utl.sort_summary(summary=summary, attr_name=attr_anchor, reverse=attr_anchor_reverse)
        df_sorted = pd.DataFrame.from_dict(dict(sorted), orient='index')
        arch_order = df_sorted.index

        errorbar_callback = partial(eval_utl.compute_corrected_ci, alpha=self.alpha, n_train=self.n_train, n_test=self.n_test)

        figures_list = list()
        metrics = ("ssim", "mse", "mae")
        metrics_labels=(
                "Structural Similarity Index (Higher is Better)",
                "Mean Squared Error (Lower is Better)",
                "Mean Absolute Error (Lower is Better)",
        )
        for metric,label in zip(metrics,metrics_labels):
            violin_filename=f"./{self.label}_best_models__{metric}_violin.png"
            violin_figure = GSUtils.make_violin(
                df=self.df.reset_index(), save=True, fname=violin_filename, 
                title=metric.upper(), x_name=metric, y_name='arch_name',
                x_label=label, y_label='Architecture', y_order=arch_order,
                errorbar_callback=errorbar_callback,
#               hue='std_valid_score', palette="coolwarm",
            )
            figures_list += violin_figure


        latex_string = ml_results_to_latex(
            df=df_sorted, caption='Overall Best Models', label='tab:OverallBestModels'
        )


        sec = [
            {
                'title' : 'Overall Best Models', 
                'text' : NoEscape(latex_string),
                #'text' : 'Teste', 
                'subsections' : self.subsections,
                'figures' : figures_list,
            }
        ]
        return sec

class EvalBestModelsAfterFastFail(EvalListBase):
    def __init__(self, subsections=None, df=None, matrix=None, alpha=0.05, dataset=None, label=None, n=10, verbose=False):
        super().__init__(subsections=subsections, df=df, matrix=matrix, dataset=dataset, label=label, verbose=verbose)
        self.n = n
        self.alpha=alpha
        self.n_test=matrix.get_test_dataset_size()
        self.n_train=matrix.get_train_dataset_size()

    def make(self):

        errorbar_callback = partial(eval_utl.compute_corrected_ci, alpha=self.alpha, n_train=self.n_train, n_test=self.n_test)

        # The metric used for discarding
        attr_anchor='mse'
        attr_anchor_reverse=False
        #attr_anchor='mae'
        #attr_anchor_reverse=False
        #attr_anchor='ssim'
        #attr_anchor_reverse=True

        summary = self.matrix.get_mccv_summary()
        sorted = eval_utl.sort_summary(summary=summary, attr_name=attr_anchor, reverse=attr_anchor_reverse)

        df_sorted = pd.DataFrame.from_dict(dict(sorted), orient='index')
        df_survivors, total_models = eval_utl.fast_fail(df_sorted=df_sorted, ci_name=attr_anchor+'_95ci', reverse=attr_anchor_reverse)

        # 1. Extract the list of surviving architectures
        # (Assuming 'arch_name' is the index of your df_filtered summary table)
        surviving_archs = df_survivors.index.tolist() 
        
        # 2. Unlock the raw dataframe's MultiIndex so 'arch_name' is a standard column
        df_raw_unlocked = self.df.reset_index()
        # If reset_index creates a column called 'index' instead, rename it:
        # df_raw_unlocked = df_raw.reset_index().rename(columns={'index': 'arch_name'})
        
        # 3. FILTER: Keep only the raw rows that belong to the survivors
        df_raw_survivors = df_raw_unlocked[df_raw_unlocked['arch_name'].isin(surviving_archs)]
        df_raw_survivors = df_raw_survivors.set_index(['arch_name'])
        arch_order = df_survivors.index
        #print(f"n total  = {total_models}")
        #os._exit()
        paired_results = eval_utl.build_paired_test_table(
            df_raw_survivors.reset_index(), 
            arch_order, 
            n_train=self.n_train,
            n_test=self.n_test,
            total_models=total_models, 
            metric_col=attr_anchor, 
            alpha=self.alpha
        )
        #paired_ci_results = calculate_adjusted_paired_ci_from_df(df=df_raw_survivors, y_order=arch_order, total_models=total_models)
        #mean_d, adjusted_ci = calculate_adjusted_paired_ci(champion_seeds, unet_seeds, total_models=20)

        # Example usage:
        #
        #print(f"Mean Difference: {mean_d:.5f}")
        #print(f"Adjusted Paired CI: [{adjusted_ci[0]:.5f}, {adjusted_ci[1]:.5f}]")
        #
        #if adjusted_ci[0] < 0 < adjusted_ci[1]:
        #    print("Result: Statistical Tie (CI contains zero)")
        #else:
        #    print("Result: Champion is statistically different!") 

        ########################### Make violinplot ########################
        figures_list = list()
        metrics = ("ssim", "mse", "mae")
        metrics_labels=(
                "Structural Similarity Index (Higher is Better)",
                "Mean Squared Error (Lower is Better)",
                "Mean Absolute Error (Lower is Better)",
        )
        for metric,label in zip(metrics,metrics_labels):
            violin_filename=f"./{self.label}_best_models_{metric}_violin_after_eval_utl_fast_fail.png"
            violin_figure = GSUtils.make_violin(
                df=df_raw_survivors.reset_index(), save=True, fname=violin_filename, 
                title=metric.upper(), x_name=metric, y_name='arch_name',
                x_label=label, y_label='Architecture', y_order=arch_order,
                errorbar_callback=errorbar_callback,
#               hue='std_valid_score', palette="coolwarm",
            )
            figures_list += violin_figure

        #Slice = self.matrix.get_index_slice()
        #idx = Slice[:,arch_order]
        #df_mse_heatmap= DfHeatmapBuilder(matrix=self.matrix,idx=idx,metric='mse').build()

        ########################### Make heatmaps from examples to every seed ########################
        # Ordena primeiro por seed (1ª chave) e depois por arch_name (2ª chave)
        idx = self.matrix.get_index_slice()
        heatmap = self.matrix.get_metric_values_by_metric(idx=idx[surviving_archs,:],metrics=[attr_anchor])
        #heatmap = self.matrix.get_metric_values(idx=idx[surviving_archs,:],metric=attr_anchor)
        examples = self.matrix.get_examples(idx=idx[surviving_archs,:], dataset_col='test_indexes')
        #examples = self.matrix.get_examples(idx=idx[surviving_archs,:], dataset_name='test_indexes')
        # Assuming your dictionary is named 'my_dict'
        df_heatmap_ = pd.DataFrame(heatmap)#.swaplevel(0,1).sort_index()
        df_heatmap_.index.names = ['arch_name', 'seed']
        df_heatmap_ = df_heatmap_.swaplevel(0,1).sort_index()
#        pd.set_option('display.max_columns', None) # Mostra todas as colunas
#        pd.set_option('display.max_rows', None)    # Mostra todas as linhas (cuidado com DFs grandes!)
#        print(f"df_heatmap_ = {df_heatmap_}")
#        print(f"df_mse_heatmap = {df_mse_heatmap}")
#        print(df_heatmap_.info())
#        print(df_mse_heatmap.info())
#        os._exit(9)



#        idx = self.matrix.get_index_slice()
#        heatmap = self.matrix.get_metric_values(idx=idx[surviving_archs,:],metric=attr_anchor)
#        examples = self.matrix.get_examples(idx=idx[surviving_archs,:], dataset_col='test_indexes')
#        # Assuming your dictionary is named 'my_dict'
#        df_heatmap_ = pd.DataFrame.from_dict(heatmap, orient='index')#.swaplevel(0,1).sort_index()
#        df_heatmap_.index = pd.MultiIndex.from_tuples(df_heatmap_.index, names=["arch_name", "seed"])
#        df_heatmap_ = df_heatmap_.swaplevel(0,1).sort_index()
#        print("xxxxxxxxxxxxxxxxxxxxxxxxxx")
#        print(df_heatmap_.index)
#        print(df_heatmap_)


        all_values = np.concatenate(df_heatmap_['mse_values'].values)
        min_mse = all_values.min()
        max_mse = all_values.max()

        seeds = df_heatmap_.index.get_level_values('seed').unique()
        n_seeds = seeds.nunique()

        fig = plt.figure(figsize=(graph.PLOT_SX*2,graph.PLOT_SY*4), constrained_layout=True)
        gs = fig.add_gridspec(n_seeds, 1)

        for i,s in enumerate(seeds):
            df_aux = df_heatmap_.loc[s,:]
            valid_archs = [a for a in arch_order if a in df_aux.index]
            df_aux = df_aux.loc[valid_archs]
            examples_list = [ values for (arch, seed), values in examples.items() if seed == s and arch == valid_archs[0] ][0]
            df_heatmap = pd.DataFrame(df_aux['mse_values'].tolist(), index=df_aux.index)
            df_heatmap.columns = [f'Example_{examples_list[ii]}' for ii in range(df_heatmap.shape[1])]
            ax = fig.add_subplot(gs[i,0])
            heatplot_filename=f"./{self.label}-heatmap_view-{self.matrix.arch_col}-idx_examples-mse-seed_{s}.png"
            heatplot_figure = GSUtils.make_heatplot( ax=ax, vmin=min_mse, vmax=max_mse,
                df=df_heatmap, save=False, fname=heatplot_filename, annot=False, 
                title=f'{self.label} MSE - Heatmap for seed {s}', xticklabels=4, yticklabels='auto',
            )
            ax.set_title(f"Seed {s}")


        #plt.tight_layout()
        fig_label = clean_label(heatplot_filename)
        fname = clean_filename(heatplot_filename)
        plt.savefig(fname)

        latex_string = ml_results_to_latex(
            df=df_survivors, caption='Survivors', label='tab:Survivors'
        )

        caption=f"Bonferroni-Adjusted Paired $t$-Test Results for {attr_anchor.upper()} against the Best Architecture."
        latex_string += paired_test_table_to_latex(paired_results, label="tab:paired_tests", caption=caption)

        figures_list += [ {
                'filename' : fname,
                'caption' : f'Heatmap.',
                'label' : f"fig:{fig_label}",
            } ]

        sec = [
            {
                'title' : 'Survivors', 
                'text' : NoEscape(latex_string),
                #'text' : 'Teste', 
                'subsections' : self.subsections,
                'figures' : figures_list,
            }
        ]
        return sec

class EvalHeatmapEvaluationMatrix(EvalListBase):
    def __init__(self, subsections=None, df=None, matrix=None, dataset=None, label=None, verbose=False):
        super().__init__(subsections=subsections, df=df, matrix=matrix, dataset=dataset, label=label, verbose=verbose)

    def make(self):

        # The metric used for discarding
        attr_anchor='mse'
        attr_anchor_reverse=False
        #attr_anchor='mae'
        #attr_anchor_reverse=False
        #attr_anchor='ssim'
        #attr_anchor_reverse=True

        summary = self.matrix.get_mccv_summary()
        sorted = eval_utl.sort_summary(summary=summary, attr_name=attr_anchor, reverse=attr_anchor_reverse)
        #eval_utl.print_mccv_summary(summary, verbose=False)

        df_sorted = pd.DataFrame.from_dict(dict(sorted), orient='index')
        total_models = len(df_sorted)

        # 1. Extract the list of all architectures
        # (Assuming 'arch_name' is the index of your df_filtered summary table)
        archs = df_sorted.index.tolist() 
        arch_order = df_sorted.index
        #print(f"n total  = {total_models}")
        #os._exit()
        figures_list = list()
        ########################### Make heatmaps from examples to every seed ########################
        # Ordena primeiro por seed (1ª chave) e depois por arch_name (2ª chave)
        idx = self.matrix.get_index_slice()
        heatmap = self.matrix.get_metric_values_by_metric(idx=idx[archs,:],metrics=[attr_anchor])
        #heatmap = self.matrix.get_metric_values(idx=idx[archs,:],metric=attr_anchor)
        examples = self.matrix.get_examples(idx=idx[archs,:], dataset_col='test_indexes')
        #examples = self.matrix.get_examples(idx=idx[surviving_archs,:], dataset_name='test_indexes')
        # Assuming your dictionary is named 'my_dict'
        df_heatmap_ = pd.DataFrame(heatmap)#.swaplevel(0,1).sort_index()
        df_heatmap_.index.names = ['arch_name', 'seed']
        df_heatmap_ = df_heatmap_.swaplevel(0,1).sort_index()


        all_values = np.concatenate(df_heatmap_['mse_values'].values)
        min_mse = all_values.min()
        max_mse = all_values.max()

        #seeds = df_heatmap_.index.get_level_values('seed').unique()
        #n_seeds = seeds.nunique()
        seeds = [7]
        n_seeds = 1

        fig = plt.figure(figsize=(graph.PLOT_SX*3,graph.PLOT_SY), constrained_layout=True)
        gs = fig.add_gridspec(n_seeds, 1)

        for i,s in enumerate(seeds):
            df_aux = df_heatmap_.loc[s,:]
            valid_archs = [a for a in arch_order if a in df_aux.index]
            df_aux = df_aux.loc[valid_archs]
            examples_list = [ values for (arch, seed), values in examples.items() if seed == s and arch == valid_archs[0] ][0]
            df_heatmap = pd.DataFrame(df_aux['mse_values'].tolist(), index=df_aux.index)
            df_heatmap = df_heatmap.rename(index=eval_utl.arch_name2arch_name_paper) 
            assert df_heatmap.index.is_unique, (
                f"Mapeamento gerou rótulos duplicados: "
                f"{df_heatmap.index[df_heatmap.index.duplicated()].tolist()}"
            )

            df_heatmap.columns = [f'{examples_list[ii]}' for ii in range(df_heatmap.shape[1])]
            ax = fig.add_subplot(gs[i,0])
            heatplot_filename=f"./{self.label}-heatmap_view-{self.matrix.arch_col}-idx_examples-mse-seed_{s}_all_models.png"
            heatplot_figure = GSUtils.make_heatplot( ax=ax, vmin=min_mse, vmax=max_mse,
                df=df_heatmap, save=False, fname=heatplot_filename, annot=False, 
                title=f'{self.label} MSE - Heatmap for seed {s}', xticklabels=1, yticklabels=1, #'auto',
            )
            ax.set_title(f"Seed {s}")


        #plt.tight_layout()
        fig_label = clean_label(heatplot_filename)
        fname = clean_filename(heatplot_filename)
        plt.savefig(fname)

        latex_string = "Evaluation Matrix"

        figures_list += [ {
                'filename' : fname,
                'caption' : f'Heatmap.',
                'label' : f"fig:{fig_label}",
            } ]

        sec = [
            {
                'title' : 'Evaluation Matrix', 
                'text' : NoEscape(latex_string),
                #'text' : 'Teste', 
                'subsections' : self.subsections,
                'figures' : figures_list,
            }
        ]
        return sec
    
class EvalPerExampleBehaviour(EvalListBase):
    def __init__(self, subsections=None, df=None, matrix=None, alpha=0.05, 
                 dataset=None, label=None, all=True, 
                 metric='mse', verbose=False):
        super().__init__(subsections=subsections, df=df, matrix=matrix, dataset=dataset, label=label, verbose=verbose)
        self.all=all
        self.alpha=alpha
        self.n_test=matrix.get_test_dataset_size()
        self.n_train=matrix.get_train_dataset_size()

        # The metric used for discarding
        self.metric = metric
        self.attr_anchor=self.metric
        self.attr_anchor_values=self.attr_anchor+'_values'
        self.attr_anchor_ci=self.attr_anchor+'_95ci'
        self.attr_anchor_reverse=False
        #attr_anchor='mae'
        #attr_anchor_reverse=False
        #attr_anchor='ssim'
        #attr_anchor_reverse=True
        idx = self.matrix.get_index_slice()

        # Build examples
        self.df['test_indexes'] = self.df.apply(
            lambda x: self.matrix.get_examples(
                idx=idx[x['arch_name'], x['seed']],
                dataset_col='test_indexes'
                )[idx[x['arch_name'], x['seed']]], axis=1
            )
        self.summary = self.matrix.get_mccv_summary()
        self.sorted = eval_utl.sort_summary(summary=self.summary, attr_name=self.attr_anchor, reverse=self.attr_anchor_reverse)
        self.example_count = self.matrix.get_examples_in_splits_count()

        # --- A) Survivors selection --- 
        df_sorted = pd.DataFrame.from_dict(dict(self.sorted), orient="index")
        if all:
            df_survivors, _ = eval_utl.fast_fail_mock(
                df_sorted=df_sorted,
                ci_name=self.attr_anchor_ci,
                reverse=self.attr_anchor_reverse
            )
        else:
            df_survivors, _ = eval_utl.fast_fail(
                df_sorted=df_sorted,
                ci_name=self.attr_anchor_ci,
                reverse=self.attr_anchor_reverse
            )
    
        self.surviving_archs = df_survivors.index.tolist()

        self.df_long = self.matrix.get_long_df(metric=self.metric, arch_subset=self.surviving_archs)


    def build_per_example_matrix(self, all: bool = True) -> pd.DataFrame:
        """
        Builds a per-example statistics dataframe used for data-centric views (scatterplots, difficulty curves).
    
        New approach:
          1) Determine surviving architectures using eval_utl.fast_fail / eval_utl.fast_fail_mock on your summary table.
          2) Materialize a long dataframe from EvaluationMatrix for the chosen metric and survivors.
          3) Group by example_id and compute mean/min/max/std/count.
          4) Optionally merge with full [0..999] list and drop missing.
        """
    
    
        # --- B) Build long dataframe from EvaluationMatrix ---
        # Anchor metric name is like: "mse_values" / "mae_values" / "ssim_values"
        # EvaluationMatrix expects metric without "_values"
        metric_base = self.attr_anchor_values.replace("_values", "")
    
        long_df = self.matrix.get_long_df(
            metric=metric_base,
            dataset_col=self.matrix.test_indexes_col,
            arch_subset=self.surviving_archs,
            cache=True
        )
        # long_df columns: [arch_name, seed, example_id, value]
    
        # --- C) Group by example_id (same outcome as your old explode approach) ---
        df_example_stats = (
            long_df.groupby("example_id")["value"]
            .agg(
                mean_error="mean",
                min_error="min",
                max_error="max",
                std_error="std",
                eval_count="count"
            )
            .reset_index()
        )
        df_example_stats['max_std'] = df_example_stats['mean_error'] + (df_example_stats['std_error'] / 2.)
        df_example_stats['min_std'] = df_example_stats['mean_error'] - (df_example_stats['std_error'] / 2.)
    
        # --- D) Optional: expose missing examples & drop untested (keeps your behavior) ---
        # If your dataset size is always 1000:
        whole_dataset_len = self.matrix.get_train_dataset_size(drop_last_example=False) + self.matrix.get_test_dataset_size(drop_last_example=False)
        assert whole_dataset_len == 1000, f"Dataset size must be 1000 for this project. Found {whole_dataset_len}"
        all_examples = pd.DataFrame({"example_id": range(whole_dataset_len)})
        df_final = pd.merge(all_examples, df_example_stats, on="example_id", how="left")
    
        ## Keep your previous behavior: drop examples never tested
        #df_final = df_final.dropna(subset=["mean_error", "std_error"]).copy()
        df_final[["mean_error","min_error","max_error","min_std", "max_std", "std_error"]] = df_final[["mean_error","min_error","max_error","min_std","max_std","std_error"]].astype(float)
        df_final["eval_count"] = df_final["eval_count"].fillna(0).astype(int) 

        return df_final


    def build_per_example_matrix_2(self, all=True):

        # 2. Unlock the raw dataframe's MultiIndex so 'arch_name' is a standard column
        df_raw_unlocked = self.df.reset_index()

        # Explode the lists into individual rows
        # (Pandas will duplicate the arch_name and seed for every unpacked item)
        df_exploded = df_raw_unlocked.copy()
        df_exploded = df_exploded[df_exploded['arch_name'].isin(self.surviving_archs)]

        # Get values from mlflow on the fly.
        df_exploded[self.attr_anchor_values] = df_exploded['run_id'].apply(
            lambda x: eval_utl.get_metric_values_from_run(
                mlflow_=self.matrix.mlflow_, run_id=x, metric=self.attr_anchor_values
                )
            )
        
        df_exploded = df_exploded.explode(['test_indexes', self.attr_anchor_values])

        # Ensure proper data types (Explode sometimes leaves them as objects)
        df_exploded['test_indexes'] = df_exploded['test_indexes'].astype(int)
        df_exploded[self.attr_anchor_values] = df_exploded[self.attr_anchor_values].astype(float)

        # Group by the example ID exactly as before
        df_example_stats = df_exploded.groupby('test_indexes')[self.attr_anchor_values].agg(
            mean_error='mean',
            min_error='min',
            max_error='max',
            std_error='std',
            eval_count='count'
        ).reset_index()

        # Rename for clarity
        df_example_stats = df_example_stats.rename(columns={'test_indexes': 'example_id'})
        # Optional: Merge with a complete list of 0-999 to expose the missing examples
        all_examples = pd.DataFrame({'example_id': range(1000)})
        df_final = pd.merge(all_examples, df_example_stats, on='example_id', how='left')
        # Drop the NaN values (the ~34% of examples that were never tested)
        df_final = df_final.dropna(subset=['mean_error', 'std_error']).copy()

        return df_final

    def add_hardest_easiest_table(self, df=None, n=10, ascending=False):
        # Get the Top 10 examples with the highest mean error
        top_10 = df.sort_values('mean_error', ascending=ascending).head(n)
        
        # Format for a clean LaTeX or Markdown table
        summary_table = top_10[['example_id', 'mean_error', 'std_error', 'min_error', 'max_error', 'eval_count']].copy()
        
        # Round the floats for clean reading
        summary_table = summary_table.round(4)
        
        return summary_table

    def add_dataset_difficulty_curve(self, 
                                     df=None, 
                                     upper_name='max_error', 
                                     bottom_name="min_error", 
                                     fill_label='Min-Max Range',
                                     all_or_surv=''):
        # Sort from easiest to hardest
        df_sorted = df.sort_values('mean_error').reset_index(drop=True)

        curveplot_filename=f"./{self.label}-{self.all}-dataset_difficulty_curve-per_example_mean__{bottom_name}__{upper_name}.png"
        curveplot_figure = GSUtils.make_fillplot(
            df=df_sorted, save=True, 
            min_name=bottom_name, max_name=upper_name, mean_name='mean_error',
            fill_label=fill_label,
            fname=curveplot_filename, title=f'Example-level Error Analysis - {all_or_surv} - Mean',
            palette=gsviewer.PLT_PALETTE_SET1,
            )
        return curveplot_figure


    def add_dataset_values_heatmap_grid(self, all_or_surv='', metric='mse'):

        vmin = self.df_long[self.matrix.value_col].min()
        vmin = max( vmin - vmin * 0.1, 0 )
        vmax = self.df_long[self.matrix.value_col].max()
        vmax = vmax + vmax * 0.1

        title =f"Per Example Information -- {all_or_surv} -- Sorted by metric mean (row-major order)."
        fig = plt.figure(figsize=(graph.PLOT_SX*2,graph.PLOT_SY*3), constrained_layout=True)
        gs = fig.add_gridspec(3,2)

        params = { 
                "count" : { 'vmin': None, 'vmax': None, 'gs' : gs[0,0], 'fill_value': None, 
                           'arch_subset': self.surviving_archs[0:1], 'cbar' : True, 'title' : "NMB OF OCCURRENCES" },
                "mean"  : { 'vmin': vmin, 'vmax': vmax, 'gs' : gs[1,0], 'fill_value': None, 
                           'arch_subset': self.surviving_archs, 'cbar' : True, 'title' : "MEAN" },
                "std"   : { 'vmin': vmin, 'vmax': vmax, 'gs' : gs[1,1], 'fill_value': None, 
                           'arch_subset': self.surviving_archs, 'cbar' : True, 'title' : "STD" },
                "min"   : { 'vmin': vmin, 'vmax': vmax, 'gs' : gs[2,0], 'fill_value': None, 
                           'arch_subset': self.surviving_archs, 'cbar' : True, 'title' : "MIN" },
                "max"   : { 'vmin': vmin, 'vmax': vmax, 'gs' : gs[2,1], 'fill_value': None, 
                           'arch_subset': self.surviving_archs, 'cbar' : True, 'title' : "MAX" },
            }
        

        for k,v in params.items():

            df = DfHeatmapExamplesMetricErrorBuilder( 
                matrix=self.matrix, metric=metric, arch_subset=v["arch_subset"], fill_value=v['fill_value'], agg_func=k,
                sort_by_mean=True
            ).build()

            ax = fig.add_subplot(v['gs'])

            # main plot 
            _ = GSUtils.make_heatplot( ax=ax, vmin=v['vmin'], vmax=v['vmax'], df=df, save=False, 
                fname="dummy", annot=False, title="dummy",
                cmap=gsviewer.PLT_PALETTE_CIVIDIS, cbar=v['cbar'],
            )

            threshold = np.nanpercentile(df.values, 95)
            mask_highlight = df.values < threshold
            marks = np.where(
            (df.notna()) & (df >= threshold), 
            '✖', #'●', '★', '✖', '▲', etc.)
            ''    # Vazio para o resto
            )

            # highlight plot
            _ = GSUtils.make_heatplot( df=df, ax=ax, save=False, 
                vmin=v['vmin'], vmax=v['vmax'], 
                fname="dummy", title="dummy",
                annot=marks, mask=mask_highlight, 
                #cmap=["#ffffff00"], #gsviewer.PLT_PALETTE_CIVIDIS,
                cmap=gsviewer.PLT_PALETTE_CIVIDIS,
                cbar=False,
                fmt='',
                annot_kws={
                    'size': 5,             # Tamanho do símbolo
                    'color': 'red',       # Cor do símbolo (ex: white, red, black)
                    'weight': 'bold'        # Negrito para destacar mais
                }
            )

            ax.set_title(v['title'])

        fig.suptitle(title)

        #plt.tight_layout()
        heatmap_filename=f"./{self.label}-{all_or_surv}-_per-example-count_heatmap_grid.png"
        fig_label = clean_label(heatmap_filename)
        fname = clean_filename(heatmap_filename)
        plt.savefig(fname)

        caption = "Dataset example hardness map (20×50 = 1000 examples). Each cell represents " \
        "one example pair, ordered by increasing mean test MSE. Color indicates the number of " \
        "OCCURRENCES, mean MSE, mean STD, mean MIN and mean MAX aggregated over the selected " \
        "model set and 10 MCCV random splits (test size 10%). This visualization highlights " \
        "globally difficult examples and supports data-centric analysis by linking the hardness " \
        "distribution to the “hardest examples” table and split/model variability summaries."

        figure_list = [ {
                'filename' : fname,
                'caption' : caption, 
                'label' : f"fig:{fig_label}",
            } ]

        return figure_list

    def add_dataset_count_heatmap(self, all_or_surv=''):

        title =f"Example's Number Of Occurrences"
        idx = self.matrix.get_index_slice()[self.surviving_archs[0], :]

        fig = plt.figure(figsize=(graph.PLOT_SX*2,graph.PLOT_SY*2), constrained_layout=True)
        gs = fig.add_gridspec(2,2)

        ### HEATMAP 

        df = DfHeatmapExamplesCountBuilder(matrix=self.matrix, idx=idx, sort=True).build()
        ax = fig.add_subplot(gs[0,0])

        _ = GSUtils.make_heatplot( ax=ax, vmin=0,# vmax=len(self.matrix.get_seeds()),
            df=df, save=False, fname="dummy", annot=False, 
            title="dummy",#title + " -- Sort by occurrences",
            cmap=gsviewer.PLT_PALETTE_CIVIDIS, cbar=False,
            #cmap=gsviewer.PLT_PALETTE_SET11, cbar=True,
        )
        ax.set_title(f"Sorted by number of occurrences.")

        ### HEATMAP 

        df = DfHeatmapExamplesCountBuilder(matrix=self.matrix, idx=idx, sort=False).build()
        ax = fig.add_subplot(gs[0,1])

        _ = GSUtils.make_heatplot( ax=ax, vmin=0, #vmax=len(self.matrix.get_seeds()),
            df=df, save=False, fname="dummy", annot=False, 
            title="dummy", #title + " -- Sort by example id (row-major order)",
            cmap=gsviewer.PLT_PALETTE_CIVIDIS, cbar=True,
        )
        ax.set_title(f"Sorted by example id (row-major order).")
        fig.suptitle(title)

        ### HISTOGRAM 

        builder = DfLineplotExamplesCountBuilder(matrix=self.matrix, idx=idx)
        df=builder.build()
        ax = fig.add_subplot(gs[1, :])

        _ = GSUtils.make_scatterplot( ax=ax, df=df, x_name=builder.x_name, y_name=builder.y_name,
                                  save=False, fname="dummy", title=title, palette=gsviewer.PLT_PALETTE_CIVIDIS, 
        )
        ax.set_title(f"Sorted by example id.")
        fig.suptitle(title)

        #plt.tight_layout()
        heatmap_filename=f"./{self.label}-{all_or_surv}-_per-example-count_heatmap_histogram.png"
        fig_label = clean_label(heatmap_filename)
        fname = clean_filename(heatmap_filename)
        plt.savefig(fname)

        figure_list = [ {
                'filename' : fname,
                'caption' : f'Number of occurrences of each example in test dataset over all random splits.',
                'label' : f"fig:{fig_label}",
            } ]


        return figure_list

    def add_dataset_difficulty_scatterplot(self, df=None, all_or_surv=''):
        scatterplot_filename=f"./{self.label}-{all_or_surv}_per-example-behaviour-mean-std_scatterplot.png"
        scatterplot_figure = GSUtils.make_scatterplot(
            df=df, save=True, 
            x_name='mean_error', y_name='std_error',
            fname=scatterplot_filename, title=f'Example-level Error Analysis - {all_or_surv} - Mean / StdErr',
            hue='max_error', hue_norm=(df['max_error'].min(), df['max_error'].max()),
            size='max_error', palette='cividis',#gsviewer.PLT_PALETTE_CIVIDIS, 
            median_lines=True, 
            )
        return scatterplot_figure

    def heatmap_df_from_dict_by_example_id(values_dict, examples_dict, index_names=("arch_name", "seed")):
        # Build per-run dict: {example_id: value}
        aligned = {}
        for key in values_dict.keys():
            ex_ids = examples_dict[key]
            vals = values_dict[key]
            if len(ex_ids) != len(vals):
                raise ValueError(f"Mismatch for {key}: {len(ex_ids)} ids vs {len(vals)} vals")
            aligned[key] = dict(zip(ex_ids, vals))
    
        df = pd.DataFrame.from_dict(aligned, orient="index")  # columns become example_id
        df.index = pd.MultiIndex.from_tuples(df.index, names=list(index_names))
        df.columns.name = "example_id"
        df = df.sort_index(axis=1)  # sort example_id columns
        return df

    def make(self):

        df_final = self.build_per_example_matrix(all=self.all)

        ######################### #############################
        all_or_surv='Survivors'
        if self.all:
            all_or_surv='All'

        figures_list = list()
        figures_list += self.add_dataset_difficulty_scatterplot(df=df_final, all_or_surv=all_or_surv)
        figures_list += self.add_dataset_difficulty_curve(df=df_final, all_or_surv=all_or_surv)
        figures_list += self.add_dataset_difficulty_curve(df=df_final, upper_name='max_std', bottom_name='min_std', fill_label='Std MSE', all_or_surv=all_or_surv)
        figures_list += self.add_dataset_count_heatmap(all_or_surv=all_or_surv)
        figures_list += self.add_dataset_values_heatmap_grid(all_or_surv=all_or_surv, metric='mse')

#- Heatmap of mean hardness hih_ihi​ (sorted)
#- Heatmap of split instability sis_isi​ (same ordering)
#- Small histogram of hih_ihi​
#- Mark the top-50 (or top-10) examples
#Etapa 3 — Heatmap 20×50 (o que você pediu agora)
#Implementaremos:
#
#Ordenação explícita dos exemplos (por h_i)
#Reorganização em grid 20×50
#Controle de colormap, clipping e colorbar
#Mesma ordem reutilizada para outros mapas (consistência visual)

        title = f"{all_or_surv} Models"
        nhard = 25

        aux_dict = { 
            "hardest" : { "ascending" : False, "caption" : "hardest examples", "label" : "hardest_examples" },
            "easiest" : { "ascending" : True, "caption" : "easiest examples", "label" : "easiest_examples" }
            }

        description_list = []
        for _,v in aux_dict.items():
            cap = v["caption"]
            label = v["label"]
            summary_table = self.add_hardest_easiest_table(df=df_final, n=nhard, ascending=v["ascending"])
            summary_table['eval_count'] = summary_table['example_id'].apply(lambda x: self.example_count[x])

            description_list.append(
                df2latex(
                df=summary_table, caption=f'{nhard}-{cap}', label=f'tab:{nhard}_{label}', sort_column=None,
                columns=['example_id', 'mean_error', 'min_error', 'max_error', 'std_error', 'eval_count'],
                columns_labels=["Example ID", "Mean", "Min", "Max", "StdErr", "Nmb Of Occurrences"], 
                n=nhard,
                )
            )

        description = "".join(description_list)

#        hardest_summary_table = self.add_hardest_easiest_table(df=df_final, n=nhard, ascending=False)
#        hardest_summary_table['eval_count'] = hardest_summary_table['example_id'].apply(lambda x: self.example_count[x]*10.)
#        #summary_table['eval_count'] = summarsummary_table['eval_count'].apply(lambda x: x/summary_table['eval_count'].max())
#        description2 = df2latex(
#            df=hardest_summary_table, caption=f'{nhard}-hardest examples', label=f'tab:{nhard}_hardest_examples', sort_column=None,
#            columns=['example_id', 'mean_error', 'min_error', 'max_error', 'std_error', 'eval_count'],
#            columns_labels=["Example ID", "Mean", "Min", "Max", "StdErr", "perc-Splits"], 
#            n=nhard,
#        )

        sec = [
            {
                'title' : title, 
                'text' : NoEscape(description),
                #'text' : 'Teste', 
                'subsections' : self.subsections,
                'figures' : figures_list,
            }
        ]
        return sec

class EvalBestModelsPerKernelSize(EvalListBase):
    def __init__(self, subsections=None, df=None, matrix=None, alpha=0.05, dataset=None, label=None, n=10, verbose=False):
        super().__init__(subsections=subsections, df=df, matrix=matrix, dataset=dataset, label=label, verbose=verbose)
        self.n = n
        self.alpha=alpha

    def make(self):
        attr_anchor="arch_name"
        attr_anchor_reverse=False
        summary = self.matrix.get_mccv_summary()
        for k,v in summary.items(): summary[k] = v | { 'arch_name' : k} #insert the key (arch_name) in its content v
        sorted = eval_utl.sort_summary(summary=summary, attr_name=attr_anchor, reverse=attr_anchor_reverse)

        df_summary = pd.DataFrame.from_dict(dict(sorted), orient='index')
        df_summary=df_summary.iloc[:10]

        lineplot_filename=f"./{self.label}_best_models_per_kernel_size__lineplot.png"

        lineplot_figure = GSUtils.make_lineplot(
            df=df_summary, save=True, fname=lineplot_filename, 
            title=f'{self.label} - conv_kernel_size', scatter=True,
            x_name='arch_name', y_name='mse',
#            hue='std_valid_score', palette="coolwarm",
        )

        columns_to_report = [
            'ssim', 'mse', 'mae',
        ]

        columns_labels = [
            'SSIM', 'MSE', 'MAE'
        ]


        latex_string = ml_results_to_latex(
            df=df_summary, caption='Overall Best Models', label='tab:BestModelsPerKernelSize'
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

class EvalSequenceLines(EvalListBase):
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

class EvalSequenceHeatMap(EvalListBase):
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
    

class EvalKernelSizeLines_old(EvalListBase):
    def __init__(
        self,
        df=None,
        matrix=None,
        dataset=None,
        label=None,
        metric="mse",
        arch="unet",   # "unet" or "transunet"
        kernel_sizes=(3, 5, 7, 9, 13),
        dilations=(1, 2, 3),
        verbose=False,
    ):
        super().__init__(df=df, matrix=matrix, dataset=dataset, label=label, verbose=verbose)

        self.metric = metric
        self.arch = arch
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations

    def build_dataframe(self):
        """
        Build aggregated dataframe:
        columns: [arch_name, k, r, metric_mean, metric_std]
        """
        df = self.df.copy().reset_index()

        # --- filter architecture ---
        df = df[df["arch_name"].str.contains(self.arch, case=False)]

        # --- enforce kernel/dilation sets ---
        df = df[df["conv_kernel_size"].isin(self.kernel_sizes)]
        df = df[df["conv_dilation_size"].isin(self.dilations)]

        # --- group over seeds (MCCV aggregation) ---
        df_grouped = (
            df.groupby(["arch_name", "conv_kernel_size", "conv_dilation_size"])[self.metric]
            .agg(metric_mean="mean", metric_std="std")
            .reset_index()
        )

        return df_grouped

    def make_plot(self, df_grouped):
        fig, ax = plt.subplots(figsize=(8, 5))

        for r in self.dilations:
            df_r = df_grouped[df_grouped["conv_dilation_size"] == r].sort_values("conv_kernel_size")

            ax.plot(
                df_r["conv_kernel_size"],
                df_r["metric_mean"],
                marker="o",
                label=f"r = {r}",
            )

            # optional: confidence band (std)
            ax.fill_between(
                df_r["conv_kernel_size"],
                df_r["metric_mean"] - df_r["metric_std"],
                df_r["metric_mean"] + df_r["metric_std"],
                alpha=0.2,
            )

        ax.set_xlabel("Kernel size (k)")
        ax.set_ylabel(self.metric.upper())
        ax.set_title(f"{self.arch.upper()} — Metric vs Kernel Size")
        ax.legend(title="Dilation size")
        ax.grid(True)

        return fig

    def make(self):
        df_grouped = self.build_dataframe()
        fig = self.make_plot(df_grouped)

        # save figure (following your style)
        curve_filename=f"./{self.label}_curve_kernel_vs_{self.metric}_{self.arch}.png"
        fig_label = clean_label(curve_filename)
        fname = clean_filename(curve_filename)

        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return [{
            "title": f"{self.arch.upper()} — Kernel Size vs {self.metric.upper()}",
            "figures": [
                {
                'filename' : fname,
                'caption' : f'Heatmap.',
                'label' : f"fig:{fig_label}",
                } 
            ],
            "subsections": []
        }]


class EvalKernelSizeLines(EvalListBase):

    def __init__(
        self,
        subsections=None,
        df=None,
        matrix=None,
        alpha=0.05,
        dataset=None,
        label=None,
        metric="mse",
        arch="unet",
        kernel_sizes=(3, 5, 7, 9, 13),
        dilations=(1, 2, 3),
        verbose=False,
    ):
        super().__init__(
            subsections=subsections,
            df=df,
            matrix=matrix,
            dataset=dataset,
            label=label,
            verbose=verbose,
        )

        self.metric = metric
        self.arch = arch
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.alpha = alpha

    # --------------------------------------------------
    # STEP 1 — Reuse get_mccv_summary
    # --------------------------------------------------
    def build_dataframe(self):
    
        summary = self.matrix.get_mccv_summary()
        #eval_utl.print_mccv_summary(summary)
    
        df = pd.DataFrame.from_dict(summary, orient="index")
        df.index.name = self.matrix.arch_col
    
        # filter arch type, kernel sizes, dilations
        df = df[df.index.str.startswith(self.arch, na=False)]
        df = df[df["conv_kernel_size"].isin(self.kernel_sizes)]
        df = df[df["conv_dilation_size"].isin(self.dilations)]

    
        # --- select best model per (k, r) ---
        # lower is better for MSE/MAE, higher is better for SSIM
        if self.metric in ("mse", "mae"):
            idx_best = df.groupby(["conv_kernel_size", "conv_dilation_size"])[self.metric].idxmin()
        else:
            idx_best = df.groupby(["conv_kernel_size", "conv_dilation_size"])[self.metric].idxmax()
    
        df = df.loc[idx_best]
        
        # unpack CI
        ci_col = f"{self.metric}_95ci"
        df["ci_lower"] = df[ci_col].apply(lambda x: x[0])
        df["ci_upper"] = df[ci_col].apply(lambda x: x[1])
    
        df = df[[
            self.metric,
            "ci_lower",
            "ci_upper",
            "conv_kernel_size",
            "conv_dilation_size",
        ]].copy()
    
        # label for legend
        df["dilation"] = df["conv_dilation_size"].apply(lambda r: f"r={int(r)}")
    
        df = df.sort_values(["conv_dilation_size", "conv_kernel_size"])
    
        return df

   
    # --------------------------------------------------
    # STEP 2 — Build plot via GSUtils
    # --------------------------------------------------
    def make_plot(self, df):

        xtick_labels = [
            f"{int(k)}x{int(k)}"
            for k in sorted(df["conv_kernel_size"].unique())
        ]

        figure_list = GSUtils.make_lineplot_errorbar(
            df=df,
            x_name="conv_kernel_size",
            y_name=self.metric,
            y_lower_name="ci_lower",
            y_upper_name="ci_upper",
            hue="dilation",
            hue_offset=0.1,
            title=(
                f"{self.arch.upper()} — {self.metric.upper()} vs Kernel Size "
                f"({int((1 - self.alpha) * 100)}% CI)"
            ),
            fname=f"{self.label}_{self.arch}_{self.metric}_kernel_lines",
            xtick_labels=xtick_labels,
        )

        return figure_list

    # --------------------------------------------------
    # STEP 3 — Framework integration
    # --------------------------------------------------
    def make(self):

        df = self.build_dataframe()
        figure_list = self.make_plot(df)

        return [{
            "title": f"{self.arch.upper()} — Kernel Size vs {self.metric.upper()}",
            "figures": figure_list,
            "subsections": [],
        }]


class Eval3DView(EvalListBase):
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



###########################################
###########################################
###########################################



