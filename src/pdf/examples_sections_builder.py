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
from src.pdf.eval_sections_builder import EvalIntroduction, EvalPerExampleBehaviour
from src.pdf.eval_sections_builder import EvalBestModelsAll, EvalBestModelsAfterFastFail, ContainerSection
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

class ExampleListBuilder:
    def build(df=None, matrix=None, alpha=0.05, dataset=None, label=None, verbose=False):

################### SECTION 1 ###################
        s1_subsections = list()
        s1_subsections += EvalBestModelsAll(
            df=df, matrix=matrix, dataset=dataset, alpha=alpha, 
            label=label, n=100, verbose=verbose
        ).make()
#        s1_subsections += EvalBestModelsAfterFastFail(
#            df=df, matrix=matrix, dataset=dataset, alpha=alpha,
#            label=label, n=100, verbose=verbose
#        ).make()

        s1 = EvalIntroduction(
            df=df, matrix=matrix, dataset=dataset, label=label, 
            subsections=s1_subsections, verbose=verbose
        ).make()

################### SECTION 2 ###################
        s2_subsections = list()
        s2_subsections += EvalPerExampleBehaviour(
            df=df, matrix=matrix, dataset=dataset, alpha=alpha,
            label=label, verbose=verbose, all=True
        ).make()
#        s2_subsections += EvalPerExampleBehaviour(
#            df=df, matrix=matrix, dataset=dataset, alpha=alpha,
#            label=label, verbose=verbose, all=False
#        ).make()

        description = "This section shows dataset example-driven analysis. It is based mainly on the hardness " \
        "of each example. Given some metric of interest, we individually aggregate it over all selected models " \
        "and all MCCV random splits (test size 10%), calculating the mean, std, min and max and show all together " \
        "as a heatmap. This visualization highlights globally difficult examples and global easy ones."

        s2 = ContainerSection(subsections=s2_subsections, 
                              df=df, matrix=matrix, dataset=dataset,
                              title="Dataset View -- Per Example Analysis",
                              description=description,
                              verbose=False).make()

        return s1 + s2 

