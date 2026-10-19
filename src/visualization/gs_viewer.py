
"""@package gs_viewer.py
Visualization of grid search information.
"""

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors0

import seaborn as sns 

# import plotly.express as px

import random

from src.utl.classes import Utils as utl
from src.utl.classes import Graph as graph

from src.utl.logging import Log as log


class GSViewer:
    """!
    Contains specific visualizations for grid search information.
    """
    DEFAULT_CMAP="Blues_r"
    DEFAULT_SNS_CONTEXT='paper' #'poster', 'talk'
    DEFAULT_SNS_STYLE='darkgrid' # ''ticks', 'darkgrid', 'whitegrid', 'white', 'black'
    DEFAULT_EDGE_COLOR = 'black'  # Set edge color for all markers 
    DEFAULT_PALETTE = 'Blues_d'  

    # Palettes from matplolib
    # Continous 
    PLT_PALETTE_CIVIDIS =  [plt.cm.cividis(i) for i in range(256)]
    PLT_PALETTE_SEISMIC = [plt.cm.seismic(i) for i in range(256)]
    PLT_PALETTE_UPPER_OUTLIER = [plt.cm.coolwarm(i) for i in range(256)]

    # Qualitative 
    PLT_PALETTE_SET1 = [plt.cm.Set1(i) for i in range(plt.cm.Set1.N)] # 9 colors
    PLT_PALETTE_SET2 = [plt.cm.Set2(i) for i in range(plt.cm.Set2.N)] # 8 colors
    PLT_PALETTE_SET3 = [plt.cm.Set3(i) for i in range(plt.cm.Set3.N)] # 12 colors
    PLT_PALETTE_SET11 = [plt.cm.Set3(i) for i in range(11)] # 12 colors

    def set_config_init():
        sns.set_style(GSViewer.DEFAULT_SNS_STYLE)
        sns.set_context(GSViewer.DEFAULT_SNS_CONTEXT, font_scale=1.4)
        plt.rcParams['scatter.edgecolors'] = 'black'  # Set edge color for all markers 

    def set_config_dark_background():
        plt.gca().set_facecolor('gray')

    def countplot(df=None, x_name=None, title="", 
                save=False, fname='./countplot.png',
                hue=None, palette=None, dark_background=False
                ):
        factor_x = 1.5
        factor_y = 1.5
        GSViewer.set_config_init()
        fig = plt.figure(figsize=(graph.PLOT_SX*factor_x,graph.PLOT_SY*factor_y), constrained_layout=True)
        sns.countplot(data=df, x=x_name) #, color='blue')
        plt.title(title)

        if dark_background:
            GSViewer.set_config_dark_background()

        if save:
            plt.savefig(fname=fname)
        else:
            plt.show()

        return fig

    def jointplot(df=None, x_name=None, y_name=None, title="", 
                save=False, fname='./jointplot.png', 
                hue=None, kind='kde', palette=None, dark_background=False):
        factor_x = 1.5
        factor_y = 1.5

        GSViewer.set_config_init()

        fig = plt.figure(figsize=(graph.PLOT_SX*factor_x,graph.PLOT_SY*factor_y), constrained_layout=True)
        ax = sns.jointplot(data=df, x=x_name, y=y_name, kind=kind, hue=hue, palette=palette)
         #, color='blue')
        # hue=None, order=None, hue_order=None, orient=None, color=None, palette=None,
        # saturation=0.75, fill=True, dodge='auto', width=0.8, gap=0, whis=1.5, linecolor='auto',
        # linewidth=None, fliersize=None, hue_norm=None, native_scale=False, log_scale=None, formatter=None,
        # legend='auto', ax=None, **kwargs)
        plt.title(title)

        if dark_background:
            GSViewer.set_config_dark_background()

        if save:
            plt.savefig(fname=fname)
        else:
            plt.show()

        return fig

    def boxplot(df=None, x_name=None, y_name=None, title="", 
                save=False, fname='./boxplot.png', count=False,
                hue=None, palette=None, dark_background=False):
        factor_x = 1.5
        factor_y = 1.5
        GSViewer.set_config_init()
        fig = plt.figure(figsize=(graph.PLOT_SX*factor_x,graph.PLOT_SY*factor_y), constrained_layout=True)
        ax = sns.boxplot(data=df, x=x_name, y=y_name, hue=hue, palette=palette) #, color='blue')
        # hue=None, order=None, hue_order=None, orient=None, color=None, palette=None,
        # saturation=0.75, fill=True, dodge='auto', width=0.8, gap=0, whis=1.5, linecolor='auto',
        # linewidth=None, fliersize=None, hue_norm=None, native_scale=False, log_scale=None, formatter=None,
        # legend='auto', ax=None, **kwargs)
        if count:
            # Count samples per category
            counts = df[x_name].value_counts().sort_index(ascending=True)
            #log.info(f"value_counts = {counts}")
            # Annotate the number of samples
            for i, x_category in enumerate(counts.index):
                ax.text(i, df[y_name].max(), f"n={counts[x_category]}", 
                #ax.text(i, 0, f"n={counts[x_category]}", 
                    ha="center", va="bottom", fontsize=12, color="black")

        plt.title(title)

        if dark_background:
            GSViewer.set_config_dark_background()

        if save:
            plt.savefig(fname=fname)
        else:
            plt.show()

        return fig

    def fillplot(df=None, min_name=None, max_name=None, mean_name=None, title="", 
                save=False, fname='./fillplot.png', fill_label="Min-Max Range",
                palette=None, dark_background=False):
        factor_x = 1.5
        factor_y = 1.5

        GSViewer.set_config_init()
        fig = plt.figure(figsize=(graph.PLOT_SX*factor_x,graph.PLOT_SY*factor_y), constrained_layout=True)

        # Plot the shaded region (Min to Max error)
        plt.fill_between(
            df.index, 
            df[min_name], 
            df[max_name], 
            color='red', alpha=0.2, label=fill_label,
        )
        
        # Plot the Mean Error line
        plt.plot(df.index, df[mean_name], color='darkred', linewidth=2, label='Mean MSE')
        
        plt.title(title)
        plt.xlabel("Test Examples (Ranked from Easiest to Hardest)")
        plt.ylabel("Mean Squared Error")
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        if dark_background:
            GSViewer.set_config_dark_background()

        if save:
            plt.savefig(fname=fname)
        else:
            plt.show()

        return fig

    def scatterplot(df=None, ax=None, x_name=None, y_name=None, title="", 
                save=False, fname='./scatterplot.png',
                size=None, sizes=None, median_lines=False,
                hue=None, hue_norm=None, palette=None, dark_background=False):
        factor_x = 1.5
        factor_y = 1.5

        fig = None 
        if ax is None:
            fig = plt.figure(figsize=(graph.PLOT_SX*factor_x,graph.PLOT_SY*factor_y), constrained_layout=True)

        GSViewer.set_config_init()

        sns.scatterplot(ax=ax, data=df, x=x_name, y=y_name, hue=hue, hue_norm=hue_norm, palette=palette, size=size, sizes=sizes) #, color='blue')

        if median_lines:
            x_threshold = df[x_name].median()
            y_threshold = df[y_name].median()

        if ax is None:
            plt.title(title)

            if dark_background:
                GSViewer.set_config_dark_background()

            if median_lines:
                # Add quadrant lines to help interpret the plot
                plt.axvline(x=x_threshold, color='red', linestyle='--')
                plt.axhline(y=y_threshold, color='red', linestyle='--')
    
            if save:
                plt.savefig(fname=fname)
            else:
                plt.show()
        else:
            if median_lines:
                # Add quadrant lines to help interpret the plot
                ax.axvline(x=x_threshold, color='red', linestyle='--')
                ax.axhline(y=y_threshold, color='red', linestyle='--')

        return fig


    def lineplot(ax=None, df=None, x_name=None, y_name=None, title="", 
                save=False, fname='./lineplot.png', hue=None, palette=None,
                style=None, markers=None, scatter=False, dark_background=False):
        factor_x = 1.5
        factor_y = 1.5
        GSViewer.set_config_init()


        fig = None 
        if ax is None:
            fig = plt.figure(figsize=(graph.PLOT_SX*factor_x,graph.PLOT_SY*factor_y), constrained_layout=True)


        sns.lineplot(ax=ax, data=df, x=x_name, y=y_name, 
                          hue=hue, palette=palette, style=style, markers=markers) #, color='blue')
        if scatter:
            sns.scatterplot(ax=ax, data=df, x=x_name, y=y_name, 
                          hue=hue, palette=palette, style=style, legend=False) #, color='blue')

        if ax is None:
            plt.title(title)
            GSViewer.set_config_dark_background()

            if dark_background:
                GSViewer.set_config_dark_background()
    
            if save:
                plt.savefig(fname=fname)
            else:
                plt.show()

        return fig

    def pairplot(df=None, title="", 
                save=False, fname='./pairplot.png', 
                hue=None, palette=None, kind='reg', diag_kind=None,
                x_vars=None, y_vars=None, corner=None,
                ):
        """
        Seaborn pairplot wrapper.
        Kind parameter values: reg, scatter, kde, hist]
        """
        factor_x = 2.
        factor_y = 2.

        GSViewer.set_config_init()

        fig = plt.figure(figsize=(graph.PLOT_SX*factor_x,graph.PLOT_SY*factor_y), constrained_layout=True)
        ax = sns.pairplot(data=df, kind=kind, hue=hue, 
            palette=palette, corner=corner, diag_kind=diag_kind, 
            x_vars=x_vars, y_vars=y_vars,
            )
         #, color='blue')
        plt.suptitle(title, y=1.)
        #GSViewer.set_config_dark_background()

        if save:
            plt.savefig(fname=fname)
        else:
            plt.show()

        return fig

    def pairgrid(df=None, title="", 
                save=False, fname='./pairgrid.png', 
                hue=None, palette=None, map=sns.scatterplot, 
                x_vars=None, y_vars=None, corner=None,
                ):
        """
        Seaborn pairgrid wrapper.
        """
        factor_x = 1.5
        factor_y = 1.5

        GSViewer.set_config_init()

        fig = plt.figure(figsize=(graph.PLOT_SX*factor_x,graph.PLOT_SY*factor_y), constrained_layout=True)
        ax = sns.PairGrid(data=df, hue=hue, 
            palette=palette, corner=corner, 
            x_vars=x_vars, y_vars=y_vars,
            )
        ax.map(map)

         #, color='blue')
        plt.suptitle(title, y=1.)
        #GSViewer.set_config_dark_background()

        if save:
            plt.savefig(fname=fname)
        else:
            plt.show()

        return fig


    @staticmethod
    def lineplot_errorbar(ax=None, df=None, x_name=None, y_name=None,
                          y_lower_name=None, y_upper_name=None,
                          title="", save=False, fname='./lineplot_errorbar.png',
                          hue=None, palette=None,
                          dark_background=False,
                          line_color='blue',
                          error_bar_color='red',
                          capsize=4,
                          xtick_labels=None,
                          hue_offset=0.3):
        """
        Parameters
        ----------
        hue_offset : float
            Total horizontal spread across all hue groups.
            Each group is shifted symmetrically around the true x value.
            E.g., with 3 groups and hue_offset=0.3, shifts are: -0.3, 0.0, +0.3
        """
    
        factor_x = 1.5
        factor_y = 1.5
        GSViewer.set_config_init()
    
        fig = None
        if ax is None:
            fig = plt.figure(
                figsize=(graph.PLOT_SX * factor_x, graph.PLOT_SY * factor_y),
                constrained_layout=True
            )
            ax = fig.gca()
    
        # --- draw lines with error bars ---
        if hue is not None:
            hue_values = df[hue].unique()
            n_hues = len(hue_values)
    
            if palette is not None:
                colors = sns.color_palette(palette, n_colors=n_hues)
            else:
                colors = sns.color_palette(n_colors=n_hues)
    
            # compute per-group offsets centered on zero
            # e.g. n_hues=3, hue_offset=0.3 → [-0.3, 0.0, +0.3]
            if n_hues > 1:
                offsets = np.linspace(-hue_offset, hue_offset, n_hues)
            else:
                offsets = [0.0]
    
            for hue_val, color, dx in zip(hue_values, colors, offsets):
                df_h = df[df[hue] == hue_val].sort_values(x_name)
    
                x_shifted = df_h[x_name].values + dx
    
                yerr_lower = (df_h[y_name] - df_h[y_lower_name]).values
                yerr_upper = (df_h[y_upper_name] - df_h[y_name]).values
    
                ax.errorbar(
                    x_shifted,
                    df_h[y_name].values,
                    yerr=[yerr_lower, yerr_upper],
                    fmt='-o',
                    color=color,
                    ecolor=color,
                    capsize=capsize,
                    capthick=1.5,
                    elinewidth=1.5,
                    linewidth=2,
                    markersize=6,
                    label=f"{hue_val}",
                )
    
            ax.legend(title=str(hue))
    
        else:
            df_sorted = df.sort_values(x_name)
    
            yerr_lower = (df_sorted[y_name] - df_sorted[y_lower_name]).values
            yerr_upper = (df_sorted[y_upper_name] - df_sorted[y_name]).values
    
            ax.errorbar(
                df_sorted[x_name].values,
                df_sorted[y_name].values,
                yerr=[yerr_lower, yerr_upper],
                fmt='-o',
                color=line_color,
                ecolor=error_bar_color,
                capsize=capsize,
                capthick=1.5,
                elinewidth=1.5,
                linewidth=2,
                markersize=6,
            )
    
        # --- x-ticks at true positions (not shifted) ---
        if xtick_labels is not None:
            true_xticks = sorted(df[x_name].unique())
            ax.set_xticks(true_xticks)
            ax.set_xticklabels(xtick_labels)
    
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.grid(True, linewidth=0.3, color='gray', alpha=0.3)
    
        if fig is not None:
            plt.title(title)
    
            if dark_background:
                GSViewer.set_config_dark_background()
    
            if save:
                plt.savefig(fname=fname)
            else:
                plt.show()
    
        return fig


    @staticmethod
    def lineplot_ci(ax=None, df=None, x_name=None, y_name=None,
                    y_lower_name=None, y_upper_name=None,
                    title="", save=False, fname='./lineplot_ci.png',
                    hue=None, palette=None, style=None, markers=None,
                    scatter=False, dark_background=False, ci_alpha=0.2):
    
        factor_x = 1.5
        factor_y = 1.5
        GSViewer.set_config_init()
    
        fig = None
        if ax is None:
            fig = plt.figure(
                figsize=(graph.PLOT_SX * factor_x, graph.PLOT_SY * factor_y),
                constrained_layout=True
            )
            ax = fig.gca()
    
        # --- lines ---
        sns.lineplot(
            ax=ax, data=df, x=x_name, y=y_name,
            hue=hue, palette=palette, style=style, markers=markers
        )
    
        if scatter:
            sns.scatterplot(
                ax=ax, data=df, x=x_name, y=y_name,
                hue=hue, palette=palette, style=style, legend=False
            )
    
        # --- confidence bands ---
        if y_lower_name is not None and y_upper_name is not None:
    
            # match seaborn's palette to keep band colors consistent with lines
            if hue is not None:
                hue_values = df[hue].unique()
                if palette is not None:
                    color_palette = sns.color_palette(palette, n_colors=len(hue_values))
                else:
                    color_palette = sns.color_palette(n_colors=len(hue_values))
    
                for hue_val, color in zip(hue_values, color_palette):
                    df_h = df[df[hue] == hue_val].sort_values(x_name)
                    ax.fill_between(
                        df_h[x_name],
                        df_h[y_lower_name],
                        df_h[y_upper_name],
                        alpha=ci_alpha,
                        color=color,
                    )
            else:
                df_sorted = df.sort_values(x_name)
                ax.fill_between(
                    df_sorted[x_name],
                    df_sorted[y_lower_name],
                    df_sorted[y_upper_name],
                    alpha=ci_alpha,
                )
    
        if fig is not None:
            plt.title(title)
    
            if dark_background:
                GSViewer.set_config_dark_background()
    
            if save:
                plt.savefig(fname=fname)
            else:
                plt.show()
    
        return fig


    def heatmap(df=None, fname=None, save=False, title=None, vmin=None, vmax=None, cmap=None,
                center=None, robust=False, annot=None, 
                fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', 
                cbar=True, cbar_kws=None, cbar_ax=None, 
                square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None,
                **kwargs
                ):
        """
        Heatmap wrapper.
        """
        factor_x = 1.5
        factor_y = 1.5

        GSViewer.set_config_init()

        ax_ = ax
        fig = None 
        if ax is None:
            fig = plt.figure(figsize=(graph.PLOT_SX*factor_x,graph.PLOT_SY*factor_y), constrained_layout=True)
            ax_ = fig.subplots()


        sns.heatmap(
                df, vmin=vmin, vmax=vmax, cmap=cmap, center=center, robust=robust, annot=annot, 
                fmt=fmt, annot_kws=annot_kws, linewidths=linewidths, linecolor=linecolor, 
                cbar=cbar, cbar_kws=cbar_kws, cbar_ax=cbar_ax, 
                square=square, xticklabels=xticklabels, yticklabels=yticklabels, mask=mask, ax=ax_,
                **kwargs
            )
        
        ax_.set_yticklabels(ax_.get_yticklabels(), fontsize=15, rotation=0)
        ax_.set_xticklabels(ax_.get_xticklabels(), fontsize=11, rotation=90)


        if ax is None:
            plt.suptitle(title, y=1.) 
            GSViewer.set_config_dark_background()

            if save:
                plt.savefig(fname=fname)
            else:
                plt.show()

        return fig
    
    def violin(df=None, fname=None, x_name=None, y_name=None, save=False, title=None, y_order=None,
                x_label=None, y_label=None, errorbar_callback=None,
                vmin=None, vmax=None, cmap=None,
                center=None, robust=False, annot=None, 
                fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', 
                cbar=True, cbar_kws=None, cbar_ax=None, 
                square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None,
                alpha=0.6,  **kwargs
                ):
        """
        Violin wrapper.
        """
        factor_x = 1.5
        factor_y = 1.5

        GSViewer.set_config_init()

        fig = None 
        if ax is None:
            fig = plt.figure(figsize=(graph.PLOT_SX*factor_x,graph.PLOT_SY*factor_y), constrained_layout=True)

        sns.set_theme(style="whitegrid")

        #y_order = df[y_name].unique()
        
        # 3. Draw the violins (inner=None removes the default boxplot)
        ax = sns.violinplot( data=df, inner=None, x=x_name, y=y_name, order=y_order, color="lightgray", linewidth=1, alpha=0.6)
            #    fmt=fmt, annot_kws=annot_kws, linewidths=linewidths, linecolor=linecolor, 
            #    cbar=cbar, cbar_kws=cbar_kws, cbar_ax=cbar_ax, 
            #    square=square, xticklabels=xticklabels, yticklabels=yticklabels, mask=mask, ax=ax, **kwargs
            #)

        if errorbar_callback is None:
            errorbar_callback=('ci', 95)

        # 4. Overlay the Mean and 95% Confidence Interval
        ax2 = sns.pointplot(
            data=df, 
            x=x_name, 
            y=y_name, 
            estimator='mean',           # Plot the mean
            errorbar=errorbar_callback,        # Calculate and draw the 95% CI
            color="red",                # Make the CI stand out
            markers="D",                # Use a diamond marker for the mean
            order=y_order,
            capsize=0.1,                # Add caps to the ends of the error bars
            linestyle="none",           # Don't connect the different models with a line
            linewidth=1.2,           # Don't connect the different models with a line
        )
        
        # 5. Clean up the labels
        plt.title(f"Model {x_name.upper()} Distributions with 95% Confidence Intervals", fontsize=14, pad=15)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        
        plt.tight_layout()

        if save:
            plt.savefig(fname=fname)
        else:
            plt.show()

        return fig

