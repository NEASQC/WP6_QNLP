import matplotlib.pyplot as plt 
import numpy as np
import random as rd 


def plotting_function(
    x: list[list[float]], y: list[list[float]], y_stds: list[list[float]],  
    legend_size: float = 25.0, alpha: float = 0.25, 
    ymax: float = None, ymin: float = None,
    xmax: float = None, xmin: float = None,
    title: str = None, xlabel: str = None,
    ylabel: str = None, **plt_kwargs: dict) -> plt.figure:
    """
    Creates a matplolib.pyplot.figure object with the plot 
    of a function with its standard deviation displayed
    in lighter shade. Additional arguments can be
    introduced to customise the plots. 

    Parameters
    ----------
    x : [list[list[float]]]
        Nested list containing the x-axis values of the functions we want to
        plot. 
    y : list[list[float]]
        Nested list containing the y-axis values of the functions we want to
        plot.
    y_stds : list[list[float]]
        Nested containig all the standard deviations values for the
        functions in y as nested lists.
        We are assuming here that each float in y has its own y_stds value
        assigned.
    mysize : float, optional
        Font size of legend of the plot (in points). Default is 25.
    alpha : float, optional
        Value that sets the opacity of standard deviation plots.
    ymax : float, optional
        Sets the maximum value for y axis plot. If it's None, the value 
        will be computed as as sum of the maximum value of y plus 
        the maximum value of y_stds. 
    ymin : float, optional
        Sets the maximum value for y axis plot. If it's None, the value 
        will be computed as as sum of the minimum value of y plus 
        the maximum value of y_stds. 
    xmax : float, optional
        Sets the maximum value for x axis plot. If it's None the value 
        will be computed as the maximum value of x. 
    xmin : float, optional
        Sets the minimum value for x axis plot. If it's None the value 
        will be computed as the maximum value of x.
    title : str, optional
        Sets the title of the plot.
    xlabel : str, optional
        Sets the label of the x-axis of the plot.
    ylabel : str, optional
        Sets the label of the y-axis of the plot.
    **plt_kwargs : dict
        Dictionary containing additional plot parameters
        (of matplotlib.pyplot.plot method) we want to introduce in the plot. 
        Each key of the dictionary will have as a value a list specyfing the
        desired parameter for each of the functions.
        For example if we want to label 3 plotted functions as 'a','b','c'
        and color them with 'red', 'blue', 'green', **plt_kwargs should be:
        **plt_kwargs = {'label': ['a', 'b', 'c'],
                        'color'; ['red', 'blue', 'green']}
    Returns
    -------
    figure : plt.figure

    """
    
    
    n_plots = len(y)
    if ymax == None:
        ymax = max([max(i) for i in y]) + max([max(i) for i in y_stds])
    if ymin == None: 
        ymin = min([min(i) for i in y]) - max([max(i) for i in y_stds])
    if xmax == None: 
        xmax = max([max(i) for i in x])
    if xmin == None:
        xmin = min([min(i) for i in x])

    figure, axis = plt.subplots(1, 1, figsize =(12,12))
    axis.set_ylim(ymin, ymax) 
    axis.set_xlim(xmin, xmax)
    for i in range(n_plots):
        optional_args = {} ;steps = len(y[i])
        for k in plt_kwargs.keys():
            optional_args[k] = plt_kwargs[k][i] 
        axis.plot(x[i], y[i], **optional_args)
        optional_args.pop('label', None)
        underline = [y[i][j] - y_stds[i][j] for j in range(steps)]
        overline = [y[i][j] + y_stds[i][j] for j in range(steps)]
        axis.fill_between(
            x[i], underline, overline, alpha=alpha, **optional_args)
    
    if 'legend' in plt_kwargs.keys():
        axis.legend(fontsize=legend_size) 
    if title != None:
        axis.set_title(title)
    if xlabel != None:
        axis.set_xlabel(xlabel)
    if ylabel != None:
        axis.set_ylabel(ylabel)

    return figure


def save_plot(figure: plt.figure, path: str, name: str):
    """
    figure : plt.figure
        Figure we want to save 
    path : str
        Path where store the generated graph
    name : str
        Name to be given to the stored graph
    """
    figure.savefig(path+f'{name}.png')
    
    
