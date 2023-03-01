import sys 
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/results_processing/")
from pre_alpha_benchmarking import benchmarking_pipeline
from plotting_tool import plotting_function, save_plot


name_plot = 'plot_004'
name_results = 'results_004'

path_results = current_path + "/results/raw/"
path_plot = current_path + "/results/analysed/"

x, y, y_stds = benchmarking_pipeline(1, save_results=True,
path=path_results, name=name_results)
figure = plotting_function([x], [y], [y_stds],
title = 'Results pre-alpha ID 004',
xlabel = 'Optimization step', ylabel = 'Cost function')
save_plot(figure, path_plot, name=name_plot)