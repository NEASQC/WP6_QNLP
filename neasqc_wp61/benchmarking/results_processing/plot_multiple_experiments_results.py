import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import plotly.graph_objects as go
from statistics import mean, stdev
from plotly.subplots import make_subplots
import numpy as np
import os
import plotly

def plot_multiple_experiment_results(json_files, plot_type):
    model_names = []
    iterations_list = []
    mean_val_accuracies_list = []
    std_val_accuracies_list = []
    mean_val_losses_list = []
    std_val_losses_list = []
    mean_train_accuracies_list = []
    std_train_accuracies_list = []
    mean_train_losses_list = []
    std_train_losses_list = []
    times_list = []

    dict_new_model_names = {"pre_alpha": "Pre_alpha 1",
                            "pre_alpha_lambeq": "Pre_alpha 2",
                            "alpha_original": "Alpha 1",
                            "alpha_lambeq": "Alpha 2",
                            "alpha_pennylane": "Alpha 3"
                            }

    for json_file in json_files:
        model_name, iterations, mean_val_accuracies, std_val_accuracies, mean_val_losses, std_val_losses, mean_train_accuracies, std_train_accuracies, mean_train_losses, std_train_losses, times = get_experiment_results(json_file)
        model_names.append(dict_new_model_names[model_name])
        iterations_list.append(iterations)
        mean_val_accuracies_list.append(mean_val_accuracies)
        std_val_accuracies_list.append(std_val_accuracies)
        mean_val_losses_list.append(mean_val_losses)
        std_val_losses_list.append(std_val_losses)
        mean_train_accuracies_list.append(mean_train_accuracies)
        std_train_accuracies_list.append(std_train_accuracies)
        mean_train_losses_list.append(mean_train_losses)
        std_train_losses_list.append(std_train_losses)
        times_list.append(times)

    if iterations_list.count(iterations_list[0]) != len(iterations_list):
        raise ValueError("The number of iterations is not the same for all experiments.")
   

    fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Training results", "Validation results"))


    x_values = list(range(1, iterations_list[0] + 1))  # Convert range to a list


    if plot_type == 'loss':
        # Plot the loss
        for i in range(len(json_files)):
            interactive_plotly_plot(x_values, mean_train_losses_list[i], std_train_losses_list[i], 1, 1, model_names[i], fig, color=i, showlegend=True)
            interactive_plotly_plot(x_values, mean_val_losses_list[i], std_val_losses_list[i], 1, 2, model_names[i], fig, color=i, showlegend=False)

            fig.update_layout(title='Experiment Results: ',
                        hovermode="x",
                        xaxis1_title = "Iterations",
                        xaxis2_title = "Iterations",
            )

        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)

        fig.update_layout(yaxis1 = dict(range=[0, 2]))    # loss
        fig.update_layout(yaxis2 = dict(range=[0, 2]))    # loss

            
    elif plot_type == 'accuracy':
        # Plot the accuracy
        for i in range(len(json_files)):
            interactive_plotly_plot(x_values, mean_train_accuracies_list[i], std_train_accuracies_list[i], 1, 1, model_names[i], fig, color=i, showlegend=True)
            interactive_plotly_plot(x_values, mean_val_accuracies_list[i], std_val_accuracies_list[i],1, 2, model_names[i], fig, color=i, showlegend=False)

            fig.update_layout(title='Experiment Results: ',
                        hovermode="x",
                        xaxis1_title = "Iterations",
                        xaxis2_title = "Iterations",
            )

        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)

        fig.update_layout(yaxis1 = dict(range=[0, 1]))    # acc
        fig.update_layout(yaxis2 = dict(range=[0, 1]))    # acc



 
    fig['layout']['xaxis'].update(autorange = True)

    fig.show()


def interactive_plotly_plot(x_values, y_values, y_std, row_index, col_index, name, fig, color, showlegend):
    # Define a colormap (you can choose any colormap of your preference)
    #cmap = ["rgba(239,85,59,1)", "rgba(99,110,250,1)", "rgba(0,202,148,1)", "rgba(171,99,250,1)", "rgba(255,102,146,1)"]
    cmap = ["rgba(99,110,250,1)", "rgba(0,204,150,1)",  "rgba(239,85,59,1)", "rgba(0,0,0,1)", "rgba(171,99,250,1)"]

    fig.add_trace(
        go.Scatter(x=x_values,y=y_values, mode='lines', showlegend=showlegend, name=name, line_color=cmap[color], line={'width': 1}),
        row=row_index, col=col_index
    )

    fig.add_trace(
        go.Scatter(x=x_values,y=y_values + y_std, mode='lines', showlegend=False, name=name, line_color=cmap[color], line={'width': 0}, hoverinfo='skip'),
        row=row_index, col=col_index
    )

    rgba_fill_color = cmap[color].replace('1)', '0.2)')

    fig.add_trace(
        go.Scatter(x=x_values,y=y_values - y_std, mode='lines', showlegend=False, name=name, line_color=cmap[color], line={'width': 0}, fill='tonexty', fillcolor=rgba_fill_color, hoverinfo='skip'),
        row=row_index, col=col_index
    )



def get_experiment_results(json_file):
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract relevant data from the JSON
    experiment_count = data['input_args']['runs']
    iterations = data['input_args']['iterations']

    model_name = "_".join(json_file.split('/')[-1].split('_')[:-1])

    # Create empty lists to store the accuracy and loss values
    val_accuracies = []
    val_losses = []
    train_accuracies = []
    train_losses = []

    times = []

    # Extract the accuracy and loss values for each runs
    if data.get('val_acc') is not None:
        for i in range(experiment_count):
            val_acc = data['val_acc'][i]
            val_accuracies.append(val_acc)
        val_accuracies = np.array(val_accuracies)

        mean_val_accuracies = np.mean(val_accuracies, axis=0)
        std_val_accuracies = np.std(val_accuracies, axis=0)
    else:
        print("No validation accuracy data found.")
       
       
    if data.get('val_loss') is not None:
        for i in range(experiment_count):
            val_loss = data['val_loss'][i]
            val_losses.append(val_loss)
        val_losses = np.array(val_losses)

        mean_val_losses = np.mean(val_losses, axis=0)
        std_val_losses = np.std(val_losses, axis=0)
    else:
        print("No validation loss data found.")

    if data.get('train_acc') is not None:
        for i in range(experiment_count):
            train_acc = data['train_acc'][i]
            train_accuracies.append(train_acc)
        train_accuracies = np.array(train_accuracies)

        mean_train_accuracies = np.mean(train_accuracies, axis=0)
        std_train_accuracies = np.std(train_accuracies, axis=0)
    else:
        print("No training accuracy data found.")

    if data.get('train_loss') is not None:
        for i in range(experiment_count):
            train_loss = data['train_loss'][i]
            train_losses.append(train_loss)
        train_losses = np.array(train_losses)

        mean_train_losses = np.mean(train_losses, axis=0)
        std_train_losses = np.std(train_losses, axis=0)
    else:
        print("No training loss data found.")

    # Extract the mean time
    if isinstance(data['time'], list):
        # We have a list of values
        for i in range(experiment_count):
            times.append(data['time'][i])
    else:
        # We have a single value
        times.append(data['time'])

    return model_name, iterations, mean_val_accuracies, std_val_accuracies, mean_val_losses, std_val_losses, mean_train_accuracies, std_train_accuracies, mean_train_losses, std_train_losses, times
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot experiment results from mutiple JSON files.')
    parser.add_argument('--files', nargs='+', help='List of JSON files to plot')
    parser.add_argument('--folder', help='Folder containing JSON files to plot')    
    parser.add_argument('--plot', help='Plot either the loss or the accuracy', choices=['loss', 'accuracy'], default='loss')

    args = parser.parse_args()

    if args.files:
        json_files = args.files
    elif args.folder:
        folder_path = args.folder
        json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    else:
        print("Please provide either --files or --folder argument.")
        exit()

    if not json_files:
        print("No JSON files found.")
        exit()


    plot_multiple_experiment_results(json_files, args.plot)
