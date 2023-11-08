import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import plotly.graph_objects as go
from statistics import mean, stdev
from plotly.subplots import make_subplots
import numpy as np

def plot_experiment_results(json_file):
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract relevant data from the JSON
    experiment_count = data['input_args']['runs']
    iterations = data['input_args']['iterations']

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
       
       
    if data.get('val_loss') is not None:
        for i in range(experiment_count):
            val_loss = data['val_loss'][i]
            val_losses.append(val_loss)
        val_losses = np.array(val_losses)

        mean_val_losses = np.mean(val_losses, axis=0)
        std_val_losses = np.std(val_losses, axis=0)

    if data.get('train_acc') is not None:
        for i in range(experiment_count):
            train_acc = data['train_acc'][i]
            train_accuracies.append(train_acc)
        train_accuracies = np.array(train_accuracies)

        mean_train_accuracies = np.mean(train_accuracies, axis=0)
        std_train_accuracies = np.std(train_accuracies, axis=0)

    if data.get('train_loss') is not None:
        for i in range(experiment_count):
            train_loss = data['train_loss'][i]
            train_losses.append(train_loss)
        train_losses = np.array(train_losses)

        mean_train_losses = np.mean(train_losses, axis=0)
        std_train_losses = np.std(train_losses, axis=0)

    # Extract the mean time
    if isinstance(data['time'], list):
        # We have a list of values
        for i in range(experiment_count):
            times.append(data['time'][i])
    else:
        # We have a single value
        times.append(data['time'])




    # Experiment Configuration Summary
    input_args = data['input_args']
    summary = f"Experiment Configuration:\n{json.dumps(input_args, indent=4)}"
    print(summary)
    print('\n')

    if data.get('best_val_acc') is not None:
        #Alpha runs
        result_keys = ['best_val_acc', 'best_run']

    else:
        #Pre_alpha and Beta runs
        result_keys = ['best_final_val_acc', 'best_run']

    summary_results_dict = {key: data[key] for key in result_keys}
    summary_results_dict['mean_time'] = np.mean(times)

    summary_results = f"Experiment Results:\n{json.dumps(summary_results_dict, indent=4)}"
    print(summary_results)

    summary = summary.replace("\n", "<br>")
    summary_results = summary_results.replace("\n", "<br>")

    print(len(mean_train_losses))
    print(len(list(range(1, iterations + 1)) ))
    # We have 3 case scenarios:
    # 1. We have all the data
    # 2. We have only the accuracy data
    # 3. We have only the loss data

    if (data.get('val_acc') is None or data.get('train_acc') is None) or (data.get('val_loss') is None or data.get('train_loss') is None):
        print("Accuracy or Loss missing from JSON file.")
         # Initialize figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.8, 0.2],
            row_heights=[0.5, 0.5],
            specs=[[{"type": "scatter"}, None],
                [{"type": "scatter"}, None]])
    else:
        # We have all the data
        # Initialize figure with subplots
        fig = make_subplots(
            rows=2, cols=3,
            column_widths=[0.4, 0.4, 0.1],
            row_heights=[0.5, 0.5],
            specs=[[{"type": "scatter"}, {"type": "scatter"}, None],
                [{"type": "scatter"}, {"type": "scatter"}, None]])


    x_values = list(range(1, iterations + 1))  # Convert range to a list

    col_index = 1

    if data.get('train_loss') is not None and data.get('val_loss') is not None:
        # Add scatter training loss trace
        rgba_line_color = "rgba(239,85,59,1)"
        rgba_fill_color = "rgba(239,85,59,0.2)"
        interactive_plotly_plot(x_values, mean_train_losses, std_train_losses, 1, col_index, "Training Loss", fig, rgba_line_color, rgba_fill_color)
        
        # Add scatter validation loss trace
        rgba_line_color = "rgba(99,110,250,1)"
        rgba_fill_color = "rgba(99,110,250,0.2)"
        interactive_plotly_plot(x_values, mean_val_losses, std_val_losses, 2, col_index, "Validation Loss", fig, rgba_line_color, rgba_fill_color)
        col_index += 1


    if data.get('train_acc') is not None and data.get('val_acc') is not None:
        # Add scatter training accuracy trace
        rgba_line_color = "rgba(0,202,148,1)"
        rgba_fill_color = "rgba(0,202,148,0.2)"
        interactive_plotly_plot(x_values, mean_train_accuracies, std_train_accuracies, 1, col_index, "Training Accuracy", fig, rgba_line_color, rgba_fill_color)

        # Add scatter validation accuracy trace
        rgba_line_color = "rgba(171,99,250,1)"
        rgba_fill_color = "rgba(171,99,250,0.2)"
        interactive_plotly_plot(x_values, mean_val_accuracies, std_val_accuracies, 2, col_index, "Validation Accuracy", fig, rgba_line_color, rgba_fill_color)

        
    
    if data.get('val_acc') is None or data.get('train_acc') is None:
        # 2. We have only the accuracy data
        fig.update_layout(title='Experiment Results: ' + json_file,
                        legend=dict(x=0, y=1),
                        xaxis=dict(range=[0, 3]),
                        hovermode="x",
                        yaxis1_title = "Training Loss",
                        yaxis2_title = "Validation Loss",
                        xaxis1_title = "Iterations",
                        xaxis2_title = "Iterations",
                        annotations=[
                            go.layout.Annotation(
                                text=summary + '<br><br>' +summary_results,
                                align='left',
                                showarrow=False,
                                xref='paper',
                                yref='paper',
                                bordercolor='black',
                                borderwidth=1,
                                x=1.2,
                                y=0.6,
                            )
                        ]
        )

    elif data.get('val_loss') is None or data.get('train_loss') is None:
        # 3. We have only the loss data
        fig.update_layout(title='Experiment Results: ' + json_file,
                        legend=dict(x=0, y=1),
                        xaxis=dict(range=[0, 3]),
                        hovermode="x",
                        yaxis1_title = "Training Accuracy",
                        yaxis2_title = "Validation Accuracy",
                        xaxis1_title = "Iterations",
                        xaxis2_title = "Iterations",
                        annotations=[
                            go.layout.Annotation(
                                text=summary + '<br><br>' +summary_results,
                                align='left',
                                showarrow=False,
                                xref='paper',
                                yref='paper',
                                bordercolor='black',
                                borderwidth=1,
                                x=1.2,
                                y=0.6,
                            )
                        ]
        )
    else:
        # 1. We have all the data
        fig.update_layout(title='Experiment Results: ' + json_file,
                        legend=dict(x=0, y=1),
                        xaxis=dict(range=[0, 3]),
                        hovermode="x",
                        yaxis1_title = "Training Loss",
                        yaxis2_title = "Training Accuracy",
                        yaxis3_title = "Validation Loss",
                        yaxis4_title = "Validation Accuracy",
                        xaxis1_title = "Iterations",
                        xaxis2_title = "Iterations",
                        xaxis3_title = "Iterations",
                        xaxis4_title = "Iterations",
                        annotations=[
                            go.layout.Annotation(
                                text=summary + '<br><br>' +summary_results,
                                align='left',
                                showarrow=False,
                                xref='paper',
                                yref='paper',
                                bordercolor='black',
                                borderwidth=1,
                                x=1.2,
                                y=0.6,
                            )
                        ]
        )

        fig.update_layout(yaxis1 = dict(range=[0, 2]))    # loss
        fig.update_layout(yaxis2 = dict(range=[0, 1]))    # acc
        fig.update_layout(yaxis3 = dict(range=[0, 2])) # loss
        fig.update_layout(yaxis4 = dict(range=[0, 1])) # acc

    
    fig['layout']['xaxis'].update(autorange = True)

    fig.show()


def interactive_plotly_plot(x_values, y_values, y_std, row_index, col_index, name, fig, rgba_line_color, rgba_fill_color):

    fig.add_trace(
        go.Scatter(x=x_values, y=y_values, mode='lines', showlegend=False, name=name, line={'width': 1},
                    line_color=rgba_line_color),
        row=row_index, col=col_index
    )
    

    fig.add_trace(
        go.Scatter(x=x_values, y=y_values + y_std, mode='lines', line={'width': 0},
                   fillcolor=rgba_fill_color, line_color=rgba_line_color, showlegend=False, name='Upper Bound'),
        row=row_index, col=col_index
    )
    fig.add_trace(
        go.Scatter(x=x_values, y=y_values - y_std, mode='lines', line={'width': 0}, fill='tonexty',
                   fillcolor=rgba_fill_color, line_color=rgba_line_color, showlegend=False, name='Lower Bound'),
        row=row_index, col=col_index
    )

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot experiment results from a JSON file.')
    parser.add_argument('json_file', type=str, help='Path to the JSON file with experiment results')

    args = parser.parse_args()

    plot_experiment_results(args.json_file)
