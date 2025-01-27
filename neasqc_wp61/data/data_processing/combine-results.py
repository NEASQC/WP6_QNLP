import os
import glob
import json
import math
import argparse

# Set up command-line arguments
parser = argparse.ArgumentParser(description="Process results.json files and aggregate statistics.")
parser.add_argument("base_dir", help="Base directory to search for results.json files.")
parser.add_argument("res_dir", help="Base directory where to save TSV output files.")
args = parser.parse_args()

# 95% CI z-score
Z = 1.96

# Output filenames
output_files = {
    'test_acc': args.res_dir + '/test_acc.tsv',
    'test_loss': args.res_dir + '/test_loss.tsv',
    'time': args.res_dir + '/time.tsv',
    'val_acc': args.res_dir + '/val_acc.tsv',
    'val_loss': args.res_dir + '/val_loss.tsv',
    'train_acc': args.res_dir + '/train_acc.tsv',
    'train_loss': args.res_dir + '/train_loss.tsv',
}

# Initialize file handles and write headers
handles = {}
for key, fname in output_files.items():
    handles[key] = open(fname, 'w', encoding='utf-8')
    # Write a header line (optional)
    handles[key].write("path\tmean\tmean_minus_CI\tmean_plus_CI\tCI\n")

def compute_mean_ci(values):
    """Compute mean and 95% CI of a list of numeric values."""
    if not values:
        return (float('nan'), float('nan'), float('nan'), float('nan'))
    mean_val = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((x - mean_val)**2 for x in values) / (len(values) - 1)
        std_dev = math.sqrt(variance)
        std_err = std_dev / math.sqrt(len(values))
        ci = Z * std_err
    else:
        ci = 0.0

    return (mean_val, mean_val - ci, mean_val + ci, ci)

# Find all results.json files
for file_path in glob.iglob(os.path.join(args.base_dir, '**', 'results.json'), recursive=True):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    best_run_idx = data["best_run"]  # Adjust if this is one-based indexing

    # Extract arrays
    test_acc = data["test_acc"]
    test_loss = data["test_loss"]
    time_arr = data["time"]

    val_acc_run = data["val_acc"][best_run_idx]
    val_loss_run = data["val_loss"][best_run_idx]
    train_acc_run = data["train_acc"][best_run_idx]
    train_loss_run = data["train_loss"][best_run_idx]

    # Compute mean and CI for each
    test_acc_stats = compute_mean_ci(test_acc)
    test_loss_stats = compute_mean_ci(test_loss)
    time_stats = compute_mean_ci(time_arr)
    val_acc_stats = compute_mean_ci(val_acc_run)
    val_loss_stats = compute_mean_ci(val_loss_run)
    train_acc_stats = compute_mean_ci(train_acc_run)
    train_loss_stats = compute_mean_ci(train_loss_run)

    # Write a single line of results to each file
    def write_line(handle, path, stats):
        mean_val, mean_minus_ci, mean_plus_ci, ci = stats
        handle.write(f"{path}\t{mean_val}\t{mean_minus_ci}\t{mean_plus_ci}\t{ci}\n")

    write_line(handles['test_acc'], file_path, test_acc_stats)
    write_line(handles['test_loss'], file_path, test_loss_stats)
    write_line(handles['time'], file_path, time_stats)
    write_line(handles['val_acc'], file_path, val_acc_stats)
    write_line(handles['val_loss'], file_path, val_loss_stats)
    write_line(handles['train_acc'], file_path, train_acc_stats)
    write_line(handles['train_loss'], file_path, train_loss_stats)

# Close all files
for f in handles.values():
    f.close()
