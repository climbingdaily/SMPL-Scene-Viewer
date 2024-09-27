import argparse
from math import isnan
import re
import matplotlib.pyplot as plt
import numpy as np
# import colorsys

# Cyberpunk color schemes
cyberpunk_blue = {
    'primary': '#359dcd',#359dcd
    'secondary': '#1E2A38',
    'file2_primary': '#84b1f4',       
    'file2_secondary': '#285498',       
    # Shade for File 2
    'text': '#8FBED8'
}

cyberpunk_pink = {
    'primary': '#f27372',
    'secondary': '#38182F',
    'file2_primary': '#EA8C8B',
    'file2_secondary': '#38182F',
    'text': '#EB6F92',
    'file2': '#E85C8D'        # Shade for File 2
}


color_schemes = [cyberpunk_blue, cyberpunk_pink]

# color_schemes[0]['primary'] = '#84b1f4'  # RGB for Person 1
# color_schemes[1]['primary'] = '#EA8C8B'  # RGB for Person 2
# color_schemes[0]['file2_primary'] = "#93d4c9"
# color_schemes[1]['file2_primary'] = "#bfcd60"


def parse_log_file(filename):
    with open(filename, "r") as f:
        log_content = f.readlines()

    segments_data = []
    current_segment = None
    current_data = []

    for line in log_content:
        segment_match = re.search(r"\[Segments (\d+)/(\d+)\]", line)
        if segment_match:
            if current_segment:
                segments_data.append({
                    'segment': current_segment,
                    'data': current_data
                })
            current_segment = (int(segment_match.group(1)), int(segment_match.group(2)))
            current_data = []
        else:
            loss_match = re.search(r"^\[.*?\] (\d+)", line)
            if loss_match:
                iteration = int(loss_match.group(1))
                loss_data = re.findall(r"(\w+) (\d+\.\d+)", line)
                current_data.append({
                    'iteration': iteration,
                    'losses': {k: float(v) for k, v in loss_data}
                })
    if current_segment:
        segments_data.append({
            'segment': current_segment,
            'data': current_data
        })
    return segments_data

def group_by_individual(segments_data):
    individuals_data = []
    current_individual = []

    for segment_data in segments_data:
        if segment_data['segment'][0] == 1:
            if current_individual:
                individuals_data.append(current_individual)
            current_individual = [segment_data]
        else:
            current_individual.append(segment_data)
    if current_individual:
        individuals_data.append(current_individual)
    return individuals_data

def plot_average_loss_time_cyberpunk_v2(avg_values1, avg_values2, max_iter, max_iter2, loss_type, person_label, color_scheme):
    plt.figure(figsize=(5, 3), dpi=200)
    
    
    if avg_values2 is not None:
        # Plot for file2 (second file)
        plt.plot(avg_values2[loss_type], 
                 label=f"W/o Multi-stage Optimization", 
                 linestyle='--', 
                 color=color_scheme['file2_primary'])
        plt.fill_between(range(max_iter2 + 1), 
                         avg_values2[loss_type], 
                         color=color_scheme['file2_secondary'], alpha=0.2)

    plt.tick_params(axis='both', which='both', direction='in')
    plt.plot(avg_values1[loss_type], label="W/ Multi-stage Optimization", color=color_scheme['primary'])
    plt.fill_between(range(max_iter + 1), avg_values1[loss_type], color=color_scheme['secondary'], alpha=0.2)
    # plt.title(f"{loss_type} ({person_label})")
    plt.xlabel("Iterations")
    plt.ylabel(f"{loss_type} (s)" if loss_type == 'Time' else f"{loss_type}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.legend()
    # plt.show()

def plot_individual_losses(individuals_data1, individuals_data2=None):
    all_loss_types = list(set([loss 
                               for segment_data in segments_data 
                               for data in segment_data['data']
                               for loss in data['losses'].keys()]))
    labels = ["First person", "Second person"]

    for idx, individual_data1 in enumerate(individuals_data1, start=1):
        person_label = labels[idx-1]
        color_scheme = color_schemes[idx-1]
        avg_time_loss_values1, max_iter = compute_average_values(individual_data1, ["Loss", "Time"])

        if individuals_data2:
            avg_time_loss_values2, max_iter2 = compute_average_values(individuals_data2[idx-1], 
                                                              ["Loss", "Time"])

        else:
            avg_time_loss_values2 = None

        plot_average_time_vs_loss_both_files_dashed(avg_time_loss_values1, 
                                                    avg_time_loss_values2, 
                                                    person_label, color_scheme)
        for loss_type in all_loss_types:
            if loss_type not in ["Loss", "Time"]:
                continue
            avg_values1, _ = compute_average_values(individual_data1, [loss_type])
            if individuals_data2:
                avg_values2, _ = compute_average_values(individuals_data2[idx-1], [loss_type])
            else:
                avg_values2 = None
            plot_average_loss_time_cyberpunk_v2(avg_values1,
                                                avg_values2,
                                                max_iter, max_iter2,
                                                loss_type, 
                                                labels[idx-1], 
                                                color_schemes[idx-1])

    plt.show()

def plot_average_time_vs_loss_both_files_dashed(avg_values1, 
                                                avg_values2, 
                                                person_label, 
                                                color_scheme):
    plt.figure(figsize=(5, 3), dpi=200)
    print(f"Iters| {len(avg_values1['Time'])}")
    print(f"{person_label} | W/ Multi-stage | Time: {avg_values1['Time'][-1]:.2f} | loss: {avg_values1['Loss'][-1]:.2f} | Iters: {len(avg_values1['Time'])}")
    if avg_values2 is not None:
        # Plot for file2 (second file)
        # avg_time_values2, avg_loss_values2 = compute_average_values(individual_data2)
        plt.plot(avg_values2['Time'], avg_values2['Loss'], 
                color=color_scheme['file2_primary'], 
                linestyle='--', 
                label=f"W/o Multi-stage Optimization", linewidth=1.5)
        plt.fill_between(avg_values2['Time'], avg_values2['Loss'], 
                        color=color_scheme['file2_secondary'], alpha=0.2)
        plt.tick_params(axis='both', which='both', direction='in')
        print(f"{person_label} | W/o Multi-stage | Time: {avg_values2['Time'][-1]:.2f} | loss: {avg_values2['Loss'][-1]:.2f} | Iters: {len(avg_values2['Time'])}")
    
    # Plot for file1 (first file)
    # avg_time_values1, avg_loss_values1 = compute_average_values(individual_data1)
    plt.plot(avg_values1['Time'], avg_values1['Loss'], 
             color=color_scheme['primary'], 
             label=f"W/ Multi-stage Optimization", linewidth=2)
    plt.fill_between(avg_values1['Time'], avg_values1['Loss'], 
                     color=color_scheme['secondary'], alpha=0.2)
    plt.tick_params(axis='both', which='both', direction='in')

    plt.xlabel("Time (s)")
    plt.ylabel("Loss")
    plt.title(f"Loss Convergence over Time ({person_label})")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    # plt.show()

def compute_average_values(individual_data, loss_types = ['Time', 'Loss']):
    max_iter = max([item['iteration'] for segment in individual_data for item in segment['data']])
    combined_values = {lt: [0] * (max_iter + 1) for lt in loss_types}
    count_values = [0] * (max_iter + 1)
    
    for segment in individual_data:
        value = {}
        for item in segment['data']:
            value = {lt: item['losses'].get(lt, np.nan) for lt in loss_types}
            if sum([np.isnan(v) for _, v in value.items()]) > 0:
                continue
            for lt in loss_types:            
                combined_values[lt][item['iteration']] += value[lt]
            count_values[item['iteration']] += 1

    avg_values = {}
    for lt in loss_types:
        avg_values[lt] = [val / count if count > 0 else np.nan 
                        for val, count in zip(combined_values[lt], count_values)]
    
    return avg_values, max_iter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and plot loss data from log file.")
    parser.add_argument("filename", type=str, 
                        default="c:\\Users\\Yudi\\Desktop\\temp\\2023-06-08T18_00_30_basket_01_all_term.log",
                        help="Path to the log file.")
    parser.add_argument("filename2", type=str, nargs='?',
                        default=None, help="Path to the second log file (optional).")
    args = parser.parse_args()

    segments_data = parse_log_file(args.filename)
    individuals_data = group_by_individual(segments_data)
    if args.filename2:
        segments_data2 = parse_log_file(args.filename2)
        individuals_data2 = group_by_individual(segments_data2)
        plot_individual_losses(individuals_data, individuals_data2)
    else:
        plot_individual_losses(individuals_data)
