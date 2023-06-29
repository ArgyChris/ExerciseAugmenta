import argparse
import sys
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d


def perform_descriptive_statistics(csv_file):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Calculate descriptive statistics 
    stats_mean_median_min_max = df.describe().loc[['mean', '50%', 'min', 'max']]
    stats_modes = df.mode().dropna()

    # Calculate measures of dispersion per column
    stats_range = df.max() - df.min()
    stats_variance = df.var()
    stats_std = df.std()
    iqr = df.quantile(0.75) - df.quantile(0.25)
    stats_percentiles = df.describe(percentiles=[0.25, 0.5, 0.75]).loc[['25%', '50%', '75%']]
    counts = df.count()

    # Create a dictionary to store the statistics
    statistics_dict = {}

    for column in df.columns:
        statistics_dict[column] = {
            'Count': counts[column],
            'Mean': stats_mean_median_min_max[column]['mean'],
            'Median': stats_mean_median_min_max[column]['50%'],
            'Minimum': stats_mean_median_min_max[column]['min'],
            'Maximum': stats_mean_median_min_max[column]['max'],
            'Range': stats_range[column],
            'Variance': stats_variance[column],
            'Standard Deviation': stats_std[column],
            'IQR': iqr[column],
            '25th Percentile': stats_percentiles[column]['25%'],
            '50th Percentile (Median)': stats_percentiles[column]['50%'],
            '75th Percentile': stats_percentiles[column]['75%'],
        }
        
        if column in stats_modes.columns:
            statistics_dict[column]['Mode'] = stats_modes[column].values.tolist()

    return statistics_dict

def save_statistics_to_csv(statistics_dict, output_csv):
    df = pd.DataFrame.from_dict(statistics_dict, orient='index')
    df = df.transpose()
    df.to_csv(output_csv)

def visualize_data(csv_file, save_file1, save_file2):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    plt.figure(figsize=(10, 6))

    columns_to_plot = ['pixel_value', 'exposure']
    smooth_column1 = gaussian_filter1d(df['pixel_value'], sigma=20)
    smooth_column2 = gaussian_filter1d(df['exposure'], sigma=20)
    
    sns.lineplot(data=df, x=df.index, y=smooth_column1, label='pixel_value', color='blue')
    sns.lineplot(data=df, x=df.index, y=smooth_column2, label='exposure', color='orange')
    plt.xlabel('Measurement')
    plt.ylabel('Image Intensity')
    plt.title('Pixel values vs. Exposure')
    plt.savefig(save_file1, format='png')
    plt.show()

    # plot each column
    values = df.values
    groups = [0, 1, 2]
    i = 1

    plt.figure(figsize = (25, 15))
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(values[:, group])  
        plt.title(df.columns[group], y=0.5, loc='right')
        i += 1

    plt.savefig(save_file2, format='png')
    plt.show()

    
# Main program
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data analysis')
    parser.add_argument('--input_csv', type=str, help='Input CSV file')
    parser.add_argument('--output_csv', type=str, help='Output CSV file with statistics')
    parser.add_argument('--output_plot', type=str, help='Ouput data visualization plot as png')
    parser.add_argument('--output_plot2', type=str, help='Ouput second data visualization plot as png')

    # Parse the arguments
    args = parser.parse_args()

    # Perform descriptive statistics
    result = perform_descriptive_statistics(args.input_csv)

    # Save the result to a text file
    save_statistics_to_csv(result, args.output_csv)

    visualize_data(args.input_csv, args.output_plot, args.output_plot2)

    print("Descriptive statistics generated, saved successfully, and plot generation!")
