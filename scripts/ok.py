#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
import glob

def find_controls_csv_files(base_input_dir):
    """
    Recursively finds all 'controls.csv' files within the base_input_dir.
    It expects subdirectories (like those created by bag_to_csv.py)
    each containing a controls.csv.
    """
    csv_files = []
    # Search for 'controls.csv' in any subdirectory of base_input_dir
    # The pattern '*/controls.csv' means look in any folder (*) directly under base_input_dir
    # If controls.csv can be nested deeper, use '**/*controls.csv' (requires recursive=True in glob)
    # For now, assuming 'base_input_dir/dataset_name_XXX/controls.csv' structure
    
    # More robust search using os.walk for arbitrary depth
    for root, dirs, files in os.walk(base_input_dir):
        for file in files:
            if file == "controls.csv":
                csv_files.append(os.path.join(root, file))
                
    return csv_files

def load_and_concatenate_data(csv_file_paths):
    """
    Loads data from a list of CSV files and concatenates them into a single DataFrame.
    """
    all_dataframes = []
    for csv_path in csv_file_paths:
        try:
            df = pd.read_csv(csv_path)
            # Basic check for essential columns before appending
            required_cols = ['vx', 'vy', 'vz'] # Expert velocity columns
            if all(col in df.columns for col in required_cols):
                all_dataframes.append(df)
                print(f"  Successfully loaded and added: {csv_path} (rows: {len(df)})")
            else:
                print(f"  Warning: Skipping {csv_path}. Missing one or more required columns ({', '.join(required_cols)}). Found: {df.columns.tolist()}")
        except Exception as e:
            print(f"  Warning: Could not read or process {csv_path}: {e}")
    
    if not all_dataframes:
        return None
        
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df

def generate_comprehensive_velocity_report(base_input_dir, output_dir):
    """
    Generates histograms and statistics for expert velocities from ALL controls.csv 
    files found under base_input_dir.

    Args:
        base_input_dir (str): The base directory containing dataset subfolders (each with a controls.csv).
        output_dir (str): Directory to save the plots and statistics file for the combined dataset.
    """
    print(f"Starting comprehensive analysis of expert velocities from: {base_input_dir}")
    
    if not os.path.isdir(base_input_dir):
        print(f"Error: Base input directory not found at {base_input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Find all controls.csv files
    csv_files = find_controls_csv_files(base_input_dir)
    if not csv_files:
        print(f"No 'controls.csv' files found under {base_input_dir}.")
        return
    
    print(f"\nFound {len(csv_files)} 'controls.csv' files to process:")
    for f_path in csv_files:
        print(f"  - {f_path}")

    # 2. Load and concatenate data
    print("\nLoading and concatenating data...")
    combined_df = load_and_concatenate_data(csv_files)

    if combined_df is None or combined_df.empty:
        print("\nNo data to analyze after attempting to load CSVs. Exiting.")
        return
        
    print(f"\nTotal rows in combined dataset: {len(combined_df)}")

    # Define the expert velocity columns 
    expert_vel_cols = ['vx', 'vy', 'vz'] 
    
    # Verify columns exist in the combined DataFrame
    missing_cols = [col for col in expert_vel_cols if col not in combined_df.columns]
    if missing_cols:
        print(f"\nError: The following expert velocity columns are missing in the combined CSV data: {', '.join(missing_cols)}")
        print(f"Available columns in combined data: {combined_df.columns.tolist()}")
        return

    expert_df = combined_df[expert_vel_cols].copy()

    # --- 3. Calculate Descriptive Statistics for the Combined Dataset ---
    print("\n--- Descriptive Statistics for Combined Expert Velocities ---")
    stats = expert_df.describe()
    print(stats)

    stats_filename = os.path.join(output_dir, "combined_expert_velocity_statistics.txt")
    try:
        with open(stats_filename, 'w') as f:
            f.write(f"Descriptive Statistics for Combined Expert Velocities (vx, vy, vz)\n")
            f.write(f"Source Directory: {base_input_dir}\n")
            f.write(f"Number of CSV files processed: {len(csv_files)}\n")
            f.write(f"Total data points (rows): {len(expert_df)}\n")
            f.write("=" * 70 + "\n")
            f.write(stats.to_string())
            f.write("\n\nProcessed CSV files:\n")
            for f_path in csv_files:
                f.write(f"  - {f_path}\n")
        print(f"\nCombined statistics saved to: {stats_filename}")
    except Exception as e:
        print(f"Error saving combined statistics file: {e}")

    # --- 4. Generate Histograms for the Combined Dataset ---
    print("\n--- Generating Combined Histograms ---")
    sns.set_theme(style="whitegrid")

    for col_idx, col in enumerate(expert_vel_cols):
        plt.figure(figsize=(12, 7)) # Slightly larger figure for combined data
        
        sns.histplot(expert_df[col], kde=True, bins=75, color=sns.color_palette("viridis", 3)[col_idx]) # More bins for potentially larger dataset
        
        plt.title(f'Distribution of Combined Expert Command: {col}', fontsize=16)
        plt.xlabel(f'{col} (m/s)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        mean_val = expert_df[col].mean()
        median_val = expert_df[col].median()
        std_val = expert_df[col].std()
        
        plt.axvline(mean_val, color='r', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.3f}')
        plt.axvline(median_val, color='g', linestyle='dashed', linewidth=1.5, label=f'Median: {median_val:.3f}')
        
        # Annotate statistics on the plot
        stats_text = f"Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nStd Dev: {std_val:.3f}\nCount: {len(expert_df[col])}"
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plt.legend()
        
        histogram_filename = os.path.join(output_dir, f"combined_expert_{col}_histogram.png")
        try:
            plt.savefig(histogram_filename, bbox_inches='tight') # bbox_inches='tight' helps ensure labels are not cut off
            print(f"Combined histogram for {col} saved to: {histogram_filename}")
        except Exception as e:
            print(f"Error saving combined histogram for {col}: {e}")
        plt.close() 

    print("\nComprehensive analysis complete.")
    print(f"All reports and plots saved in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze and combine expert velocities from all controls.csv files in a given base dataset directory.'
    )
    parser.add_argument(
        '--input_dataset_dir', 
        required=True, 
        help='Base directory containing subfolders, each with a controls.csv file (e.g., the output_dir from bag_to_csv.py runs).'
    )
    parser.add_argument(
        '--output_dir', 
        required=True, 
        help='Directory to save the generated combined statistics and plots.'
    )
    
    args = parser.parse_args()
    
    generate_comprehensive_velocity_report(args.input_dataset_dir, args.output_dir)

if __name__ == '__main__':
    main()