import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.interpolate import splev, splrep
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


def landt_file_loader(filepath, process=True):
    extension = os.path.splitext(filepath)[-1].lower()
    # add functionality to read csv as it will be quicker
    if extension == '.xlsx':
        xlsx = pd.ExcelFile(os.path.join(filepath), engine='openpyxl')
        df = xlsx_process(xlsx)
    elif extension == '.xls':
        xlsx = pd.ExcelFile(os.path.join(filepath))
        df = xlsx_process(xlsx)
    elif extension == '.csv':
        df = pd.read_csv(os.path.join(filepath))
        if df.columns[0] != 'Record':
            raise ValueError('CSV file in wrong format')


    # sheet_names = xlsx.sheet_names
    # if len(sheet_names) == 1:       # if only one sheet, use that sheet
    #     df = xlsx.parse(sheet_names[0])
    #     if check_cycle_split(df):
    #         df = multi_column_handler(xlsx, 0)
    # else:
    #     record_tab = find_record_tab(sheet_names)  # find the sheet with the record tab
    #     if record_tab is not None:
    #         df = xlsx.parse(sheet_names[record_tab])
    #         if check_cycle_split(df):
    #             df = multi_column_handler(xlsx, record_tab)
    #     else:
    #         raise ValueError('No sheet with record tab found in file')
    df = process_dataframe(df) if process else df
    return df


def xlsx_process(xlsx):
    sheet_names = xlsx.sheet_names
    if len(sheet_names) == 1:       # if only one sheet, use that sheet
        df = xlsx.parse(sheet_names[0])
        if check_cycle_split(df):
            df = multi_column_handler(xlsx, 0)
    else:
        record_tab = find_record_tab(sheet_names)  # find the sheet with the record tab
        if record_tab is not None:
            df = xlsx.parse(sheet_names[record_tab])
            if check_cycle_split(df):
                df = multi_column_handler(xlsx, record_tab)
        else:
            raise ValueError('No sheet with record tab found in file')
    return df


def find_record_tab(sheet_names):
    # Finds the index of the sheet with the record tab
    for i in range(len(sheet_names)):
        if 'record' in sheet_names[i].lower():
            return i


def check_cycle_split(df):
    # Checks if the data is split into multiple columns
    if 'Cycle' in df.columns[0]:
        return True
    else:
        return False


def multi_column_handler(xlsx, record_tab):
    # Handles the multi-column data
    df = xlsx.parse(record_tab, header=[0, 1])
    df.drop(['EnergyD'], axis=1, level=1, inplace=True)   # Drop column that only appears in last cycle
    if len(df.columns.levels) > 1:
        frames = []
        for i in df.columns.get_level_values(0).unique():   # Splits df by cycle
            # The following 2 lines remove empty rows and rows where only the 'Record' column is filled
            new_df = df[i].dropna(how='all')
            new_df = new_df[~(new_df['Record'] != np.nan) | ~(new_df.drop('Record', axis=1).isna().all(axis=1))]
            frames.append(new_df)
        new_df = pd.concat(frames, axis=0)
    return new_df


def process_dataframe(df):
    # Process the DataFrame
    columns_to_keep = [
        'Current/mA', 'Capacity/mAh',
        'SpeCap/mAh/g', 'Voltage/V', 'dQ/dV/mAh/V',
        'CycleNo', 'StepNo'
    ]
    # Gets the step time column that can be with different units
    step_time = next((item for item in df.columns.tolist() if item.startswith('StepTime')), None)
    columns_to_keep.append(step_time)
    new_df = df.copy()
    new_df = new_df[columns_to_keep]

    def land_state(x):
        if x > 0:
            return 1
        elif x < 0:
            return 0
        elif x == 0:
            return 'R'
        else:
            print(x)
            raise ValueError('Unexpected value in current - not a number')
        
    new_df['state'] = new_df.loc[:, ('Current/mA')].apply(lambda x: land_state(x))

    return new_df


def invert_charge_discharge(df):
    # Inverts charge and discharge cycles zero becomes positive current and 1 becomes negative current
    df.loc[df['state'] == 0, 'state'] = 2
    df.loc[df['state'] == 1, 'state'] = 0
    df.loc[df['state'] == 2, 'state'] = 1
    return df


def create_cycle_summary(df, inverse=False):
    # Creates a summary of cycling information from df from one file

    def calculate_electrode_mass(df):
        # Calculate the electrode mass using capacity and specific capacity
        cap = df.loc[df['state'] == 0].groupby('CycleNo')['Capacity/mAh'].max().values
        spe_cap = df.loc[df['state'] == 0].groupby('CycleNo')['SpeCap/mAh/g'].max().values
        mass = cap / spe_cap
        return round(mass.mean()*1000, 3)
    
    if inverse:
        df = invert_charge_discharge(df)

    summary_df = df.groupby('CycleNo')['Current/mA'].max().to_frame()
    summary_df.attrs['Electrode mass/mg'] = calculate_electrode_mass(df)
    summary_df['Current rate/mA/g'] = round(summary_df['Current/mA'] / summary_df.attrs['Electrode mass/mg']*1000, 0)
    # summary_df['Discharge Cap/mAh'] = df.loc[df['state'] == 0].groupby('CycleNo')['Capacity/mAh'].max()
    # summary_df['Charge Cap/mAh'] = df.loc[df['state'] == 1].groupby('CycleNo')['Capacity/mAh'].max()
    summary_df['Discharge SpeCap/mAh/g'] = df.loc[df['state'] == 0].groupby('CycleNo')['SpeCap/mAh/g'].max()
    summary_df['Charge SpeCap/mAh/g'] = df.loc[df['state'] == 1].groupby('CycleNo')['SpeCap/mAh/g'].max()
    summary_df['CE'] = round(summary_df['Charge SpeCap/mAh/g'] / summary_df['Discharge SpeCap/mAh/g'], 4)
    summary_df['Electrode mass/mg'] = summary_df.attrs['Electrode mass/mg']

    return summary_df


def create_summary_from_file(filepath, save_dir=None):
    # Creates a summary of cycling information from multiple files
    file_name = Path(filepath).stem
    file_folder = str(Path(filepath).parent)
    df = landt_file_loader(filepath)
    summary_df = create_cycle_summary(df)
    if save_dir is not None:
        summary_df.to_csv(os.path.join(save_dir, file_name+'_summary.csv'))
    else:
        summary_df.to_csv(os.path.join(file_folder, file_name+'_summary.csv'))


def create_summary_from_folder(dir_path, save_dir=None):
    # Creates a summary of cycling information from multiple files
    file_list = os.listdir(dir_path)
    for file in tqdm(file_list):
        if file.endswith('.xlsx') or file.endswith('.xls'):
            try:
                create_summary_from_file(os.path.join(dir_path, file), save_dir)
            except:
                print(f'Error processing file: {file}')
                continue


def clean_signal(voltage, capacity, dqdv,
                 polynomial_spline=3, s_spline=1e-5,
                 polyorder_1=5, window_size_1=101,
                 polyorder_2=5, window_size_2=1001):
    # Function that cleans the raw voltage, cap and dqdv data so can get smooth curves and derivatives

    df = pd.DataFrame({'voltage': voltage, 'capacity': capacity, 'dqdv': dqdv}) 
    unique_v = df.astype(float).groupby('voltage').mean().index() # get unique voltage values
    unique_v_cap = df.astype(float).groupby('voltage').mean()['cap']
    unique_v_dqdv = df.astype(float).groupby('voltage').mean()['dqdv']

    x_volt = np.linspace(unique_v.min(), unique_v.max(), num=int(1e4))

    spl_cap = splrep(unique_v, unique_v_cap, k=1, s=1.0)
    cap = splev(x_volt, spl_cap)
    smooth_cap = savgol_filter(cap, window_size_1, polyorder_1)

    spl = splrep(unique_v, unique_v_dqdv, k=1, s=1.0)
    y_dqdq = splev(x_volt, spl)
    smooth_dqdv = savgol_filter(y_dqdq, window_size_1, polyorder_1)
    smooth_spl_dqdv = splrep(x_volt, smooth_dqdv, k=polynomial_spline, s=s_spline)
    dqdv_2 = splev(x_volt, smooth_spl_dqdv, der=1)
    smooth_dqdv_2 = savgol_filter(dqdv_2, window_size_2, polyorder_2)
    return x_volt, smooth_cap, smooth_dqdv_2


def check_state(dqdv):
    # Check if dqdv from discharge or charge (negative or positive peak)
    peak_val = max(dqdv.min(), dqdv.max(), key=abs)
    if peak_val > 0:
        return 1
    elif peak_val < 0:
        return 0
    else:
        return 'R'


def find_plat_cap(voltage, capacity, dqdv):
    x_volt, smooth_cap, smooth_dqdv_2 = clean_signal(voltage, capacity, dqdv)
    state = check_state(dqdv)
    if state == 1:
        plat_cap = smooth_cap[smooth_dqdv_2.argmin()]
    elif state == 0:
        plat_cap = smooth_cap.max() - smooth_cap[smooth_dqdv_2.argmax()]
    else:
        plat_cap = np.nan
    return plat_cap


def plot_plateau(voltage, capacity, dqdv, line=False):
    x_volt, smooth_cap, smooth_dqdv_2 = clean_signal(voltage, capacity, dqdv)
    plat_cap = find_plat_cap(voltage, capacity, dqdv)
    if plat_cap is not np.nan:
        plt.plot(smooth_cap, x_volt)
        plt.scatter(plat_cap, x_volt[smooth_cap.index(plat_cap)], c='r', marker='x')
        if line:
            plt.axhline(x_volt[smooth_cap.index(plat_cap)], color='r', linestyle='--')
        plt.show()