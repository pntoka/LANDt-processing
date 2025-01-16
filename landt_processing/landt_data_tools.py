import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.interpolate import splev, splrep
from scipy.signal import savgol_filter, argrelmax, argrelmin
import matplotlib.pyplot as plt


def landt_file_loader(filepath, process=True):
    extension = os.path.splitext(filepath)[-1].lower()
    # add functionality to read csv as it will be quicker
    if extension == ".xlsx":
        xlsx = pd.ExcelFile(os.path.join(filepath), engine="openpyxl")
        df = xlsx_process(xlsx)
    elif extension == ".xls":
        xlsx = pd.ExcelFile(os.path.join(filepath))
        df = xlsx_process(xlsx)
    elif extension == ".csv":
        df = pd.read_csv(os.path.join(filepath))
        if df.columns[0] != 'Record':
            raise ValueError('CSV file in wrong format')
    if extension in ['.xlsx', '.csv', 'xls']:
        df = process_dataframe(df) if process else df
        return df
    elif extension not in ['.xlsx', '.csv', 'xls']:
        raise ValueError('File extension not supported')


def xlsx_process(xlsx):
    sheet_names = xlsx.sheet_names
    if len(sheet_names) == 1:  # if only one sheet, use that sheet
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
            raise ValueError("No sheet with record tab found in file")
    return df


def find_record_tab(sheet_names):
    # Finds the index of the sheet with the record tab
    for i in range(len(sheet_names)):
        if "record" in sheet_names[i].lower():
            return i


def check_cycle_split(df):
    # Checks if the data is split into multiple columns
    if "Cycle" in df.columns[0]:
        return True
    else:
        return False


def multi_column_handler(xlsx, record_tab):
    # Handles the multi-column data
    df = xlsx.parse(record_tab, header=[0, 1])
    df.drop(
        ["EnergyD"], axis=1, level=1, inplace=True
    )  # Drop column that only appears in last cycle
    if len(df.columns.levels) > 1:
        frames = []
        for i in df.columns.get_level_values(0).unique():  # Splits df by cycle
            # The following 2 lines remove empty rows and rows where only the 'Record' column is filled
            new_df = df[i].dropna(how="all")
            new_df = new_df[
                ~(new_df["Record"] != np.nan)
                | ~(new_df.drop("Record", axis=1).isna().all(axis=1))
            ]
            frames.append(new_df)
        new_df = pd.concat(frames, axis=0)
    return new_df


def process_dataframe(df):
    # Process the DataFrame
    # Elements taken from old_land_processing function from BenSmithGreyGroup navani

    df = df[df["Current/mA"].apply(type) != str]
    df = df[pd.notna(df["Current/mA"])]

    def land_state(x):
        # 1 is positive current and 0 is negative current
        if x > 0:
            return 1
        elif x < 0:
            return 0
        elif x == 0:
            return "R"
        else:
            print(x)
            raise ValueError("Unexpected value in current - not a number")

    df["state"] = df["Current/mA"].map(lambda x: land_state(x))
    not_rest_idx = df[df["state"] != "R"].index
    df.loc[not_rest_idx, "cycle change"] = df.loc[not_rest_idx, "state"].ne(
        df.loc[not_rest_idx, "state"].shift()
    )
    df["half cycle"] = (df["cycle change"] == True).cumsum()
    df["full cycle"] = (df["half cycle"] / 2).apply(np.ceil)

    columns_to_keep = [
        "Current/mA",
        "Capacity/mAh",
        "state",
        "SpeCap/mAh/g",
        "Voltage/V",
        "dQ/dV/mAh/V",
    ]

    new_df = df.copy()
    new_df = new_df[columns_to_keep]
    new_df["CycleNo"] = df["full cycle"]

    return new_df


def invert_charge_discharge(df):
    # Inverts charge and discharge cycles 0 becomes positive current and 1 becomes negative current
    df.loc[df["state"] == 0, "state"] = 2
    df.loc[df["state"] == 1, "state"] = 0
    df.loc[df["state"] == 2, "state"] = 1
    return df


def create_cycle_summary(df, invert=False):
    # Creates a summary of cycling information from df from one file

    def calculate_electrode_mass(df):
        # Calculate the electrode mass using capacity and specific capacity
        cap = df.loc[df["state"] == 0].groupby("CycleNo")["Capacity/mAh"].max().values
        spe_cap = (
            df.loc[df["state"] == 0].groupby("CycleNo")["SpeCap/mAh/g"].max().values
        )
        mass = cap / spe_cap
        return round(mass.mean() * 1000, 3)

    if invert:
        df = invert_charge_discharge(df)

    summary_df = df.groupby("CycleNo")["Current/mA"].max().to_frame()
    summary_df.attrs["Electrode mass/mg"] = calculate_electrode_mass(df)
    summary_df["Current rate/mA/g"] = round(
        summary_df["Current/mA"] / summary_df.attrs["Electrode mass/mg"] * 1000, 0
    )
    # summary_df['Discharge Cap/mAh'] = df.loc[df['state'] == 0].groupby('CycleNo')['Capacity/mAh'].max()
    # summary_df['Charge Cap/mAh'] = df.loc[df['state'] == 1].groupby('CycleNo')['Capacity/mAh'].max()
    summary_df["Discharge SpeCap/mAh/g"] = (
        df.loc[df["state"] == 0].groupby("CycleNo")["SpeCap/mAh/g"].max()
    )
    summary_df["Charge SpeCap/mAh/g"] = (
        df.loc[df["state"] == 1].groupby("CycleNo")["SpeCap/mAh/g"].max()
    )
    summary_df["CE"] = round(
        summary_df["Charge SpeCap/mAh/g"] / summary_df["Discharge SpeCap/mAh/g"], 4
    )
    summary_df["Electrode mass/mg"] = summary_df.attrs["Electrode mass/mg"]

    return summary_df


def create_summary_from_file(filepath, save_dir=None, invert=False):
    # Creates a summary of cycling information from multiple files
    file_name = Path(filepath).stem
    file_folder = str(Path(filepath).parent)
    df = landt_file_loader(filepath)
    summary_df = create_cycle_summary(df, invert=invert)
    if save_dir is not None:
        summary_df.to_csv(os.path.join(save_dir, file_name + "_summary.csv"))
    else:
        os.makedirs(os.path.join(file_folder, "summary"), exist_ok=True)
        summary_df.to_csv(
            os.path.join(file_folder, "summary", file_name + "_summary.csv")
        )


def create_summary_from_folder(dir_path, save_dir=None):
    # Creates a summary of cycling information from multiple files
    file_list = os.listdir(dir_path)
    for file in tqdm(file_list):
        if file.endswith(".xlsx") or file.endswith(".xls") or file.endswith(".csv"):
            try:
                create_summary_from_file(os.path.join(dir_path, file), save_dir)
            except:
                print(f"Error processing file: {file}")
                continue


def clean_signal(
    voltage,
    capacity,
    dqdv,
    polynomial_spline=3,
    s_spline=1e-5,
    polyorder_1=5,
    window_size_1=101,
    polyorder_2=5,
    window_size_2=1001,
):
    # Function that cleans the raw voltage, cap and dqdv data so can get smooth curves and derivatives

    df = pd.DataFrame({"voltage": voltage, "capacity": capacity, "dqdv": dqdv})
    unique_v = (
        df.astype(float).groupby("voltage").mean().index
    )  # get unique voltage values
    unique_v_cap = df.astype(float).groupby("voltage").mean()["capacity"]
    unique_v_dqdv = df.astype(float).groupby("voltage").mean()["dqdv"]

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
    peak_val = max(smooth_dqdv.min(), smooth_dqdv.max(), key=abs)
    peak_idx = np.where(smooth_dqdv == peak_val)[0]
    return (
        x_volt,
        smooth_cap,
        smooth_dqdv_2,
        peak_idx,
    )  # need to return peak index to ignore very low volt data


def check_state(dqdv):
    # Check if dqdv from discharge or charge (negative or positive peak)
    peak_val = max(dqdv.min(), dqdv.max(), key=abs)
    if peak_val > 0:
        return 1
    elif peak_val < 0:
        return 0
    else:
        return "R"


def find_plat_cap_conv(voltage, capacity, dqdv, cutoff=0.1):
    state = check_state(dqdv)
    if state == 1:
        volt_idx = np.argmin(np.abs(voltage - cutoff))
        plat_cap = capacity[volt_idx]
    elif state == 0:
        volt_idx = np.argmin(np.abs(voltage - cutoff))
        plat_cap = capacity.max() - capacity[volt_idx]
    else:
        plat_cap = np.nan
    return plat_cap


def find_plat_cap(voltage, capacity, dqdv):
    # Finds the plateau capacity, takes min of 2nd derivative for charge and point in between max/min inflection points for discharge
    # This methodd gives better visual results for discharge, uses argrelmin and argrelmax to find inflection points
    _, smooth_cap, smooth_dqdv_2, peak_idx = clean_signal(voltage, capacity, dqdv)
    state = check_state(dqdv)
    if state == 1:
        plat_cap = smooth_cap[smooth_dqdv_2[peak_idx[0]:].argmin() + peak_idx[0]]
    elif state == 0:
        min_peak = argrelmin(smooth_dqdv_2[peak_idx[0]:], order=10)[0].min() + peak_idx[0]
        max_peak = argrelmax(smooth_dqdv_2[peak_idx[0]:], order=10)[0].min() + peak_idx[0]
        plat_point = round((min_peak + max_peak) / 2, 0).astype(int)
        plat_cap = smooth_cap.max() - smooth_cap[plat_point]
    else:
        plat_cap = np.nan
    return plat_cap


def find_plat_cap_2(voltage, capacity, dqdv):
    # Alternative vesion of finding the plateau capacity, takes point in between max/min inflection points for charge and discharge
    _, smooth_cap, smooth_dqdv_2, peak_idx = clean_signal(voltage, capacity, dqdv)
    state = check_state(dqdv)
    if state == 1:
        min_peak = smooth_dqdv_2[peak_idx[0] :].argmin() + peak_idx[0]
        max_peak = smooth_dqdv_2[peak_idx[0] :].argmax() + peak_idx[0]
        plat_point = round((min_peak + max_peak) / 2, 0).astype(int)
        plat_cap = smooth_cap[plat_point]
    elif state == 0:
        min_peak = argrelmin(smooth_dqdv_2[peak_idx[0]:], order=10)[0].min() + peak_idx[0]
        max_peak = argrelmax(smooth_dqdv_2[peak_idx[0]:], order=10)[0].min() + peak_idx[0]
        plat_point = round((min_peak + max_peak) / 2, 0).astype(int)
        plat_cap = smooth_cap.max() - smooth_cap[plat_point]
    else:
        plat_cap = np.nan
    return plat_cap


def find_plat_cap_3(voltage, capacity, dqdv):
    # First iteration of finding the plateau capacity, takes min of 2nd derivative for charge and max for discharge

    _, smooth_cap, smooth_dqdv_2, peak_idx = clean_signal(voltage, capacity, dqdv)
    state = check_state(dqdv)
    if state == 1:
        plat_cap = smooth_cap[smooth_dqdv_2[peak_idx[0] :].argmin() + peak_idx[0]]
    elif state == 0:
        max_peak = argrelmax(smooth_dqdv_2[peak_idx[0]:], order=10)[0].min() + peak_idx[0]
        plat_cap = smooth_cap.max() - smooth_cap[max_peak]
    else:
        plat_cap = np.nan
    return plat_cap


# def find_plat_cap_3(voltage, capacity, dqdv):
#     # Third iteration of finding the plateau capacity, takes point in between max/min inflection points for charge and discharge
#     _, smooth_cap, smooth_dqdv_2, peak_idx = clean_signal(voltage, capacity, dqdv)
#     state = check_state(dqdv)
#     if state == 1:
#         min_peak = smooth_dqdv_2[peak_idx[0] :].argmin() + peak_idx[0]
#         max_peak = smooth_dqdv_2[peak_idx[0] :].argmax() + peak_idx[0]
#         plat_point = round((min_peak + max_peak) / 2, 0).astype(int)
#         plat_cap = smooth_cap[plat_point]
#     elif state == 0:
#         min_peak = smooth_dqdv_2[peak_idx[0] :].argmin() + peak_idx[0]
#         max_peak = smooth_dqdv_2[peak_idx[0] :].argmax() + peak_idx[0]
#         plat_point = round((min_peak + max_peak) / 2, 0).astype(int)
#         plat_cap = smooth_cap.max() - smooth_cap[plat_point]
#     else:
#         plat_cap = np.nan
#     return plat_cap


# def find_plat_cap_4(voltage, capacity, dqdv):
#     # Fourth iteration of finding the plateau capacity, takes point in between max/min inflection points for charge and discharge
#     _, smooth_cap, smooth_dqdv_2, peak_idx = clean_signal(voltage, capacity, dqdv)
#     state = check_state(dqdv)
#     if state == 1:
#         min_peak = smooth_dqdv_2[peak_idx[0] :peak_idx[0]+800].argmin() + peak_idx[0]
#         max_peak = smooth_dqdv_2[peak_idx[0] :peak_idx[0]+800].argmax() + peak_idx[0]
#         plat_point = round((min_peak + max_peak) / 2, 0).astype(int)
#         plat_cap = smooth_cap[plat_point]
#     elif state == 0:
#         min_peak = smooth_dqdv_2[peak_idx[0] :peak_idx[0]+800].argmin() + peak_idx[0]
#         max_peak = smooth_dqdv_2[peak_idx[0] :peak_idx[0]+800].argmax() + peak_idx[0]
#         plat_point = round((min_peak + max_peak) / 2, 0).astype(int)
#         plat_cap = smooth_cap.max() - smooth_cap[plat_point]
#     else:
#         plat_cap = np.nan
#     return plat_cap


def plot_plateau(ax, voltage, capacity, dqdv, line=False, input_plat=None, method=1):
    x_volt, smooth_cap, _, _ = clean_signal(voltage, capacity, dqdv)
    if input_plat is None and method == 1:
        plat_cap = find_plat_cap(voltage, capacity, dqdv)
    elif input_plat is None and method == 2:
        plat_cap = find_plat_cap_2(voltage, capacity, dqdv)
    elif input_plat is None and method == 3:
        plat_cap = find_plat_cap_3(voltage, capacity, dqdv)
    elif input_plat is None and method == 'conv':
        plat_cap = find_plat_cap_conv(voltage, capacity, dqdv)
    # elif input_plat is None and method == 4:
    #     plat_cap = find_plat_cap_4(voltage, capacity, dqdv)
    else:
        plat_cap = input_plat
    index_plat = np.argmin(np.abs(smooth_cap - plat_cap))
    index_slop = np.argmin(np.abs(smooth_cap - (smooth_cap.max() - plat_cap)))
    state = check_state(dqdv)
    if plat_cap is not np.nan:
        ax.plot(smooth_cap, x_volt)
        if state == 0:
            ax.scatter(
                smooth_cap[index_slop], x_volt[index_slop], c="r", marker="x", s=100
            )
        elif state == 1:
            ax.scatter(
                smooth_cap[index_plat], x_volt[index_plat], c="r", marker="x", s=100
            )
        ax.set_xlabel("Capacity/mAh/g")
        ax.set_ylabel("Voltage/V")
        ax.text(
            90,
            1.7,
            f"Plateau Capacity: {round(plat_cap, 2)} mAh/g \nSloping capacity: {round(smooth_cap.max()-plat_cap, 2)} mAh/g",
            fontsize=14,
        )
        if line:
            if state == 0:
                ax.axhline(x_volt[index_slop], color="r", linestyle="--")
            elif state == 1:
                ax.axhline(x_volt[index_plat], color="r", linestyle="--")


def get_plat_from_file(filepath, cycle_no=1, plot=False, display_plot=False, save_dir=None):
    df = landt_file_loader(filepath)
    volt_0 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 0)]["Voltage/V"].values
    volt_1 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 1)]["Voltage/V"].values
    cap_0 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 0)]["SpeCap/mAh/g"].values
    cap_1 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 1)]["SpeCap/mAh/g"].values
    dqdv_0 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 0)]["dQ/dV/mAh/V"].values
    dqdv_1 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 1)]["dQ/dV/mAh/V"].values

    plat_cap_0 = find_plat_cap(volt_0, cap_0, dqdv_0)
    plat_cap_1 = find_plat_cap(volt_1, cap_1, dqdv_1)
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(16, 10))
        ax[0].set_title("Discharge", fontsize=16)
        plot_plateau(ax[0], volt_0, cap_0, dqdv_0, line=True)
        ax[1].set_title("Charge", fontsize=16)
        plot_plateau(ax[1], volt_1, cap_1, dqdv_1, line=True)
        fig.suptitle(f"{Path(filepath).stem}", fontsize=16)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, Path(filepath).stem + "_plateau.png"))
        if display_plot:
            plt.show()
        return plat_cap_0, plat_cap_1, fig
    return plat_cap_0, plat_cap_1


def get_plat_from_file_2(filepath, cycle_no=1, plot=False, display_plot=False, save_dir=None):
    df = landt_file_loader(filepath)
    volt_0 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 0)]["Voltage/V"].values
    volt_1 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 1)]["Voltage/V"].values
    cap_0 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 0)]["SpeCap/mAh/g"].values
    cap_1 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 1)]["SpeCap/mAh/g"].values
    dqdv_0 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 0)]["dQ/dV/mAh/V"].values
    dqdv_1 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 1)]["dQ/dV/mAh/V"].values

    plat_cap_0 = find_plat_cap_2(volt_0, cap_0, dqdv_0)
    plat_cap_1 = find_plat_cap_2(volt_1, cap_1, dqdv_1)
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(16, 10))
        ax[0].set_title("Discharge", fontsize=16)
        plot_plateau(ax[0], volt_0, cap_0, dqdv_0, line=True, method=2)
        ax[1].set_title("Charge", fontsize=16)
        plot_plateau(ax[1], volt_1, cap_1, dqdv_1, line=True, method=2)
        fig.suptitle(f"{Path(filepath).stem}", fontsize=16)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, Path(filepath).stem + "_plateau.png"))
        if display_plot:
            plt.show()
        return plat_cap_0, plat_cap_1, fig
    return plat_cap_0, plat_cap_1


def get_plat_from_file_3(filepath, cycle_no=1, plot=False, display_plot=False, save_dir=None):
    df = landt_file_loader(filepath)
    volt_0 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 0)]["Voltage/V"].values
    volt_1 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 1)]["Voltage/V"].values
    cap_0 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 0)]["SpeCap/mAh/g"].values
    cap_1 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 1)]["SpeCap/mAh/g"].values
    dqdv_0 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 0)]["dQ/dV/mAh/V"].values
    dqdv_1 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 1)]["dQ/dV/mAh/V"].values

    plat_cap_0 = find_plat_cap_3(volt_0, cap_0, dqdv_0)
    plat_cap_1 = find_plat_cap_3(volt_1, cap_1, dqdv_1)
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(16, 10))
        ax[0].set_title("Discharge", fontsize=16)
        plot_plateau(ax[0], volt_0, cap_0, dqdv_0, line=True, method=3)
        ax[1].set_title("Charge", fontsize=16)
        plot_plateau(ax[1], volt_1, cap_1, dqdv_1, line=True, method=3)
        fig.suptitle(f"{Path(filepath).stem}", fontsize=16)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, Path(filepath).stem + "_plateau.png"))
        if display_plot:
            plt.show()
        return plat_cap_0, plat_cap_1, fig
    return plat_cap_0, plat_cap_1


def get_plat_from_file_conv(filepath, cycle_no=1, plot=False, display_plot=False, save_dir=None):
    df = landt_file_loader(filepath)
    volt_0 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 0)]["Voltage/V"].values
    volt_1 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 1)]["Voltage/V"].values
    cap_0 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 0)]["SpeCap/mAh/g"].values
    cap_1 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 1)]["SpeCap/mAh/g"].values
    dqdv_0 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 0)]["dQ/dV/mAh/V"].values
    dqdv_1 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 1)]["dQ/dV/mAh/V"].values

    plat_cap_0 = find_plat_cap_conv(volt_0, cap_0, dqdv_0)
    plat_cap_1 = find_plat_cap_conv(volt_1, cap_1, dqdv_1)
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(16, 10))
        ax[0].set_title("Discharge", fontsize=16)
        plot_plateau(ax[0], volt_0, cap_0, dqdv_0, line=True, method='conv')
        ax[1].set_title("Charge", fontsize=16)
        plot_plateau(ax[1], volt_1, cap_1, dqdv_1, line=True, method='conv')
        fig.suptitle(f"{Path(filepath).stem}", fontsize=16)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, Path(filepath).stem + "_plateau.png"))
        if display_plot:
            plt.show()
        return plat_cap_0, plat_cap_1, fig
    return plat_cap_0, plat_cap_1


# def get_plat_from_file_4(filepath, plot=False, display_plot=False, save_dir=None):
#     df = landt_file_loader(filepath)
#     volt_0 = df.loc[(df["CycleNo"] == 1) & (df["state"] == 0)]["Voltage/V"].values
#     volt_1 = df.loc[(df["CycleNo"] == 1) & (df["state"] == 1)]["Voltage/V"].values
#     cap_0 = df.loc[(df["CycleNo"] == 1) & (df["state"] == 0)]["SpeCap/mAh/g"].values
#     cap_1 = df.loc[(df["CycleNo"] == 1) & (df["state"] == 1)]["SpeCap/mAh/g"].values
#     dqdv_0 = df.loc[(df["CycleNo"] == 1) & (df["state"] == 0)]["dQ/dV/mAh/V"].values
#     dqdv_1 = df.loc[(df["CycleNo"] == 1) & (df["state"] == 1)]["dQ/dV/mAh/V"].values

#     plat_cap_0 = find_plat_cap_4(volt_0, cap_0, dqdv_0)
#     plat_cap_1 = find_plat_cap_4(volt_1, cap_1, dqdv_1)
#     if plot:
#         fig, ax = plt.subplots(1, 2, figsize=(16, 10))
#         ax[0].set_title("Discharge", fontsize=16)
#         plot_plateau(ax[0], volt_0, cap_0, dqdv_0, line=True, method=4)
#         ax[1].set_title("Charge", fontsize=16)
#         plot_plateau(ax[1], volt_1, cap_1, dqdv_1, line=True, method=4)
#         fig.suptitle(f"{Path(filepath).stem}", fontsize=16)
#         plt.tight_layout()
#         if save_dir:
#             plt.savefig(os.path.join(save_dir, Path(filepath).stem + "_plateau.png"))
#         if display_plot:
#             plt.show()
#     return plat_cap_0, plat_cap_1



def get_charge_ice_mass(file_name_cex, summary_folder_path):
    summary_file_csv = file_name_cex.replace(".cex", "_summary.csv")
    summary_file_path = os.path.join(summary_folder_path, summary_file_csv)
    summary_df = pd.read_csv(summary_file_path)
    charge = summary_df.loc[summary_df["CycleNo"] == 1, "Charge SpeCap/mAh/g"].values[0]
    ice = summary_df.loc[summary_df["CycleNo"] == 1, "CE"].values[0]
    mass = summary_df.loc[summary_df["CycleNo"] == 1, "Electrode mass/mg"].values[0]
    return charge, ice, mass


class PointPicker:
    def __init__(self, ax):
        self.ax = ax
        self.picked_point = None
        self.cid = ax.figure.canvas.mpl_connect("pick_event", self.on_pick)

    def on_pick(self, event):
        ind = event.ind[0]
        x_picked = event.artist.get_xdata()[ind]
        y_picked = event.artist.get_ydata()[ind]
        print(f"Picked point: x={x_picked}, y={y_picked}")
        self.picked_point = {"x": x_picked, "y": y_picked}

    def disconnect(self):
        self.ax.figure.canvas.mpl_disconnect(self.cid)


def extract_echem_features(filepath, cycle_no=1, method=1, invert=False):
    df = landt_file_loader(filepath)
    if invert:
        df = invert_charge_discharge(df)
    volt_0 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 0)][
        "Voltage/V"
    ].values
    cap_0 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 0)][
        "SpeCap/mAh/g"
    ].values
    dqdv_0 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 0)][
        "dQ/dV/mAh/V"
    ].values
    volt_1 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 1)][
        "Voltage/V"
    ].values
    cap_1 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 1)][
        "SpeCap/mAh/g"
    ].values
    dqdv_1 = df.loc[(df["CycleNo"] == cycle_no) & (df["state"] == 1)][
        "dQ/dV/mAh/V"
    ].values
    if method == 1:
        plat_cap_0 = find_plat_cap(volt_0, cap_0, dqdv_0)
        plat_cap_1 = find_plat_cap(volt_1, cap_1, dqdv_1)
    elif method == 2:
        plat_cap_0 = find_plat_cap_2(volt_0, cap_0, dqdv_0)
        plat_cap_1 = find_plat_cap_2(volt_1, cap_1, dqdv_1)
    elif method == 3:
        plat_cap_0 = find_plat_cap_3(volt_0, cap_0, dqdv_0)
        plat_cap_1 = find_plat_cap_3(volt_1, cap_1, dqdv_1)
    elif method == 'conv':
        plat_cap_0 = find_plat_cap_conv(volt_0, cap_0, dqdv_0)
        plat_cap_1 = find_plat_cap_conv(volt_1, cap_1, dqdv_1)

    ice = {"Parameter": "ICE", "Value": round(cap_1.max() / cap_0.max(), 4)}
    charge_cap = {"Parameter": "Charge SpeCap/mAh/g", "Value": round(cap_1.max(), 2)}
    discharge_plat_cap = {
        "Parameter": "Discharge plateau SpeCap/mAh/g",
        "Value": round(plat_cap_0, 2),
    }
    charge_plat_cap = {
        "Parameter": "Charge plateau SpeCap/mAh/g",
        "Value": round(plat_cap_1, 2),
    }

    echem_df = pd.DataFrame([ice, charge_cap, discharge_plat_cap, charge_plat_cap])
    return echem_df


def df_transpose(df):
    df = df.transpose().reset_index()
    df.drop('index', axis=1, inplace=True)
    df.columns = df.iloc[0]
    df = df[1:]
    return df


def plot_rate_cap(echem_summary_df, charge=True, display_plot=False, save_dir=None):
    if charge:
        table_key = "Charge SpeCap/mAh/g"
    else:
        table_key = "Discharge SpeCap/mAh/g"
    echem_summary_df.reset_index(inplace=True)  # this is here because the summary dataframe uses the cycle number as the index
    echem_summary_df = echem_summary_df.loc[echem_summary_df['CycleNo'] != 0]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(echem_summary_df['CycleNo'], echem_summary_df[table_key], s=100)
    vline_pos = list(range(5, int(echem_summary_df['CycleNo'].max()), 5))
    text_pos = list(range(1, int(echem_summary_df['CycleNo'].max()), 5))
    ax.set_ylim(0, echem_summary_df[table_key].max()*1.2)
    ax.set_xlim(0, echem_summary_df['CycleNo'].max()+1)
    for pos in vline_pos:
        ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.7)
    for pos in text_pos:
        ax.text((pos+1), plt.gca().get_ylim()[1] * 0.9, f"{echem_summary_df['Current rate/mA/g'][pos]} mA g$^{{-1}}$",
                horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'), fontsize=8)
    tick_pos = np.arange(0, echem_summary_df['CycleNo'].max()+1, 1)
    ax.set_xticks(tick_pos, minor=True)
    ax.set_xticks(vline_pos)
    ax.set_xticklabels(vline_pos)
    ax.tick_params(axis='x', which='major', length=7)
    ax.tick_params(axis='x', which='minor', length=4)
    ax.set_xlabel('Cycle number', fontsize=16)
    if charge:
        ax.set_ylabel('Charge Specific Capacity/mAh g$^{-1}$', fontsize=16)
    else:
        ax.set_ylabel('Discharge Specific Capacity/mAh g$^{-1}$', fontsize=16)
    if display_plot:
        plt.show()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"rate_cap_{table_key}.png"))
    return fig