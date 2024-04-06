# import landt_processing.data_tools as lp
import pandas as pd
import landt_processing.landt_data_tools as lp
import os

# file = 'full_column_1.xlsx'
# path = r'C:\Users\Piotr\Documents\Github\LANDt-processing\tests\echem_xlsx_test'

# test_df = pd.ExcelFile(os.path.join(path, file)).parse(2, header=[0, 1])
# test_df =pd.read_excel(os.path.join(path, file), header=[0, 1], sheet_name=2)
# print(type(test_df))

# test_df.drop(columns=['SysTime.1', 'EnergyD'], axis=1, level=1, inplace=True)
# print(test_df.columns.levels[1])

# test_df = lp.landt_file_loader(os.path.join(path, file))
# print(test_df.columns)
# print(test_df.head(10))
# print(test_df.shape)
# print(test_df.columns[test_df.isna().any()].tolist())



# Sample DataFrame
# data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
# df = pd.DataFrame(data)
# print(type(df))

# # Drop columns
# df.drop(columns=['B', 'C'], inplace=True)  # Use inplace=True to modify the DataFrame in place

# # Now, access df.columns
# print(df.columns)  # This should only print ['A']
