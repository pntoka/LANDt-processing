import landt_processing.landt_data_tools as lp
import os
import pandas as pd
import pytest
import coverage

p = os.path.dirname(__file__)


file_names = [
    'blank.xlsx',
    'full_column_1.xlsx',
    'full_column_2.xlsx',
    'full_column_2_sec.xlsx',
    'full_no_column_1_sec.xlsx',
    'part_no_column_1_sec.xlsx',
    'part_column_2.xlsx',
]

blank = pd.ExcelFile(os.path.join(p, 'echem_xlsx_test', file_names[0]))
full_column_1 = pd.ExcelFile(os.path.join(p, 'echem_xlsx_test', file_names[1]))
full_column_2 = pd.ExcelFile(os.path.join(p, 'echem_xlsx_test', file_names[2]))


@pytest.mark.parametrize('dataframe, expected', [
    (blank, None),
    (full_column_1, 2),
    (full_column_2, 2)
])
def test_find_record_tab(dataframe, expected):
    assert lp.find_record_tab(dataframe.sheet_names) == expected


full_column_2 = full_column_2.parse(2)
part_no_column_1_sec = pd.ExcelFile(os.path.join(p, 'echem_xlsx_test', file_names[5])).parse()


@pytest.mark.parametrize('dataframe, expected', [
    (full_column_2, True),
    (part_no_column_1_sec, False)
])
def test_check_cycle_split(dataframe, expected):
    assert lp.check_cycle_split(dataframe) == expected


def test_landt_file_loader_raise_error():
    with pytest.raises(ValueError):
        lp.landt_file_loader(os.path.join(p, 'echem_xlsx_test', file_names[0]))


full_column_1 = os.path.join(p, 'echem_xlsx_test', file_names[1])
part_no_column_1_sec = os.path.join(p, 'echem_xlsx_test', file_names[5])


@pytest.mark.parametrize('filepath, expected', [
    (full_column_1, 22),
    (part_no_column_1_sec, 23)
])
def test_landt_file_loader(filepath, expected):
    assert lp.landt_file_loader(filepath) is not None
    assert len(lp.landt_file_loader(filepath).columns) == expected