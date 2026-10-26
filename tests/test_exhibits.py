"""Tests for exhibit-list reading (SPEC 4.2)."""

import pandas as pd

from ffs.exhibits import read_exhibit_file

ROWS = [["Marker", "Description", "Qty"],
        [1, "5 white A4 sheets", 5],
        [2, "half-empty bottle of vodka", 1],
        [5, "knife with black handle", 1],
        [7, "seated body in purple and pink tunic, hands tied", 1]]


def test_reads_xlsx_with_header_and_quantity_column(tmp_path):
    path = tmp_path / "th_0001_IPC_ENG.xlsx"
    pd.DataFrame(ROWS).to_excel(path, header=False, index=False)
    df = read_exhibit_file(path)
    assert list(df.columns) == ["marker_no", "description"]
    assert df["marker_no"].tolist() == [1, 2, 5, 7]          # numbering may skip numbers
    assert df["description"].iloc[1] == "half-empty bottle of vodka"


def test_reads_csv_and_skips_rows_without_number(tmp_path):
    path = tmp_path / "list.csv"
    pd.DataFrame(ROWS + [["total", "four exhibits", ""]]).to_csv(path, header=False, index=False)
    df = read_exhibit_file(path)
    assert df["marker_no"].tolist() == [1, 2, 5, 7]


def test_float_numbers_from_excel_are_ints(tmp_path):
    path = tmp_path / "l.xlsx"
    pd.DataFrame([["No", "Item"], [3.0, "pistol"], [4.0, "cartridge case"]]).to_excel(
        path, header=False, index=False)
    assert read_exhibit_file(path)["marker_no"].tolist() == [3, 4]


def test_column_override(tmp_path):
    path = tmp_path / "l.csv"
    pd.DataFrame([["x", 9, "mobile phone"], ["y", 4, "laptop"]]).to_csv(path, header=False, index=False)
    df = read_exhibit_file(path, marker_col=1, desc_col=2)
    assert df.to_dict("records") == [{"marker_no": 9, "description": "mobile phone"},
                                     {"marker_no": 4, "description": "laptop"}]
