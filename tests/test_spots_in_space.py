from sis import example_function
from sis import unpack_test_data
from sis import SpotTable
import zipfile
import pathlib



XENIUM_INTERNAL_NAME = 'output-XETG00044__0025057__Test__20240307__194734'

MERSCOPE_INTERNAL_NAME = '202403080826_1334050571_VMSC21002_region_0'




def test_example_function():
    assert example_function() == 2


def test_unpack_test_data():
    (a,b) = unpack_test_data()
    print(a)
    print(b)
    assert all([a==XENIUM_INTERNAL_NAME, b ==MERSCOPE_INTERNAL_NAME ])