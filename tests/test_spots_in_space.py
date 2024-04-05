from sis import example_function
from sis import unpack_test_data
from sis import SegmentedSpotTable
import zipfile
import pathlib



XENIUM_INTERNAL_NAME = 'output-XETG00044__0025057__Test__20240307__194734'

MERSCOPE_INTERNAL_NAME = '202403080826_1334050571_VMSC21002_region_0'





SIS_DIR = pathlib.Path().absolute()
print(SIS_DIR)

XENIUM_STEM = "xenium_test"
MERSCOPE_STEM = "merscope_test"

TEST_DIR = SIS_DIR.joinpath("tests").joinpath("spatial_test_data")
XENIUM_DIR = TEST_DIR.joinpath(XENIUM_STEM)
MERSCOPE_DIR = TEST_DIR.joinpath(MERSCOPE_STEM)

XENIUM_DIR.mkdir(exist_ok = True)
MERSCOPE_DIR.mkdir(exist_ok = True)
DETECTED_TRANSCRIPTS_CSV = MERSCOPE_DIR.joinpath("region_0").joinpath("detected_transcripts.csv")

MERSCOPE_N_CELL_IDS = 1041363




def test_example_function():
    assert example_function() == 2


def test_unpack_test_data():
    (a,b) = unpack_test_data()
    print(a)
    print(b)
    assert all([a==XENIUM_INTERNAL_NAME, b ==MERSCOPE_INTERNAL_NAME ])



def test_load_merscope():

    a = SegmentedSpotTable.load_merscope(DETECTED_TRANSCRIPTS_CSV, DETECTED_TRANSCRIPTS_CSV.parent.joinpath("detected_transcripts.npz"))
    print(MERSCOPE_N_CELL_IDS)
    assert a.cell_ids.shape[0] == MERSCOPE_N_CELL_IDS