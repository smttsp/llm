# from read_mimic import read_csvs

from .patients import get_patient_statistics
from .admissions import get_admissions_statistics
from .services import get_services_statistics
from .labitem import get_labitem_statistics
from .diagnosis_icd import get_diagnosis_statistics
from .omr import get_omr_statistics
