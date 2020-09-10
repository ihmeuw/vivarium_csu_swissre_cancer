from pathlib import Path

import vivarium_csu_swissre_breast_cancer
import vivarium_csu_swissre_breast_cancer.globals as project_globals

BASE_DIR = Path(vivarium_csu_swissre_breast_cancer.__file__).resolve().parent
ARTIFACT_ROOT = Path(f"/share/costeffectiveness/artifacts/{project_globals.PROJECT_NAME}/")
MODEL_SPEC_DIR = (Path(__file__).parent / 'model_specifications').resolve()
RESULTS_ROOT = Path(f'/share/costeffectiveness/results/{project_globals.PROJECT_NAME}/')

RAW_DATA_ROOT = ARTIFACT_ROOT / 'raw'
RAW_ACMR_DATA_PATH = RAW_DATA_ROOT / 'all_cause_mortality_rate.hdf'
RAW_INCIDENCE_RATE_DATA_PATH = RAW_DATA_ROOT / 'incidence_rate.hdf'
RAW_MORTALITY_DATA_PATH = RAW_DATA_ROOT / 'mortality.hdf'
RAW_PREVALENCE_DATA_PATH = RAW_DATA_ROOT / 'prevalence.hdf'
