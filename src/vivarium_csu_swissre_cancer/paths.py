from pathlib import Path

import vivarium_csu_swissre_cancer.globals as project_globals

ARTIFACT_ROOT = Path(f"/share/costeffectiveness/artifacts/{project_globals.PROJECT_NAME}/")
MODEL_SPEC_DIR = (Path(__file__).parent / 'model_specifications').resolve()
RESULTS_ROOT = Path(f'/share/costeffectiveness/results/{project_globals.PROJECT_NAME}/')

RAW_ACMR_DATA_PATH = Path(f'{ARTIFACT_ROOT}/raw' '/all_cause_mortality_rate.hdf')
