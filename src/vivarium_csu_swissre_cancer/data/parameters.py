from typing import List

from vivarium.framework.randomness import get_hash

from vivarium_csu_swissre_cancer.utilities import sample_truncnorm_distribution


HIGH_LDL_BASELINE = 4.9
LOW_DOSE_THRESHOLD = 0.744
PROBABILITY_FDC_LOW_DOSE = 0.5

LOCATION_COLUMN = 'location'
MEAN_COLUMN = 'mean_value'
SD_COLUMN = 'sd_value'

INCREASE_DOSE = 'increasing_dose'
ADD_SECOND_DRUG = 'adding_2nd_drug'
SWITCH_DRUG = 'switching_drugs'

MONOTHERAPY = 'monotherapy'
FDC = 'fdc'

STATIN_HIGH = 'high_potency_statin'
STATIN_LOW = 'low_potency_statin'

SINGLE_NO_CVE = (0, 0)
MULTI_NO_CVE = (1, 0)
SINGLE_CVE = (0, 1)
MULTI_CVE = (1, 1)


def sample_screening_parameter(screening_parameter: str, mean: float, sd: float, draw: int) -> float:
    return sample_truncnorm_distribution(get_hash(f'{screening_parameter}_draw_{draw}'), mean, sd)


def get_adjusted_probabilities(*drug_probabilities: float) -> List[float]:
    """Use on sets of raw results"""
    scaling_factor = sum(drug_probabilities)
    return [drug_probability / scaling_factor for drug_probability in drug_probabilities]
