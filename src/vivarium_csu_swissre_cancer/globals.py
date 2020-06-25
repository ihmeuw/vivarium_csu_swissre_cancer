import itertools
from typing import List, NamedTuple, Tuple

from vivarium_csu_swissre_cancer.utilities import TruncnormDist

####################
# Project metadata #
####################

PROJECT_NAME = 'vivarium_csu_swissre_cancer'
CLUSTER_PROJECT = 'proj_csu'

CLUSTER_QUEUE = 'all.q'
MAKE_ARTIFACT_MEM = '3G'
MAKE_ARTIFACT_CPU = '1'
MAKE_ARTIFACT_RUNTIME = '3:00:00'
MAKE_ARTIFACT_SLEEP = 10

LOCATIONS = [
    'SwissRE Coverage',
]


SWISSRE_LOCATION_WEIGHTS = {
    'Tianjin': 0.18,
    'Jiangsu': 0.28,
    'Guangdong': 0.24,
    'Henan': 0.17,
    'Heilongjiang': 0.16,
}


#############
# Data Keys #
#############

METADATA_LOCATIONS = 'metadata.locations'


class __Population(NamedTuple):
    STRUCTURE: str = 'population.structure'
    AGE_BINS: str = 'population.age_bins'
    DEMOGRAPHY: str = 'population.demographic_dimensions'
    TMRLE: str = 'population.theoretical_minimum_risk_life_expectancy'
    ACMR: str = 'cause.all_causes.cause_specific_mortality_rate'

    @property
    def name(self):
        return 'population'

    @property
    def log_name(self):
        return 'population'


POPULATION = __Population()


class __BreastCancer(NamedTuple):
    LCIS_PREVALENCE_RATIO: str = 'sequela.lobular_carcinoma_in_situ.prevalence_ratio'
    DCIS_PREVALENCE_RATIO: str = 'sequela.ductal_carcinoma_in_situ.prevalence_ratio'
    LCIS_PREVALENCE: str = 'sequela.lobular_carcinoma_in_situ.prevalence'
    DCIS_PREVALENCE: str = 'sequela.ductal_carcinoma_in_situ.prevalence'
    PREVALENCE: str = 'cause.breast_cancer.prevalence'
    LCIS_INCIDENCE_RATE: str = 'sequela.lobular_carcinoma_in_situ.incidence_rate'
    DCIS_INCIDENCE_RATE: str = 'sequela.ductal_carcinoma_in_situ.incidence_rate'
    INCIDENCE_RATE: str = 'cause.breast_cancer.incidence_rate'
    LCIS_BREAST_CANCER_TRANSITION_RATE: str = 'sequela.lobular_carcinoma_in_situ.transition_rate'
    DCIS_BREAST_CANCER_TRANSITION_RATE: str = 'sequela.ductal_carcinoma_in_situ.transition_rate'
    DISABILITY_WEIGHT: str = 'cause.breast_cancer.disability_weight'
    EMR: str = 'cause.breast_cancer.excess_mortality_rate'
    CSMR: str = 'cause.breast_cancer.cause_specific_mortality_rate'
    RESTRICTIONS: str = 'cause.breast_cancer.restrictions'

    @property
    def name(self):
        return 'breast_cancer'

    @property
    def log_name(self):
        return 'breast cancer'


BREAST_CANCER = __BreastCancer()


MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    BREAST_CANCER
]

###########################
# Disease Model variables #
###########################


class TransitionString(str):

    def __new__(cls, value):
        # noinspection PyArgumentList
        obj = str.__new__(cls, value.lower())
        obj.from_state, obj.to_state = value.split('_TO_')
        return obj


BREAST_CANCER_MODEL_NAME = BREAST_CANCER.name
BREAST_CANCER_SUSCEPTIBLE_STATE_NAME = f'susceptible_to_{BREAST_CANCER_MODEL_NAME}'
LCIS_STATE_NAME = 'lobular_carcinoma_in_situ'
DCIS_STATE_NAME = 'ductal_carcinoma_in_situ'
BREAST_CANCER_STATE_NAME = 'breast_cancer'
BREAST_CANCER_MODEL_STATES = (
    BREAST_CANCER_SUSCEPTIBLE_STATE_NAME,
    LCIS_STATE_NAME,
    DCIS_STATE_NAME,
    BREAST_CANCER_STATE_NAME
)
BREAST_CANCER_MODEL_TRANSITIONS = (
    TransitionString(f'{BREAST_CANCER_SUSCEPTIBLE_STATE_NAME}_TO_{LCIS_STATE_NAME}'),
    TransitionString(f'{BREAST_CANCER_SUSCEPTIBLE_STATE_NAME}_TO_{DCIS_STATE_NAME}'),
    TransitionString(f'{DCIS_STATE_NAME}_TO_{BREAST_CANCER_STATE_NAME}'),
    TransitionString(f'{LCIS_STATE_NAME}_TO_{BREAST_CANCER_STATE_NAME}'),
)

DISEASE_MODELS = (
    BREAST_CANCER_MODEL_NAME,
)
DISEASE_MODEL_MAP = {
    BREAST_CANCER_MODEL_NAME: {
        'states': BREAST_CANCER_MODEL_STATES,
        'transitions': BREAST_CANCER_MODEL_TRANSITIONS,
    },
}

STATES = tuple(state for model in DISEASE_MODELS for state in DISEASE_MODEL_MAP[model]['states'])
TRANSITIONS = tuple(transition for model in DISEASE_MODELS for transition in DISEASE_MODEL_MAP[model]['transitions'])


########################
# Risk Model Constants #
########################


########################
# Screening and Treatment Model Constants #
########################

class __Screening(NamedTuple):
    MAMMOGRAM_SENSITIVITY: TruncnormDist = TruncnormDist('mammogram_sensitivity', 0.848, 0.00848)
    MAMMOGRAM_SPECIFICITY: TruncnormDist = TruncnormDist('mammogram_specificity', 1.0, 0.0)
    
    MRI_SENSITIVITY: TruncnormDist = TruncnormDist('mri_sensitivity', 0.91, 0.0091)
    MRI_SPECIFICITY: TruncnormDist = TruncnormDist('mri_specificity', 1.0, 0.0)
    
    ULTRASOUND_SENSITIVITY: TruncnormDist = TruncnormDist('ultrasound_sensitivity', 0.737, 0.00737)
    ULTRASOUND_SPECIFICITY: TruncnormDist = TruncnormDist('ultrasound_specificity', 1.0, 0.0)

    MAMMOGRAM_ULTRASOUND_SENSITIVITY: TruncnormDist = TruncnormDist('mammogram_ultrasound_sensitivity', 0.939, 0.00939)
    MAMMOGRAM_ULTRASOUND_SPECIFICITY: TruncnormDist = TruncnormDist('mammogram_ultrasound_specificity', 1.0, 0.0)

    BASE_PROBABILITY: TruncnormDist = TruncnormDist('probability_attending_screening', 0.3, 0.003)
    PROBABILITY_GIVEN_ADHERENT: TruncnormDist = TruncnormDist('probability_attending_screening', 0.397, 0.00397)
    PROBABILITY_GIVEN_NOT_ADHERENT: TruncnormDist = TruncnormDist('probability_attending_screening', 0.258, 0.00258)


SCREENING = __Screening()


DAYS_UNTIL_NEXT_ANNUAL = TruncnormDist('days_until_next_annual', 364.0, 156.0, 100.0, 700.0)
DAYS_UNTIL_NEXT_BIENNIAL = TruncnormDist('days_until_next_biennial', 728.0, 156.0, 200.0, 1400.0)

SCREENING_RESULT = 'screening_result'
ATTENDED_LAST_SCREENING = 'attended_last_screening'
NEXT_SCREENING_DATE = 'next_screening_date'

#################################
# Results columns and variables #
#################################

TOTAL_POPULATION_COLUMN = 'total_population'
TOTAL_YLDS_COLUMN = 'years_lived_with_disability'
TOTAL_YLLS_COLUMN = 'years_of_life_lost'

# Columns from parallel runs
INPUT_DRAW_COLUMN = 'input_draw'
RANDOM_SEED_COLUMN = 'random_seed'
# TODO add back when we have scenarios
# OUTPUT_SCENARIO_COLUMN = 'ldlc_treatment_algorithm.scenario'

STANDARD_COLUMNS = {
    'total_population': TOTAL_POPULATION_COLUMN,
    'total_ylls': TOTAL_YLLS_COLUMN,
    'total_ylds': TOTAL_YLDS_COLUMN,
}

THROWAWAY_COLUMNS = [f'{state}_event_count' for state in STATES]

TOTAL_POPULATION_COLUMN_TEMPLATE = 'total_population_{POP_STATE}'
PERSON_TIME_COLUMN_TEMPLATE = 'person_time_in_{YEAR}_among_{SEX}_age_cohort_{AGE_COHORT}'
DEATH_COLUMN_TEMPLATE = 'death_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_among_{SEX}_age_cohort_{AGE_COHORT}'
YLLS_COLUMN_TEMPLATE = 'ylls_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_among_{SEX}_age_cohort_{AGE_COHORT}'
YLDS_COLUMN_TEMPLATE = 'ylds_due_to_{CAUSE_OF_DISABILITY}_in_{YEAR}_among_{SEX}_age_cohort_{AGE_COHORT}'
STATE_PERSON_TIME_COLUMN_TEMPLATE = '{STATE}_person_time_in_{YEAR}_among_{SEX}_age_cohort_{AGE_COHORT}'
TRANSITION_COUNT_COLUMN_TEMPLATE = '{TRANSITION}_event_count_in_{YEAR}_among_{SEX}_age_cohort_{AGE_COHORT}'

COLUMN_TEMPLATES = {
    'population': TOTAL_POPULATION_COLUMN_TEMPLATE,
    'person_time': PERSON_TIME_COLUMN_TEMPLATE,
    'deaths': DEATH_COLUMN_TEMPLATE,
    'ylls': YLLS_COLUMN_TEMPLATE,
    'ylds': YLDS_COLUMN_TEMPLATE,
    'state_person_time': STATE_PERSON_TIME_COLUMN_TEMPLATE,
    'transition_count': TRANSITION_COUNT_COLUMN_TEMPLATE,
}

NON_COUNT_TEMPLATES = [
]

POP_STATES = ('living', 'dead', 'tracked', 'untracked')
SEXES = ('male', 'female')
YEARS = tuple(range(2020, 2039))
AGE_COHORTS = tuple(f'{2020 - (x + 5)}_to_{2020 - x}' for x in range(15, 85, 5))
CAUSES_OF_DEATH = ('other_causes', BREAST_CANCER_STATE_NAME,)
CAUSES_OF_DISABILITY = (BREAST_CANCER_STATE_NAME,)

TEMPLATE_FIELD_MAP = {
    'POP_STATE': POP_STATES,
    'YEAR': YEARS,
    'SEX': SEXES,
    'AGE_COHORT': AGE_COHORTS,
    'CAUSE_OF_DEATH': CAUSES_OF_DEATH,
    'CAUSE_OF_DISABILITY': CAUSES_OF_DISABILITY,
    'STATE': STATES,
    'TRANSITION': TRANSITIONS,
}


def RESULT_COLUMNS(kind='all'):
    if kind not in COLUMN_TEMPLATES and kind != 'all':
        raise ValueError(f'Unknown result column type {kind}')
    columns = []
    if kind == 'all':
        for k in COLUMN_TEMPLATES:
            columns += RESULT_COLUMNS(k)
        columns = list(STANDARD_COLUMNS.values()) + columns
    else:
        template = COLUMN_TEMPLATES[kind]
        filtered_field_map = {field: values
                              for field, values in TEMPLATE_FIELD_MAP.items() if f'{{{field}}}' in template}
        fields, value_groups = filtered_field_map.keys(), itertools.product(*filtered_field_map.values())
        for value_group in value_groups:
            columns.append(template.format(**{field: value for field, value in zip(fields, value_group)}))
    return columns

