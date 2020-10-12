from datetime import datetime
import itertools
import math
from typing import NamedTuple

from vivarium_public_health.utilities import TargetString

from vivarium_csu_swissre_breast_cancer.utilities import TruncnormDist

####################
# Project metadata #
####################

PROJECT_NAME = 'vivarium_csu_swissre_breast_cancer'
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
# Scenarios #
#############

class __Scenarios(NamedTuple):
    baseline: str = 'baseline'
    alternative: str = 'alternative'


SCENARIOS = __Scenarios()


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
    LCIS_PREVALENCE: TargetString = TargetString('sequela.lobular_carcinoma_in_situ.prevalence')
    DCIS_PREVALENCE: TargetString = TargetString('sequela.ductal_carcinoma_in_situ.prevalence')
    PREVALENCE: TargetString = TargetString('cause.breast_cancer.prevalence')
    LCIS_INCIDENCE_RATE: TargetString = TargetString('sequela.lobular_carcinoma_in_situ.incidence_rate')
    DCIS_INCIDENCE_RATE: TargetString = TargetString('sequela.ductal_carcinoma_in_situ.incidence_rate')
    INCIDENCE_RATE: TargetString = TargetString('cause.breast_cancer.incidence_rate')
    LCIS_BREAST_CANCER_TRANSITION_RATE: TargetString = TargetString('sequela.lobular_carcinoma_in_situ.transition_rate')
    DCIS_BREAST_CANCER_TRANSITION_RATE: TargetString = TargetString('sequela.ductal_carcinoma_in_situ.transition_rate')
    DISABILITY_WEIGHT: TargetString = TargetString('cause.breast_cancer.disability_weight')
    EMR: TargetString = TargetString('cause.breast_cancer.excess_mortality_rate')
    CSMR: TargetString = TargetString('cause.breast_cancer.cause_specific_mortality_rate')
    RESTRICTIONS: TargetString = TargetString('cause.breast_cancer.restrictions')

    LCIS_PREVALENCE_RATIO = TargetString('sequela.lobular_carcinoma_in_situ.prevalence_ratio')
    DCIS_PREVALENCE_RATIO = TargetString('sequela.ductal_carcinoma_in_situ.prevalence_ratio')

    REMISSION_RATE_VALUE = 0.1

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


########################
# Screening and Treatment Model Constants #
########################

PROBABILITY_ATTENDING_SCREENING_KEY = 'probability_attending_screening'
ATTENDED_PREVIOUS_SCREENING_MULTIPLIER = 1.89
RAMP_UP_START = datetime(2021, 1, 1)
RAMP_UP_END = datetime(2030, 1, 1)


class __Screening(NamedTuple):
    REMISSION_SENSITIVITY: TruncnormDist = TruncnormDist('remission_sensitivity', 1.0, 0.0)
    REMISSION_SPECIFICITY: TruncnormDist = TruncnormDist('remission_specificity', 1.0, 0.0)

    MAMMOGRAM_SENSITIVITY: TruncnormDist = TruncnormDist('mammogram_sensitivity', 0.848, 0.00848)
    MAMMOGRAM_SPECIFICITY: TruncnormDist = TruncnormDist('mammogram_specificity', 1.0, 0.0)

    MRI_SENSITIVITY: TruncnormDist = TruncnormDist('mri_sensitivity', 0.91, 0.0091)
    MRI_SPECIFICITY: TruncnormDist = TruncnormDist('mri_specificity', 1.0, 0.0)

    ULTRASOUND_SENSITIVITY: TruncnormDist = TruncnormDist('ultrasound_sensitivity', 0.737, 0.00737)
    ULTRASOUND_SPECIFICITY: TruncnormDist = TruncnormDist('ultrasound_specificity', 1.0, 0.0)

    MAMMOGRAM_ULTRASOUND_SENSITIVITY: TruncnormDist = TruncnormDist('mammogram_ultrasound_sensitivity', 0.939, 0.00939)
    MAMMOGRAM_ULTRASOUND_SPECIFICITY: TruncnormDist = TruncnormDist('mammogram_ultrasound_specificity', 1.0, 0.0)

    BASE_ATTENDANCE: TruncnormDist = TruncnormDist('start_attendance_base', 0.3, 0.003,
                                                   key=PROBABILITY_ATTENDING_SCREENING_KEY)
    START_ATTENDED_PREV_ATTENDANCE: TruncnormDist = TruncnormDist('start_attendance_attended_prev', 0.397, 0.00397,
                                                                  key=PROBABILITY_ATTENDING_SCREENING_KEY)
    START_NOT_ATTENDED_PREV_ATTENDANCE: TruncnormDist = TruncnormDist('start_attendance_not_attended_prev', 0.258,
                                                                      0.00258, key=PROBABILITY_ATTENDING_SCREENING_KEY)
    END_ATTENDED_PREV_ATTENDANCE: TruncnormDist = TruncnormDist('end_attendance_attended_prev', 0.782, 0.00782,
                                                                key=PROBABILITY_ATTENDING_SCREENING_KEY)
    END_NOT_ATTENDED_PREV_ATTENDANCE: TruncnormDist = TruncnormDist('end_attendance_not_attended_prev', 0.655, 0.00655,
                                                                    key=PROBABILITY_ATTENDING_SCREENING_KEY)

    @property
    def name(self):
        return 'screening_result'

    @property
    def log_name(self):
        return 'screening result'


SCREENING = __Screening()


class __Treatment(NamedTuple):
    LOG_LCIS_EFFICACY: TruncnormDist = TruncnormDist('lcis_treatment_efficacy', math.log(0.44), 0.48, -100.0, 0.0)
    DCIS_EFFICACY: TruncnormDist = TruncnormDist('dcis_treatment_efficacy', 0.40, 0.05, 0.0, 0.99)

    LCIS_COVERAGE_PARAMS = (0.15, 0.2, 0.25)
    DCIS_COVERAGE_PARAMS = (0.95, 0.975, 1.0)


TREATMENT = __Treatment()

DAYS_UNTIL_NEXT_ANNUAL = TruncnormDist('days_until_next_annual', 364.0, 156.0, 100.0, 700.0)
DAYS_UNTIL_NEXT_BIENNIAL = TruncnormDist('days_until_next_biennial', 728.0, 156.0, 200.0, 1400.0)

ATTENDED_LAST_SCREENING = 'attended_last_screening'
PREVIOUS_SCREENING_DATE = 'previous_screening_date'
NEXT_SCREENING_DATE = 'next_screening_date'

###########################
# State Machine Model variables #
###########################


class TransitionString(str):

    def __new__(cls, value):
        # noinspection PyArgumentList
        obj = str.__new__(cls, value.lower())
        obj.from_state, obj.to_state = value.split('_TO_')
        return obj


BREAST_CANCER_MODEL_NAME = BREAST_CANCER.name
SUSCEPTIBLE_STATE_NAME = f'susceptible_to_{BREAST_CANCER_MODEL_NAME}'
LCIS_STATE_NAME = 'lobular_carcinoma_in_situ'
DCIS_STATE_NAME = 'ductal_carcinoma_in_situ'
BREAST_CANCER_STATE_NAME = 'breast_cancer'
RECOVERED_STATE_NAME = f'recovered_from_{BREAST_CANCER_MODEL_NAME}'
BREAST_CANCER_MODEL_STATES = (
    SUSCEPTIBLE_STATE_NAME,
    LCIS_STATE_NAME,
    DCIS_STATE_NAME,
    BREAST_CANCER_STATE_NAME,
    RECOVERED_STATE_NAME,
)
BREAST_CANCER_MODEL_TRANSITIONS = (
    TransitionString(f'{SUSCEPTIBLE_STATE_NAME}_TO_{LCIS_STATE_NAME}'),
    TransitionString(f'{SUSCEPTIBLE_STATE_NAME}_TO_{DCIS_STATE_NAME}'),
    TransitionString(f'{DCIS_STATE_NAME}_TO_{BREAST_CANCER_STATE_NAME}'),
    TransitionString(f'{LCIS_STATE_NAME}_TO_{BREAST_CANCER_STATE_NAME}'),
    TransitionString(f'{BREAST_CANCER_STATE_NAME}_TO_{RECOVERED_STATE_NAME}')
)


SCREENING_RESULT_MODEL_NAME = SCREENING.name
NEGATIVE_STATE_NAME = 'negative_screening'
POSITIVE_LCIS_STATE_NAME = 'positive_screening_lobular_carcinoma_in_situ'
POSITIVE_DCIS_STATE_NAME = 'positive_screening_ductal_carcinoma_in_situ'
POSITIVE_BREAST_CANCER_STATE_NAME = 'positive_screening_breast_cancer'
REMISSION_STATE_NAME = 'remission'
SCREENING_MODEL_STATES = (
    NEGATIVE_STATE_NAME,
    POSITIVE_LCIS_STATE_NAME,
    POSITIVE_DCIS_STATE_NAME,
    POSITIVE_BREAST_CANCER_STATE_NAME,
)
SCREENING_MODEL_TRANSITIONS = (
    TransitionString(f'{NEGATIVE_STATE_NAME}_TO_{POSITIVE_LCIS_STATE_NAME}'),
    TransitionString(f'{NEGATIVE_STATE_NAME}_TO_{POSITIVE_DCIS_STATE_NAME}'),
    TransitionString(f'{NEGATIVE_STATE_NAME}_TO_{POSITIVE_BREAST_CANCER_STATE_NAME}'),
    TransitionString(f'{POSITIVE_DCIS_STATE_NAME}_TO_{POSITIVE_BREAST_CANCER_STATE_NAME}'),
    TransitionString(f'{POSITIVE_LCIS_STATE_NAME}_TO_{POSITIVE_BREAST_CANCER_STATE_NAME}'),
    TransitionString(f'{NEGATIVE_STATE_NAME}_TO_{REMISSION_STATE_NAME}'),
    TransitionString(f'{POSITIVE_LCIS_STATE_NAME}_TO_{REMISSION_STATE_NAME}'),
    TransitionString(f'{POSITIVE_DCIS_STATE_NAME}_TO_{REMISSION_STATE_NAME}'),
    TransitionString(f'{POSITIVE_BREAST_CANCER_STATE_NAME}_TO_{REMISSION_STATE_NAME}'),
)

STATE_MACHINE_MAP = {
    BREAST_CANCER_MODEL_NAME: {
        'states': BREAST_CANCER_MODEL_STATES,
        'transitions': BREAST_CANCER_MODEL_TRANSITIONS,
    },
    SCREENING_RESULT_MODEL_NAME: {
        'states': SCREENING_MODEL_STATES,
        'transitions': SCREENING_MODEL_TRANSITIONS,
    }
}


def get_screened_state(breast_cancer_model_state: str) -> str:
    """Get screening result state name from a breast cancer model state"""
    return {
        SUSCEPTIBLE_STATE_NAME: NEGATIVE_STATE_NAME,
        LCIS_STATE_NAME: POSITIVE_LCIS_STATE_NAME,
        DCIS_STATE_NAME: POSITIVE_DCIS_STATE_NAME,
        BREAST_CANCER_STATE_NAME: POSITIVE_BREAST_CANCER_STATE_NAME,
        RECOVERED_STATE_NAME: REMISSION_STATE_NAME,
    }[breast_cancer_model_state]


STATES = tuple(state for model in STATE_MACHINE_MAP.values() for state in model['states'])
TRANSITIONS = tuple(state for model in STATE_MACHINE_MAP.values() for state in model['transitions'])


########################
# Risk Model Constants #
########################

#################################
# Results columns and variables #
#################################

TOTAL_POPULATION_COLUMN = 'total_population'
TOTAL_YLDS_COLUMN = 'years_lived_with_disability'
TOTAL_YLLS_COLUMN = 'years_of_life_lost'

SCREENING_SCHEDULED = 'screening_scheduled_count'
SCREENING_ATTENDED = 'screening_attended_count'



# Columns from parallel runs
INPUT_DRAW_COLUMN = 'input_draw'
RANDOM_SEED_COLUMN = 'random_seed'
OUTPUT_SCENARIO_COLUMN = 'screening_algorithm.scenario'

STANDARD_COLUMNS = {
    'total_population': TOTAL_POPULATION_COLUMN,
    'total_ylls': TOTAL_YLLS_COLUMN,
    'total_ylds': TOTAL_YLDS_COLUMN,
}

THROWAWAY_COLUMNS = [f'{state}_event_count' for state in STATES]

TOTAL_POPULATION_COLUMN_TEMPLATE = 'total_population_{POP_STATE}'
PERSON_TIME_COLUMN_TEMPLATE = 'person_time_in_{YEAR}_among_{SEX}_age_cohort_{AGE_COHORT}_family_history_{HISTORY}'
DEATH_COLUMN_TEMPLATE = ('death_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_among_{SEX}_age_cohort_{AGE_COHORT}'
                         '_family_history_{HISTORY}_screening_result_{SCREENING_STATE}')
YLLS_COLUMN_TEMPLATE = ('ylls_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_among_{SEX}_age_cohort_{AGE_COHORT}'
                        '_family_history_{HISTORY}_screening_result_{SCREENING_STATE}')
YLDS_COLUMN_TEMPLATE = ('ylds_due_to_{CAUSE_OF_DISABILITY}_in_{YEAR}_among_{SEX}_age_cohort_{AGE_COHORT}'
                        '_family_history_{HISTORY}')
DISEASE_STATE_PERSON_TIME_COLUMN_TEMPLATE = ('{DISEASE_STATE}_person_time_in_{YEAR}_among_{SEX}_age_cohort_{AGE_COHORT}'
                                             '_family_history_{HISTORY}_screening_result_{SCREENING_STATE}')
SCREENING_STATE_PERSON_TIME_COLUMN_TEMPLATE = ('{SCREENING_STATE}_person_time_in_{YEAR}_among_{SEX}'
                                               '_age_cohort_{AGE_COHORT}_family_history_{HISTORY}')
DISEASE_TRANSITION_COUNT_COLUMN_TEMPLATE = ('{DISEASE_TRANSITION}_event_count_in_{YEAR}_among_{SEX}'
                                            '_age_cohort_{AGE_COHORT}_family_history_{HISTORY}'
                                            '_screening_result_{SCREENING_STATE}')
SCREENING_TRANSITION_COUNT_COLUMN_TEMPLATE = ('{SCREENING_TRANSITION}_event_count_in_{YEAR}_among_{SEX}'
                                              '_age_cohort_{AGE_COHORT}_family_history_{HISTORY}')
EVENT_COUNT_COLUMN_TEMPLATE = '{EVENT}_in_{YEAR}_among_{SEX}_age_cohort_{AGE_COHORT}_family_history_{HISTORY}'
TREATMENT_COUNT_TEMPLATE = ('began_{TREATMENT_TYPE}_treatment_count_in_{YEAR}_among_{SEX}_age_cohort_{AGE_COHORT}'
                            '_family_history_{HISTORY}')

COLUMN_TEMPLATES = {
    'population': TOTAL_POPULATION_COLUMN_TEMPLATE,
    'person_time': PERSON_TIME_COLUMN_TEMPLATE,
    'deaths': DEATH_COLUMN_TEMPLATE,
    'ylls': YLLS_COLUMN_TEMPLATE,
    'ylds': YLDS_COLUMN_TEMPLATE,
    'disease_state_person_time': DISEASE_STATE_PERSON_TIME_COLUMN_TEMPLATE,
    'screening_state_person_time': SCREENING_STATE_PERSON_TIME_COLUMN_TEMPLATE,
    'disease_transition_count': DISEASE_TRANSITION_COUNT_COLUMN_TEMPLATE,
    'screening_transition_count': SCREENING_TRANSITION_COUNT_COLUMN_TEMPLATE,
    'event_count': EVENT_COUNT_COLUMN_TEMPLATE,
    'treatment_count': TREATMENT_COUNT_TEMPLATE,
}

NON_COUNT_TEMPLATES = [
]

POP_STATES = ('living', 'dead', 'tracked', 'untracked')
SEXES = ('male', 'female')
YEARS = tuple(range(2020, 2041))
AGE_COHORTS = tuple(f'{2020 - (x + 5)}_to_{2020 - x}' for x in range(15, 85, 5))
EVENTS = (SCREENING_SCHEDULED, SCREENING_ATTENDED)
CAUSES_OF_DEATH = ('other_causes', BREAST_CANCER_STATE_NAME,)
CAUSES_OF_DISABILITY = (BREAST_CANCER_STATE_NAME,)
FAMILY_HISTORY_STATE = ('positive', 'negative',)
TREATMENT_TYPES = ('dcis', 'lcis',)

TEMPLATE_FIELD_MAP = {
    'POP_STATE': POP_STATES,
    'YEAR': YEARS,
    'SEX': SEXES,
    'AGE_COHORT': AGE_COHORTS,
    'CAUSE_OF_DEATH': CAUSES_OF_DEATH,
    'CAUSE_OF_DISABILITY': CAUSES_OF_DISABILITY,
    'DISEASE_STATE': BREAST_CANCER_MODEL_STATES,
    'SCREENING_STATE': SCREENING_MODEL_STATES,
    'DISEASE_TRANSITION': BREAST_CANCER_MODEL_TRANSITIONS,
    'SCREENING_TRANSITION': SCREENING_MODEL_TRANSITIONS,
    'EVENT': EVENTS,
    'HISTORY': FAMILY_HISTORY_STATE,
    'TREATMENT_TYPE': TREATMENT_TYPES,
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
