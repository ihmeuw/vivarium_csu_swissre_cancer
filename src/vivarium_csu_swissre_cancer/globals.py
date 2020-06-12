import itertools
from pathlib import Path
from typing import NamedTuple

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
# TODO - remove if you don't need lbwsg
LBWSG_MODEL_NAME = 'low_birth_weight_and_short_gestation'


class __LBWSG_MISSING_CATEGORY(NamedTuple):
    CAT: str = 'cat212'
    NAME: str = 'Birth prevalence - [37, 38) wks, [1000, 1500) g'
    EXPOSURE: float = 0.


LBWSG_MISSING_CATEGORY = __LBWSG_MISSING_CATEGORY()


#################################
# Results columns and variables #
#################################

TOTAL_POPULATION_COLUMN = 'total_population'
TOTAL_YLDS_COLUMN = 'years_lived_with_disability'
TOTAL_YLLS_COLUMN = 'years_of_life_lost'

STANDARD_COLUMNS = {
    'total_population': TOTAL_POPULATION_COLUMN,
    'total_ylls': TOTAL_YLLS_COLUMN,
    'total_ylds': TOTAL_YLDS_COLUMN,
}

TOTAL_POPULATION_COLUMN_TEMPLATE = 'total_population_{POP_STATE}'
PERSON_TIME_COLUMN_TEMPLATE = 'person_time_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'
DEATH_COLUMN_TEMPLATE = 'death_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'
YLLS_COLUMN_TEMPLATE = 'ylls_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'
YLDS_COLUMN_TEMPLATE = 'ylds_due_to_{CAUSE_OF_DISABILITY}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'
STATE_PERSON_TIME_COLUMN_TEMPLATE = '{STATE}_person_time_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'
TRANSITION_COUNT_COLUMN_TEMPLATE = '{TRANSITION}_event_count_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'

COLUMN_TEMPLATES = {
    'population': TOTAL_POPULATION_COLUMN_TEMPLATE,
    'person_time': PERSON_TIME_COLUMN_TEMPLATE,
    'deaths': DEATH_COLUMN_TEMPLATE,
    'ylls': YLLS_COLUMN_TEMPLATE,
    'ylds': YLDS_COLUMN_TEMPLATE,
    'state_person_time': STATE_PERSON_TIME_COLUMN_TEMPLATE,
    'transition_count': TRANSITION_COUNT_COLUMN_TEMPLATE,
}

POP_STATES = ('living', 'dead', 'tracked', 'untracked')
SEXES = ('male', 'female')
# TODO - add literals for years in the model
YEARS = ()
# TODO - add literals for ages in the model
AGE_GROUPS = ()
# TODO - add causes of death
CAUSES_OF_DEATH = (
    'other_causes',
    BREAST_CANCER_STATE_NAME,
)
# TODO - add causes of disability
CAUSES_OF_DISABILITY = (
    BREAST_CANCER_STATE_NAME,
)

TEMPLATE_FIELD_MAP = {
    'POP_STATE': POP_STATES,
    'YEAR': YEARS,
    'SEX': SEXES,
    'AGE_GROUP': AGE_GROUPS,
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

