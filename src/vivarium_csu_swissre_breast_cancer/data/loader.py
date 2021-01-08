"""Loads, standardizes and validates input data for the simulation.

Abstract the extract and transform pieces of the artifact ETL.
The intent here is to provide a uniform interface around this portion
of artifact creation. The value of this interface shows up when more
complicated data needs are part of the project. See the BEP project
for an example.

`BEP <https://github.com/ihmeuw/vivarium_gates_bep/blob/master/src/vivarium_gates_bep/data/loader.py>`_

.. admonition::

   No logging is done here. Logging is done in vivarium inputs itself and forwarded.
"""
from pathlib import Path
import random
from typing import Dict

from gbd_mapping import causes, covariates, risk_factors
import numpy as np
import pandas as pd
from vivarium.framework.artifact import EntityKey
from vivarium.framework.randomness import get_hash
from vivarium_gbd_access import gbd
from vivarium_inputs import interface
from vivarium_inputs.mapping_extension import alternative_risk_factors

from vivarium_csu_swissre_breast_cancer import paths, globals as project_globals
from vivarium_csu_swissre_breast_cancer.utilities import TruncnormDist

ARTIFACT_INDEX_COLUMNS = [
    'location',
    'sex',
    'age_start',
    'age_end',
    'year_start',
    'year_end',
]

LCIS_DURATION = 5.0
DCIS_DURATION = 3.0

PREVALENCE_RATIOS = {
    project_globals.BREAST_CANCER.LCIS_PREVALENCE_RATIO: TruncnormDist(
        project_globals.BREAST_CANCER.LCIS_PREVALENCE_RATIO, 0.07, 0.06, 0.0, 100.0
    ),
    project_globals.BREAST_CANCER.DCIS_PREVALENCE_RATIO: TruncnormDist(
        project_globals.BREAST_CANCER.DCIS_PREVALENCE_RATIO, 0.33, 0.10, 0.0, 100.0
    ),
}


def get_data(lookup_key: str, location: str) -> pd.DataFrame:
    """Retrieves data from an appropriate source.

    Parameters
    ----------
    lookup_key
        The key that will eventually get put in the artifact with
        the requested data.
    location
        The location to get data for.

    Returns
    -------
        The requested data.

    """
    mapping = {
        project_globals.POPULATION.STRUCTURE: load_population_structure,
        project_globals.POPULATION.AGE_BINS: load_age_bins,
        project_globals.POPULATION.DEMOGRAPHY: load_demographic_dimensions,
        project_globals.POPULATION.TMRLE: load_theoretical_minimum_risk_life_expectancy,
        project_globals.POPULATION.ACMR: load_acmr,

        project_globals.BREAST_CANCER.LCIS_PREVALENCE: load_prevalence,
        project_globals.BREAST_CANCER.DCIS_PREVALENCE: load_prevalence,
        project_globals.BREAST_CANCER.PREVALENCE: load_prevalence,
        project_globals.BREAST_CANCER.LCIS_INCIDENCE_RATE: load_incidence_rate,
        project_globals.BREAST_CANCER.DCIS_INCIDENCE_RATE: load_incidence_rate,
        project_globals.BREAST_CANCER.INCIDENCE_RATE: load_incidence_rate,
        project_globals.BREAST_CANCER.LCIS_BREAST_CANCER_TRANSITION_RATE: load_breast_cancer_transition_rate,
        project_globals.BREAST_CANCER.DCIS_BREAST_CANCER_TRANSITION_RATE: load_breast_cancer_transition_rate,
        project_globals.BREAST_CANCER.DISABILITY_WEIGHT: load_disability_weight,
        project_globals.BREAST_CANCER.EMR: load_emr,
        project_globals.BREAST_CANCER.CSMR: load_csmr,
        project_globals.BREAST_CANCER.RESTRICTIONS: load_metadata,
    }
    return mapping[lookup_key](lookup_key, location)


def load_population_structure(key: str, location: str) -> pd.DataFrame:
    def get_row(sex, year):
        return {
            'location': location,
            'sex': sex,
            'age_start': 0,
            'age_end': 85,
            'year_start': year,
            'year_end': year + 1,
            'value': 50,
        }

    # TODO there is an issue in vivarium_public_health.population.data_transformations.assign_demographic_proportions()
    #   where this fails if there is only one provided year
    return pd.DataFrame([
        get_row('Male', 2019),
        get_row('Female', 2019),
        get_row('Male', 2020),
        get_row('Female', 2020),
    ]).set_index(['location', 'sex', 'age_start', 'age_end', 'year_start', 'year_end'])


def load_age_bins(key: str, location: str) -> pd.DataFrame:
    return interface.get_age_bins()


def load_demographic_dimensions(key: str, location: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            'location': location,
            'sex': 'Male',
            'age_start': 0,
            'age_end': 85,
            'year_start': 2019,
            'year_end': 2020,
        }, {
            'location': location,
            'sex': 'Female',
            'age_start': 0,
            'age_end': 85,
            'year_start': 2019,
            'year_end': 2020,
        },
    ]).set_index(['location', 'sex', 'age_start', 'age_end', 'year_start', 'year_end'])


def load_theoretical_minimum_risk_life_expectancy(key: str, location: str) -> pd.DataFrame:
    return interface.get_theoretical_minimum_risk_life_expectancy()


def load_standard_data(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = get_entity(key)
    return interface.get_measure(entity, key.measure, location)


def load_metadata(key: str, location: str):
    key = EntityKey(key)
    entity = get_entity(key)
    metadata = entity[key.measure]
    if hasattr(metadata, 'to_dict'):
        metadata = metadata.to_dict()
    return metadata


def load_acmr(key: str, location: str) -> pd.DataFrame:
    return _transform_raw_data(location, paths.RAW_ACMR_DATA_PATH, True)


def load_prevalence(key: str, location: str) -> pd.DataFrame:
    base_prevalence = _transform_raw_data(location, paths.RAW_PREVALENCE_DATA_PATH, True)
    prevalence_ratio = 1
    if key == project_globals.BREAST_CANCER.LCIS_PREVALENCE:
        prevalence_ratio = _get_prevalence_ratio(project_globals.BREAST_CANCER.LCIS_PREVALENCE_RATIO, location)
    elif key == project_globals.BREAST_CANCER.DCIS_PREVALENCE:
        prevalence_ratio = _get_prevalence_ratio(project_globals.BREAST_CANCER.DCIS_PREVALENCE_RATIO, location)
    return base_prevalence * prevalence_ratio


def load_incidence_rate(key: str, location: str):
    if key == project_globals.BREAST_CANCER.INCIDENCE_RATE:
        incidence_rate = _transform_raw_data(location, paths.RAW_INCIDENCE_RATE_DATA_PATH, False)
    else:
        base_prevalence = load_prevalence(project_globals.BREAST_CANCER.PREVALENCE, location)
        if key == project_globals.BREAST_CANCER.LCIS_INCIDENCE_RATE:
            prevalence_ratio = _get_prevalence_ratio(project_globals.BREAST_CANCER.LCIS_PREVALENCE_RATIO, location)
            duration = LCIS_DURATION
        else:
            prevalence_ratio = _get_prevalence_ratio(project_globals.BREAST_CANCER.DCIS_PREVALENCE_RATIO, location)
            duration = DCIS_DURATION
        incidence_rate = base_prevalence * prevalence_ratio / duration
    return incidence_rate


def load_breast_cancer_transition_rate(key: str, location: str):
    return (
            get_data(project_globals.BREAST_CANCER.INCIDENCE_RATE, location)
            / (get_data(project_globals.BREAST_CANCER.LCIS_PREVALENCE, location)
               + get_data(project_globals.BREAST_CANCER.DCIS_PREVALENCE, location))
    )


def load_disability_weight(key: str, location: str):
    if key == project_globals.BREAST_CANCER.DISABILITY_WEIGHT:
        # Get breast cancer prevalence by location
        prevalence_data = _transform_raw_data_granular(paths.RAW_PREVALENCE_DATA_PATH, True)
        location_weighted_disability_weight = 0

        for swissre_location, location_weight in project_globals.SWISSRE_LOCATION_WEIGHTS.items():
            prevalence_disability_weight = 0
            breast_cancer_prevalence = prevalence_data[swissre_location]
            total_sequela_prevalence = 0
            for sequela in causes.breast_cancer.sequelae:
                # Get prevalence and disability weight for location and sequela
                prevalence = interface.get_measure(sequela, 'prevalence', swissre_location)
                disability_weight = interface.get_measure(sequela, 'disability_weight', swissre_location)
                # Apply prevalence weight
                prevalence_disability_weight += prevalence * disability_weight
                total_sequela_prevalence += prevalence

            # Calculate disability weight and apply location weight
            disability_weight = prevalence_disability_weight / total_sequela_prevalence
            location_weighted_disability_weight += disability_weight * location_weight

        disability_weight = location_weighted_disability_weight / sum(project_globals.SWISSRE_LOCATION_WEIGHTS.values())
    else:
        # LCIS and DCIS cause no disability
        disability_weight = 0

    return disability_weight


def load_emr(key: str, location: str):
    return (
            get_data(project_globals.BREAST_CANCER.CSMR, location)
            / get_data(project_globals.BREAST_CANCER.PREVALENCE, location)
    )


def load_csmr(key: str, location: str):
    return _transform_raw_data(location, paths.RAW_MORTALITY_DATA_PATH, False)


def _get_prevalence_ratio(key: str, location: str) -> float:
    random.seed(get_hash(f'{location}_{key}'))
    random_values = np.random.random(1000)
    return PREVALENCE_RATIOS[key].ppf(random_values).tolist()


def _transform_raw_data(location: str, data_path: Path, is_log_data: bool) -> pd.DataFrame:
    processed_data = _transform_raw_data_preliminary(data_path, is_log_data)
    processed_data['location'] = location

    # Weight the covered provinces
    processed_data['value'] = (sum(processed_data[province] * weight for province, weight
                                   in project_globals.SWISSRE_LOCATION_WEIGHTS.items())
                               / sum(project_globals.SWISSRE_LOCATION_WEIGHTS.values()))

    processed_data = (
        processed_data
        # Remove province columns
        .drop([province for province in project_globals.SWISSRE_LOCATION_WEIGHTS.keys()], axis=1)
        # Set index to final columns and unstack with draws as columns
        .reset_index()
        .set_index(ARTIFACT_INDEX_COLUMNS + ['draw'])
        .unstack()
    )

    # Simplify column index and rename draw columns
    processed_data.columns = [c[1] for c in processed_data.columns]
    processed_data = processed_data.rename(columns={col: f'draw_{col}' for col in processed_data.columns})
    return processed_data


def _transform_raw_data_granular(data_path: Path, is_log_data: bool = False) -> Dict[str, pd.DataFrame]:
    processed_data = _transform_raw_data_preliminary(data_path, is_log_data)
    processed_data_by_location = {}
    for swissre_location in project_globals.SWISSRE_LOCATION_WEIGHTS:
        location_data = (
            processed_data[swissre_location]
            .to_frame()
            .unstack()
        )

        # Simplify column index and rename draw columns
        location_data.columns = [c[1] for c in location_data.columns]
        location_data = location_data.rename(columns={col: f'draw_{col}' for col in location_data.columns})
        processed_data_by_location[swissre_location] = location_data
    return processed_data_by_location


def _transform_raw_data_preliminary(data_path: Path, is_log_data: bool = False) -> pd.DataFrame:
    """Transforms data to a form with draws in the index and raw locations as columns"""
    raw_data: pd.DataFrame = pd.read_hdf(data_path)
    age_bins = gbd.get_age_bins().set_index('age_group_name')

    processed_data = raw_data[raw_data['location_id'].isin(project_globals.SWISSRE_LOCATION_WEIGHTS)]
    processed_data = (
        processed_data
        .set_index('age_group_id')
        .join(age_bins, how='left')
        .reset_index()
        .rename(columns={
            'age_group_years_start': 'age_start',
            'age_group_years_end': 'age_end',
            'year_id': 'year_start',
            'sex_id': 'sex',
            'location_id': 'location',
        })
    )

    # Add year end column
    processed_data['year_end'] = processed_data['year_start'] + 1

    # Drop unneeded columns
    processed_data = processed_data[ARTIFACT_INDEX_COLUMNS + ['draw', 'noised_forecast']]

    # Make draw column numeric
    processed_data['draw'] = pd.to_numeric(processed_data['draw'])

    # Set index and unstack data with locations as columns
    processed_data = (
        processed_data
        .set_index(ARTIFACT_INDEX_COLUMNS + ['draw'])
        .unstack(level=0)
    )

    # Simplify column index and add back location column
    processed_data.columns = [c[1] for c in processed_data.columns]
    return processed_data


def get_entity(key: str):
    # Map of entity types to their gbd mappings.
    type_map = {
        'cause': causes,
        'covariate': covariates,
        'risk_factor': risk_factors,
        'alternative_risk_factor': alternative_risk_factors
    }
    key = EntityKey(key)
    return type_map[key.type][key.name]
