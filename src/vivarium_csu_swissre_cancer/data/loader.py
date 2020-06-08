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
from gbd_mapping import causes, risk_factors, covariates
import numpy as np
import pandas as pd
from vivarium.framework.artifact import EntityKey
from vivarium_gbd_access import gbd
from vivarium_inputs import interface, utilities, utility_data, globals as vi_globals
from vivarium_inputs.mapping_extension import alternative_risk_factors

from vivarium_csu_swissre_cancer import paths, globals as project_globals


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

        # TODO - add appropriate mappings
        # project_globals.DIARRHEA_PREVALENCE: load_standard_data,
        # project_globals.DIARRHEA_INCIDENCE_RATE: load_standard_data,
        # project_globals.DIARRHEA_REMISSION_RATE: load_standard_data,
        # project_globals.DIARRHEA_CAUSE_SPECIFIC_MORTALITY_RATE: load_standard_data,
        # project_globals.DIARRHEA_EXCESS_MORTALITY_RATE: load_standard_data,
        # project_globals.DIARRHEA_DISABILITY_WEIGHT: load_standard_data,
        # project_globals.DIARRHEA_RESTRICTIONS: load_metadata,
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


def load_acmr(key: str, location: str):
    raw_acmr_data: pd.DataFrame = pd.read_hdf(paths.RAW_ACMR_DATA_PATH)
    age_bins = gbd.get_age_bins().set_index('age_group_id')
    locations = gbd.get_location_ids().set_index('location_id')

    # Transform raw acmr data from log space to linear space
    raw_acmr_data['mean'] = np.exp(raw_acmr_data['mr'])

    acmr = (
        raw_acmr_data
        .reset_index()
        # Set index to match age_bins and join
        .set_index('age_group_id')
        .join(age_bins, how='left')
        .reset_index()
        # Set index to match location and join
        .set_index('location_id')
        .join(locations, how='left')
        .reset_index()
        .rename(columns={
            'age_group_years_start': 'age_start',
            'age_group_years_end': 'age_end',
            'year_id': 'year_start',
            'location_name': 'location',
        })
    )

    # Filter locations down to the regions covered by SwissRE
    swissre_locations_mask = acmr['location'].isin(project_globals.SWISSRE_LOCATION_WEIGHTS)
    acmr = acmr[swissre_locations_mask]

    # Add year end column and create sex column with strings rather than ids
    acmr['year_end'] = acmr['year_start'] + 1
    acmr['sex'] = acmr['sex_id'].apply(lambda x: 'Male' if x == 1 else 'Female')

    # Drop unneeded columns
    acmr = acmr.drop(
        ['age_group_id', 'age_group_name', 'location_id', 'mr', 'sex_id'], axis=1
    )

    # Make draw column numeric
    acmr['draw'] = pd.to_numeric(acmr['draw'])

    final_idx_columns = [
        'location',
        'sex',
        'age_start',
        'age_end',
        'year_start',
        'year_end',
        'draw',
    ]

    # Set index and unstack data with locations as columns
    acmr = (
        acmr
        .set_index(final_idx_columns)
        .unstack(level=0)
    )

    # Simplify column index and add back location column
    acmr.columns = [c[1] for c in acmr.columns]
    acmr['location'] = location

    # Weight the covered provinces
    acmr['value'] = (sum(acmr[province] * weight for province, weight
                         in project_globals.SWISSRE_LOCATION_WEIGHTS.items())
                     / sum(project_globals.SWISSRE_LOCATION_WEIGHTS.values()))

    acmr = (
        acmr
        # Remove province columns
        .drop([province for province in project_globals.SWISSRE_LOCATION_WEIGHTS.keys()], axis=1)
        # Set index to final columns and unstack with draws as columns
        .reset_index()
        .set_index(final_idx_columns)
        .unstack()
    )

    # Simplify column index and rename draw columns
    acmr.columns = [c[1] for c in acmr.columns]
    acmr = acmr.rename(columns={col: f'draw_{col}' for col in acmr.columns})
    return acmr


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
