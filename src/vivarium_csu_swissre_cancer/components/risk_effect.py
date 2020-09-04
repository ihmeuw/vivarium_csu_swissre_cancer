import numpy as np
import pandas as pd
import typing

from vivarium.framework.randomness import RandomnessStream
from vivarium_public_health.risks.data_transformations import (generate_relative_risk_from_distribution,
                                                               get_distribution_type,
                                                               get_exposure_data,
                                                               get_relative_risk_data,
                                                               pivot_categorical,
                                                               rebin_relative_risk_data,
                                                               validate_relative_risk_data_source)
from vivarium_public_health.risks.effect import RiskEffect
from vivarium_public_health.utilities import EntityString, TargetString

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder


class LogNormalRiskEffect(RiskEffect):

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        self.randomness = builder.randomness.get_stream(
            f'effect_of_{self.risk.name}_on_{self.target.name}.{self.target.measure}'
        )

        relative_risk_data = self.load_relative_risk_data(builder)
        self.relative_risk = builder.lookup.build_table(relative_risk_data, key_columns=['sex'],
                                                        parameter_columns=['age', 'year'])
        population_attributable_fraction_data = self.load_population_attributable_fraction_data(builder)
        self.population_attributable_fraction = builder.lookup.build_table(population_attributable_fraction_data,
                                                                           key_columns=['sex'],
                                                                           parameter_columns=['age', 'year'])
        self.exposure_effect = self.load_exposure_effect(builder)

        builder.value.register_value_modifier(f'{self.target.name}.{self.target.measure}',
                                              modifier=self.adjust_target,
                                              requires_values=[f'{self.risk.name}.exposure'],
                                              requires_columns=['age', 'sex'])
        builder.value.register_value_modifier(f'{self.target.name}.{self.target.measure}.paf',
                                              modifier=self.population_attributable_fraction,
                                              requires_columns=['age', 'sex'])

    def load_relative_risk_data(self, builder):
        return get_relative_risk_data(builder, self.risk, self.target, self.randomness)

    def load_population_attributable_fraction_data(self, builder: 'Builder'):
        exposure_source = builder.configuration[f'{self.risk.name}']['exposure']
        rr_source_type = validate_relative_risk_data_source(builder, self.risk, self.target)

        if exposure_source == 'data' and rr_source_type == 'data' and self.risk.type == 'risk_factor':
            paf_data = builder.data.load(f'{self.risk}.population_attributable_fraction')
            correct_target = ((paf_data['affected_entity'] == self.target.name)
                              & (paf_data['affected_measure'] == self.target.measure))
            paf_data = (paf_data[correct_target]
                        .drop(['affected_entity', 'affected_measure'], 'columns'))
        else:
            key_cols = ['sex', 'age_start', 'age_end', 'year_start', 'year_end']
            exposure_data = get_exposure_data(builder, self.risk).set_index(key_cols)
            relative_risk_data = get_relative_risk_data(builder, self.risk, self.target,
                                                        self.randomness).set_index(key_cols)
            mean_rr = (exposure_data * relative_risk_data).sum(axis=1)
            paf_data = ((mean_rr - 1)/mean_rr).reset_index().rename(columns={0: 'value'})
        return paf_data


def get_relative_risk_data(builder, risk: EntityString, target: TargetString, randomness: RandomnessStream):
    source_type = validate_relative_risk_data_source(builder, risk, target)
    relative_risk_data = load_relative_risk_data(builder, risk, target, source_type, randomness)
    relative_risk_data = rebin_relative_risk_data(builder, risk, relative_risk_data)

    if get_distribution_type(builder, risk) in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous']:
        relative_risk_data = pivot_categorical(relative_risk_data)

    else:
        relative_risk_data = relative_risk_data.drop(['parameter'], 'columns')

    return relative_risk_data


def load_relative_risk_data(builder, risk: EntityString, target: TargetString, source_type: str,
                            randomness: RandomnessStream):
    relative_risk_source = builder.configuration[f'effect_of_{risk.name}_on_{target.name}'][target.measure]

    if source_type == 'data':
        relative_risk_data = builder.data.load(f'{risk}.relative_risk')
        correct_target = ((relative_risk_data['affected_entity'] == target.name)
                          & (relative_risk_data['affected_measure'] == target.measure))
        relative_risk_data = (relative_risk_data[correct_target]
                              .drop(['affected_entity', 'affected_measure'], 'columns'))

    elif source_type == 'relative risk value':
        relative_risk_data = _make_relative_risk_data(builder, float(relative_risk_source['relative_risk']))

    else:  # distribution
        parameters = {k: v for k, v in relative_risk_source.items() if v is not None}
        random_state = np.random.RandomState(randomness.get_seed())
        cat1_value = generate_relative_risk_from_distribution(random_state, parameters)
        relative_risk_data = _make_relative_risk_data(builder, cat1_value)

    return relative_risk_data


def _make_relative_risk_data(builder, cat1_value: float) -> pd.DataFrame:
    cat1 = builder.data.load('population.demographic_dimensions')
    cat1['parameter'] = 'cat1'
    cat1['value'] = cat1_value
    cat2 = cat1.copy()
    cat2['parameter'] = 'cat2'
    cat2['value'] = 1
    return pd.concat([cat1, cat2], ignore_index=True)
