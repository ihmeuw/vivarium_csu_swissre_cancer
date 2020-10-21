import numpy as np
import pandas as pd
import typing

from vivarium.framework.randomness import RandomnessStream
from vivarium_public_health.risks.data_transformations import (generate_relative_risk_from_distribution,
                                                               get_exposure_data,
                                                               get_relative_risk_data,
                                                               pivot_categorical,
                                                               rebin_relative_risk_data)
from vivarium_public_health.risks.effect import RiskEffect
from vivarium_public_health.utilities import EntityString, TargetString

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder


class LogNormalRiskEffect(RiskEffect):

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        self.validate_config(builder)

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

    def validate_config(self, builder: 'Builder'):
        source_key = f'effect_of_{self.risk.name}_on_{self.target.name}'
        relative_risk_source = builder.configuration[source_key][self.target.measure]
        provided_keys = set(k for k, v in relative_risk_source.to_dict().items() if isinstance(v, (int, float)))
        if provided_keys != {"mean", "se"}:
            raise ValueError(f'The acceptable parameter options for specifying relative risk are: '
                             f'{{"mean", "se"}}. You provided {provided_keys} for {source_key}.')

    def load_relative_risk_data(self, builder: 'Builder'):
        rr_data = get_relative_risk_data(builder, self.risk, self.target)
        rr_data.loc[:, 'cat1'] = np.exp(rr_data.loc[:, 'cat1'])
        return rr_data

    def load_population_attributable_fraction_data(self, builder: 'Builder'):
        key_cols = ['sex', 'age_start', 'age_end', 'year_start', 'year_end']
        exposure_data = get_exposure_data(builder, self.risk).set_index(key_cols)
        relative_risk_data = self.load_relative_risk_data(builder).set_index(key_cols)
        mean_rr = (exposure_data * relative_risk_data).sum(axis=1)
        paf_data = ((mean_rr - 1) / mean_rr).reset_index().rename(columns={0: 'value'})
        return paf_data
