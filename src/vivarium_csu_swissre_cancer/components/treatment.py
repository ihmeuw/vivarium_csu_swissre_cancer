"""Treatment model."""
import numpy as np
import pandas as pd
import typing

from vivarium_csu_swissre_cancer import globals as project_globals
from vivarium_csu_swissre_cancer.utilities import get_triangular_dist_random_variable


if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population import SimulantData


# Columns
AGE = 'age'
SEX = 'sex'
TREATMENT_PROPENSITY = 'treatment_propensity'


class TreatmentEffect:
    """Manages treatment."""

    @property
    def name(self) -> str:
        """The name of this component."""
        return 'treatment_effect'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        self.clock = builder.time.clock()
        draw = builder.configuration.input_data.input_draw_number

        self.efficacy = {
            project_globals.POSITIVE_LCIS_STATE_NAME: (
                np.exp(project_globals.TREATMENT.LOG_LCIS_EFFICACY.get_random_variable(draw))
            ),
            project_globals.POSITIVE_DCIS_STATE_NAME: project_globals.TREATMENT.DCIS_EFFICACY.get_random_variable(draw),
        }

        self.coverage = {
            project_globals.POSITIVE_LCIS_STATE_NAME: get_triangular_dist_random_variable(
                *project_globals.TREATMENT.LCIS_COVERAGE_PARAMS, 'lcis_treatment_coverage', draw
            ),
            project_globals.POSITIVE_DCIS_STATE_NAME: get_triangular_dist_random_variable(
                *project_globals.TREATMENT.DCIS_COVERAGE_PARAMS, 'dcis_treatment_coverage', draw
            )
        }

        required_columns = [project_globals.SCREENING_RESULT_MODEL_NAME]
        columns_created = [TREATMENT_PROPENSITY]
        builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=columns_created)

        self.population_view = builder.population.get_view(required_columns)

        builder.value.register_value_modifier(
            'lobular_carcinoma_in_situ_to_breast_cancer.transition_rate',
            modifier=lambda index, target: self.treat(index, target, project_globals.POSITIVE_LCIS_STATE_NAME),
            # TODO add intermediate pipeline between screening result and treatment to allow for less than full
            #  treatment coverage
            requires_columns=[project_globals.SCREENING_RESULT_MODEL_NAME, TREATMENT_PROPENSITY]
        )

        builder.value.register_value_modifier(
            'ductal_carcinoma_in_situ_to_breast_cancer.transition_rate',
            modifier=lambda index, target: self.treat(index, target, project_globals.POSITIVE_DCIS_STATE_NAME),
            # TODO add intermediate pipeline between screening result and treatment to allow for less than full
            #  treatment coverage
            requires_columns=[project_globals.SCREENING_RESULT_MODEL_NAME, TREATMENT_PROPENSITY]
        )

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        propensity = pd.Series(np.random.uniform(size=len(pop_data.index)), index=pop_data.index)
        self.population_view.update(propensity)

    def treat(self, index, target, state_name):
        pop = self.population_view.get(index)
        would_be_treated_if_positive = pop.loc[:, TREATMENT_PROPENSITY] < self.coverage[state_name]
        is_positive = pop.loc[:, project_globals.SCREENING_RESULT_MODEL_NAME] == state_name
        is_treated = would_be_treated_if_positive & is_positive
        return target * (1 - self.efficacy[state_name] * is_treated)
