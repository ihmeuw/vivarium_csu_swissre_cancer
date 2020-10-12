"""Treatment model."""
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Dict

from vivarium_csu_swissre_breast_cancer import globals as project_globals
from vivarium_csu_swissre_breast_cancer.utilities import get_triangular_dist_random_variable


if TYPE_CHECKING:
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
        self.randomness = builder.randomness.get_stream(self.name)

        draw = builder.configuration.input_data.input_draw_number

        self.efficacy = {
            project_globals.POSITIVE_LCIS_STATE_NAME: (
                np.exp(project_globals.TREATMENT.LOG_LCIS_EFFICACY.get_random_variable(draw))
            ),
            project_globals.POSITIVE_DCIS_STATE_NAME: project_globals.TREATMENT.DCIS_EFFICACY.get_random_variable(draw),
        }

        self.coverage = get_treatment_coverage(draw)

        required_columns = [project_globals.SCREENING_RESULT_MODEL_NAME]
        created_columns = [TREATMENT_PROPENSITY]
        builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=created_columns)

        self.population_view = builder.population.get_view(required_columns + created_columns)

        builder.value.register_value_modifier(
            'lobular_carcinoma_in_situ_to_breast_cancer.transition_rate',
            modifier=lambda index, target: self.treat(index, target, project_globals.POSITIVE_LCIS_STATE_NAME),
            requires_columns=[project_globals.SCREENING_RESULT_MODEL_NAME, TREATMENT_PROPENSITY]
        )

        builder.value.register_value_modifier(
            'ductal_carcinoma_in_situ_to_breast_cancer.transition_rate',
            modifier=lambda index, target: self.treat(index, target, project_globals.POSITIVE_DCIS_STATE_NAME),
            requires_columns=[project_globals.SCREENING_RESULT_MODEL_NAME, TREATMENT_PROPENSITY]
        )

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        propensity = pd.Series(self.randomness.get_draw(pop_data.index), name=TREATMENT_PROPENSITY)
        self.population_view.update(propensity)

    def treat(self, index, target, state_name):
        pop = self.population_view.get(index)
        is_treated = is_treated_in_state(state_name, self.coverage[state_name], pop.loc[:, TREATMENT_PROPENSITY],
                                         pop.loc[:, project_globals.SCREENING_RESULT_MODEL_NAME])

        return target * (1 - self.efficacy[state_name] * is_treated)


def get_treatment_coverage(draw: int) -> Dict[str, float]:
    return {
        project_globals.POSITIVE_LCIS_STATE_NAME: get_triangular_dist_random_variable(
            *project_globals.TREATMENT.LCIS_COVERAGE_PARAMS, 'lcis_treatment_coverage', draw
        ),
        project_globals.POSITIVE_DCIS_STATE_NAME: get_triangular_dist_random_variable(
            *project_globals.TREATMENT.DCIS_COVERAGE_PARAMS, 'dcis_treatment_coverage', draw
        )
    }


def is_treated_in_state(state_name: str, coverage_threshold: float, treatment_propensity: pd.Series,
                        state: pd.Series) -> pd.Series:
    return (treatment_propensity < coverage_threshold) & (state == state_name)
