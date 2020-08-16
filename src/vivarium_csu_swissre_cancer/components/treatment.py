"""Treatment model."""
import typing

from vivarium_csu_swissre_cancer import globals as project_globals


if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder


# Columns
AGE = 'age'
SEX = 'sex'


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
            project_globals.POSITIVE_LCIS_STATE_NAME: project_globals.LCIS_TREATMENT_EFFICACY.get_random_variable(draw),
            project_globals.POSITIVE_DCIS_STATE_NAME: project_globals.DCIS_TREATMENT_EFFICACY.get_random_variable(draw),
        }

        required_columns = [project_globals.SCREENING_RESULT_MODEL_NAME]
        self.population_view = builder.population.get_view(required_columns)

        builder.value.register_value_modifier(
            'lobular_carcinoma_in_situ_to_breast_cancer.transition_rate',
            modifier=lambda index, target: self.treat(index, target, project_globals.POSITIVE_LCIS_STATE_NAME),
            # TODO add intermediate pipeline between screening result and treatment to allow for less than full
            #  treatment coverage
            requires_columns=[project_globals.SCREENING_RESULT_MODEL_NAME]
        )

        builder.value.register_value_modifier(
            'ductal_carcinoma_in_situ_to_breast_cancer.transition_rate',
            modifier=lambda index, target: self.treat(index, target, project_globals.POSITIVE_DCIS_STATE_NAME),
            # TODO add intermediate pipeline between screening result and treatment to allow for less than full
            #  treatment coverage
            requires_columns=[project_globals.SCREENING_RESULT_MODEL_NAME]
        )

    def treat(self, index, target, state_name):
        pop = self.population_view.get(index)
        screening_result = pop.loc[:, project_globals.SCREENING_RESULT_MODEL_NAME]
        lcis_positive_mask = screening_result == state_name
        return target * (1 - self.efficacy[state_name] * lcis_positive_mask)
