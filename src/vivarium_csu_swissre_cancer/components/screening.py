"""Healthcare utilization and treatment model."""
import typing
from typing import Dict, NamedTuple, Tuple

import pandas as pd

from vivarium_csu_swissre_cancer import globals as project_globals
from vivarium_csu_swissre_cancer.utilities import TruncnormDist


if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.event import Event
    from vivarium.framework.population import SimulantData


SCREENING_SCENARIO = 'screening_algorithm'

# Columns
AGE = 'age'
SEX = 'sex'


class __Scenarios(NamedTuple):
    baseline: str = 'baseline'
    # TODO add scenarios


SCENARIOS = __Scenarios()


class ScreeningAlgorithm:
    """Manages screening."""

    configuration_defaults = {
        SCREENING_SCENARIO: {
            'scenario': SCENARIOS.baseline
        }
    }

    @property
    def name(self) -> str:
        """The name of this component."""
        return SCREENING_SCENARIO

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        """Select an algorithm based on the current scenario

        Parameters
        ----------
        builder
            The simulation builder object.

        """
        self.draw = builder.configuration.input_data.input_draw_number
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.randomness = builder.randomness.get_stream(self.name)

        self.screening_parameters = {parameter.name: parameter.sample_screening_parameter(self.draw)
                                     for parameter in project_globals.SCREENING}

        required_columns = [AGE, SEX, project_globals.BREAST_CANCER_MODEL_NAME]
        columns_created = [
            project_globals.SCREENING_RESULT,
            project_globals.ATTENDED_LAST_SCREENING,
            project_globals.NEXT_SCREENING_DATE,
        ]
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created,
                                                 requires_columns=required_columns)
        self.population_view = builder.population.get_view(required_columns + columns_created)

        builder.event.register_listener('time_step',
                                        self.on_time_step)

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        """Assign all simulants a next screening date. Also determine if they attended their previous screening"""

        pop = self.population_view.subview([
            SEX,
            AGE,
        ]).get(pop_data.index)

        # TODO need to initialize this value properly - waiting on guidance from RT
        screening_result = pd.Series(project_globals.BREAST_CANCER_SUSCEPTIBLE_STATE_NAME,
                                     index=pop.index,
                                     name=project_globals.SCREENING_RESULT)

        female_under_30 = (pop[SEX] == 'Female') & (pop[AGE] < 30)
        female_under_70 = (pop[SEX] == 'Female') & (pop[AGE] < 70)

        previous_screening = pd.Series(self.clock(), index=pop.index)
        previous_screening[female_under_30] = (
                previous_screening[female_under_30]
                + pd.to_timedelta(30 - pop.loc[female_under_30, AGE], unit='Y')
        )

        next_screening = pd.Series(pd.NaT, index=pop.index, name=project_globals.NEXT_SCREENING_DATE)
        next_screening[female_under_70] = self._schedule_screening(
            previous_screening[female_under_70],
            screening_result[female_under_70],
            is_init=True
        )

        attended_last_screening = (self.randomness.get_draw(pop.index, 'attended_previous')
                                   < self.screening_parameters[project_globals.SCREENING.BASE_PROBABILITY.name])
        attended_last_screening.name = project_globals.ATTENDED_LAST_SCREENING

        self.population_view.update(
            pd.concat([screening_result, next_screening, attended_last_screening], axis=1)
        )

    def on_time_step(self, event: 'Event'):
        """Determine if someone will go for a screening"""
        # Get all simulants with a screening scheduled during this timestep
        pop = self.population_view.get(event.index)
        screening_scheduled_mask = pop[project_globals.NEXT_SCREENING_DATE] < self.clock()
        screening_scheduled = pop[screening_scheduled_mask]

        # Get probability of attending the next screening for scheduled simulants
        p_attends_screening = self._get_screening_attendance_probability(screening_scheduled)

        # Get all simulants who actually attended their screening
        attended_this_screening = self.randomness.get_draw(screening_scheduled.index) < p_attends_screening
        attended_last_screening = pd.Series(pop[project_globals.ATTENDED_LAST_SCREENING])
        attended_last_screening[screening_scheduled_mask] = attended_this_screening
        attended_last_screening = attended_last_screening.astype(bool)

        # Screening results for everyone
        screening_result = pd.Series(pop[project_globals.SCREENING_RESULT])
        screening_result[screening_scheduled_mask][attended_this_screening] = self._do_screening(
            screening_scheduled[attended_this_screening]
        )

        # Next scheduled screening for everyone
        next_screening = pd.Series(pop[project_globals.NEXT_SCREENING_DATE])
        next_screening[screening_scheduled_mask] = self._schedule_screening(
            screening_scheduled[project_globals.NEXT_SCREENING_DATE],
            screening_result[screening_scheduled_mask]
        )

        # Update values
        self.population_view.update(
            pd.concat([screening_result, next_screening, attended_last_screening], axis=1)
        )

    def _get_screening_attendance_probability(self, pop):
        p_attends_screening = pd.Series(
            self.screening_parameters[project_globals.SCREENING.PROBABILITY_GIVEN_NOT_ADHERENT.name],
            index=pop.index
        )
        p_attends_screening[pop[project_globals.ATTENDED_LAST_SCREENING]] = (
            self.screening_parameters[project_globals.SCREENING.PROBABILITY_GIVEN_ADHERENT.name]
        )
        return p_attends_screening

    def _do_screening(self, pop: pd.Series) -> pd.Series:
        """Perform screening for all simulants who attended their screening"""

        # TODO update with family history information
        family_history = pd.Series(False, index=pop.index)
        has_dcis_lcis = pop[project_globals.SCREENING_RESULT].isin([project_globals.DCIS_STATE_NAME,
                                                                    project_globals.LCIS_STATE_NAME])
        mri = family_history & (30 <= pop[AGE]) & (pop[AGE] < 70)
        ultrasound = ~family_history & has_dcis_lcis & (30 <= pop[AGE]) & (pop[AGE] < 45)
        mammogram_ultrasound = (~family_history & has_dcis_lcis & (45 <= pop[AGE]) & (pop[AGE] < 70))
        mammogram = ~family_history & ~has_dcis_lcis & (30 <= pop[AGE]) & (pop[AGE] < 70)

        # Get sensitivity values for all individuals
        # TODO address different sensitivity values for tests of different conditions
        sensitivity = pd.Series(0.0, index=pop.index)
        sensitivity[mri] = self.screening_parameters[project_globals.SCREENING.MRI_SENSITIVITY.name]
        sensitivity[ultrasound] = self.screening_parameters[project_globals.SCREENING.ULTRASOUND_SENSITIVITY.name]
        sensitivity[mammogram_ultrasound] = self.screening_parameters[
            project_globals.SCREENING.MAMMOGRAM_ULTRASOUND_SENSITIVITY.name
        ]
        sensitivity[mammogram] = self.screening_parameters[project_globals.SCREENING.MAMMOGRAM_SENSITIVITY.name]

        # Get sensitivity and specificity values for all individuals
        # TODO address different specificity values for tests of different conditions
        specificity = pd.Series(1.0, index=pop.index)
        specificity[mri] = self.screening_parameters[project_globals.SCREENING.MRI_SPECIFICITY.name]
        specificity[ultrasound] = self.screening_parameters[project_globals.SCREENING.ULTRASOUND_SPECIFICITY.name]
        specificity[mammogram_ultrasound] = self.screening_parameters[
            project_globals.SCREENING.MAMMOGRAM_ULTRASOUND_SPECIFICITY.name
        ]
        specificity[mammogram] = self.screening_parameters[project_globals.SCREENING.MAMMOGRAM_SPECIFICITY.name]

        # Perform screening on those who attended screening
        new_true_positives = ((self.randomness.get_draw(pop.index) < sensitivity)
                              & (pop[project_globals.BREAST_CANCER_MODEL_NAME]
                                 != pop[project_globals.SCREENING_RESULT]))

        new_false_positives = ((self.randomness.get_draw(pop.index) >= sensitivity)
                               & (pop[project_globals.BREAST_CANCER_MODEL_NAME]
                                  != pop[project_globals.SCREENING_RESULT]))

        # Screening results for everyone who was screened
        screening_result = pd.Series(pop[project_globals.SCREENING_RESULT])
        screening_result[new_true_positives] = (
            pop[project_globals.BREAST_CANCER_MODEL_NAME][new_true_positives]
        )
        screening_result[new_false_positives] = project_globals.BREAST_CANCER_STATE_NAME
        return screening_result

    def _schedule_screening(self, previous_screening: pd.Series, screening_result: pd.Series,
                            is_init=False) -> pd.Series:
        """Schedules follow up visits."""
        has_dcis_lcis = screening_result.isin([project_globals.DCIS_STATE_NAME, project_globals.LCIS_STATE_NAME])

        draw = self.randomness.get_draw(previous_screening.index, 'schedule_next')
        time_to_next_screening = pd.Series(pd.NaT, previous_screening.index)
        if is_init:
            # For initialization use a uniform distribution up to the mean of the true distribution
            time_to_next_screening[has_dcis_lcis] = (
                pd.to_timedelta(draw * project_globals.DAYS_UNTIL_NEXT_ANNUAL.loc, unit='day')
            )
            time_to_next_screening[~has_dcis_lcis] = (
                pd.to_timedelta(draw * project_globals.DAYS_UNTIL_NEXT_BIENNIAL.loc, unit='day')
            )
        else:
            time_to_next_screening[has_dcis_lcis] = pd.to_timedelta(
                pd.Series(project_globals.DAYS_UNTIL_NEXT_ANNUAL.get_draw(draw[has_dcis_lcis])), unit='day'
            )
            time_to_next_screening[~has_dcis_lcis] = pd.to_timedelta(
                pd.Series(project_globals.DAYS_UNTIL_NEXT_BIENNIAL.get_draw(draw[~has_dcis_lcis])), unit='day'
            )

        return previous_screening + time_to_next_screening
