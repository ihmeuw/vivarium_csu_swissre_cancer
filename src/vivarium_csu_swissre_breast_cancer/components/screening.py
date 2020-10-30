"""Healthcare utilization and treatment model."""
import typing

import pandas as pd

from vivarium_csu_swissre_breast_cancer import globals as project_globals


if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.event import Event
    from vivarium.framework.population import SimulantData


# Columns
AGE = 'age'
SEX = 'sex'


class ScreeningAlgorithm:
    """Manages screening."""

    configuration_defaults = {
        'screening_algorithm': {
            'scenario': project_globals.SCENARIOS.baseline
        }
    }

    @property
    def name(self) -> str:
        """The name of this component."""
        return 'screening_algorithm'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        """Select an algorithm based on the current scenario

        Parameters
        ----------
        builder
            The simulation builder object.

        """
        self.scenario = builder.configuration.screening_algorithm.scenario
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.randomness = builder.randomness.get_stream(self.name)

        draw = builder.configuration.input_data.input_draw_number
        self.screening_parameters = {parameter.name: parameter.get_random_variable(draw)
                                     for parameter in project_globals.SCREENING}
        self.screening_parameters[project_globals.P_SYMPTOMS] = self.step_size() / project_globals.MEAN_SYMPTOMS

        self.family_history = builder.value.get_value('family_history.exposure')

        required_columns = [AGE, SEX, project_globals.BREAST_CANCER_MODEL_NAME, 'family_history_propensity']
        columns_created = [
            project_globals.SCREENING_RESULT_MODEL_NAME,
            project_globals.ATTENDED_LAST_SCREENING,
            project_globals.PREVIOUS_SCREENING_DATE,
            project_globals.NEXT_SCREENING_DATE,
        ]
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created,
                                                 requires_columns=[col for col in required_columns
                                                                   if col != project_globals.BREAST_CANCER_MODEL_NAME])
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
        screening_result = pd.Series(project_globals.NEGATIVE_STATE_NAME,
                                     index=pop.index,
                                     name=project_globals.SCREENING_RESULT_MODEL_NAME)
        family_history = self.family_history(pop.index) == 'cat1'

        female_under_30 = (pop.loc[:, SEX] == 'Female') & (pop.loc[:, AGE] < 30)
        female_under_70 = (pop.loc[:, SEX] == 'Female') & (pop.loc[:, AGE] < 70)

        # Get beginning time for screening of all individuals
        #  - never for men and women over 70
        #  - beginning of sim for women between 30 & 70
        #  - 30th birthday for women younger than 30
        screening_start = pd.Series(pd.NaT, index=pop.index)
        screening_start.loc[female_under_70] = self.clock()
        screening_start.loc[female_under_30] = (
                screening_start.loc[female_under_30] + pd.to_timedelta(30 - pop.loc[female_under_30, AGE], unit='Y')
        )

        # Draw a duration between screenings to use for scheduling the first screening
        time_between_screenings = (self._schedule_screening(screening_start, screening_result, family_history)
                                   - screening_start)

        # Determine how far along between screenings we are the time screening starts
        progress_to_next_screening = self.randomness.get_draw(pop.index, 'progress_to_next_screening')

        # Get previous screening date for use in calculating next screening date
        previous_screening = pd.Series(screening_start - progress_to_next_screening * time_between_screenings,
                                       name=project_globals.PREVIOUS_SCREENING_DATE)
        next_screening = pd.Series(previous_screening + time_between_screenings,
                                   name=project_globals.NEXT_SCREENING_DATE)
        # Remove the "appointment" used to determine the first appointment after turning 30
        previous_screening.loc[female_under_30] = pd.NaT

        attended_previous = pd.Series(self.randomness.get_draw(pop.index, 'attended_previous')
                                      < self.screening_parameters[project_globals.SCREENING.BASE_ATTENDANCE.name],
                                      name=project_globals.ATTENDED_LAST_SCREENING)

        self.population_view.update(
            pd.concat([screening_result, previous_screening, next_screening, attended_previous], axis=1)
        )

    def on_time_step(self, event: 'Event'):
        """Determine if someone will go for a screening"""
        # Get all simulants with a screening scheduled during this timestep
        pop = self.population_view.get(event.index, query='alive == "alive"')

        # Get all simulants who have breast cancer and found a lump on this timestep
        found_lump = self.is_symptomatic(pop)

        # Set next screening date for simulants who found a lump to today
        next_screening_date = pop.loc[:, project_globals.NEXT_SCREENING_DATE]
        next_screening_date.loc[found_lump] = self.clock()

        # Get all simulants with a scheduled screening
        screening_scheduled = (next_screening_date <= self.clock())

        # Get probability of attending the next screening for scheduled simulants
        p_attends_screening = self._get_screening_attendance_probability(pop)

        # Get all simulants who actually attended their screening
        attends_screening: pd.Series = (screening_scheduled &
                                        (found_lump | (self.randomness.get_draw(pop.index, 'attendance')
                                                       < p_attends_screening)))

        # Update attended previous screening column
        attended_last_screening = pop.loc[:, project_globals.ATTENDED_LAST_SCREENING].copy()
        attended_last_screening.loc[screening_scheduled] = attends_screening.loc[screening_scheduled]
        attended_last_screening = attended_last_screening.astype(bool)

        # Screening results for everyone
        screening_result = pop.loc[:, project_globals.SCREENING_RESULT_MODEL_NAME].copy()
        screening_result[attends_screening] = self._do_screening(pop.loc[attends_screening, :])

        # Update previous screening column
        previous_screening = pop.loc[:, project_globals.PREVIOUS_SCREENING_DATE].copy()
        previous_screening.loc[screening_scheduled] = pop.loc[screening_scheduled, project_globals.NEXT_SCREENING_DATE]

        # Next scheduled screening for everyone
        next_screening = pop.loc[:, project_globals.NEXT_SCREENING_DATE].copy()
        next_screening.loc[screening_scheduled] = self._schedule_screening(
            pop.loc[screening_scheduled, project_globals.NEXT_SCREENING_DATE],
            screening_result.loc[screening_scheduled],
            self.family_history(pop.index) == 'cat1'
        )

        # Update values
        self.population_view.update(
            pd.concat([screening_result, previous_screening, next_screening, attended_last_screening], axis=1)
        )

    def _get_screening_attendance_probability(self, pop: pd.DataFrame) -> pd.Series:
        # Get base probability of screening attendance based on the current date
        screening_start_attended_previous = self.screening_parameters[
            project_globals.SCREENING.START_ATTENDED_PREV_ATTENDANCE.name
        ]
        screening_start_not_attended_previous = self.screening_parameters[
            project_globals.SCREENING.START_NOT_ATTENDED_PREV_ATTENDANCE.name
        ]
        screening_end_attended_previous = self.screening_parameters[
            project_globals.SCREENING.END_ATTENDED_PREV_ATTENDANCE.name
        ]
        screening_end_not_attended_previous = self.screening_parameters[
            project_globals.SCREENING.END_NOT_ATTENDED_PREV_ATTENDANCE.name
        ]
        if self.scenario == project_globals.SCENARIOS.baseline:
            conditional_probabilities = {
                True: screening_start_attended_previous,
                False: screening_start_not_attended_previous,
            }
        else:
            if self.clock() < project_globals.RAMP_UP_START:
                conditional_probabilities = {
                    True: screening_start_attended_previous,
                    False: screening_start_not_attended_previous,
                }
            elif self.clock() < project_globals.RAMP_UP_END:
                elapsed_time = self.clock() - project_globals.RAMP_UP_START
                progress_to_ramp_up_end = elapsed_time / (project_globals.RAMP_UP_END - project_globals.RAMP_UP_START)
                attended_prev_ramp_up = screening_end_attended_previous - screening_start_attended_previous
                not_attended_prev_ramp_up = screening_end_not_attended_previous - screening_start_not_attended_previous

                conditional_probabilities = {
                    True: attended_prev_ramp_up * progress_to_ramp_up_end + screening_start_attended_previous,
                    False: not_attended_prev_ramp_up * progress_to_ramp_up_end + screening_start_not_attended_previous,
                }
            else:
                conditional_probabilities = {
                    True: screening_end_attended_previous,
                    False: screening_end_not_attended_previous,
                }

        return pop.loc[:, project_globals.ATTENDED_LAST_SCREENING].apply(lambda x: conditional_probabilities[x])

    def _do_screening(self, pop: pd.Series) -> pd.Series:
        """Perform screening for all simulants who attended their screening"""
        screened = (30 <= pop.loc[:, AGE]) & (pop.loc[:, AGE] < 70) & (pop.loc[:, SEX] == 'Female')
        family_history = self.family_history(pop.index) == 'cat1'
        in_remission = pop.loc[:, project_globals.BREAST_CANCER_MODEL_NAME] == project_globals.RECOVERED_STATE_NAME
        has_lcis_dcis = pop.loc[:, project_globals.SCREENING_RESULT_MODEL_NAME].isin([
            project_globals.POSITIVE_LCIS_STATE_NAME,
            project_globals.POSITIVE_DCIS_STATE_NAME
        ])

        found_lump = self.is_symptomatic(pop)
        screened_remission = screened & in_remission
        mri = screened & family_history & ~found_lump & ~in_remission
        ultrasound = (screened & ~family_history & has_lcis_dcis & ~found_lump & ~in_remission
                      & (30 <= pop.loc[:, AGE]) & (pop.loc[:, AGE] < 45))
        mammogram_ultrasound = (screened & ~family_history & has_lcis_dcis & ~found_lump & ~in_remission
                                & (45 <= pop.loc[:, AGE]) & (pop.loc[:, AGE] < 70))
        mammogram = screened & ~family_history & ~has_lcis_dcis & ~found_lump & ~in_remission

        # Get sensitivity values for all individuals
        # TODO address different sensitivity values for tests of different conditions
        sensitivity = pd.Series(0.0, index=pop.index)
        sensitivity.loc[found_lump] = self.screening_parameters[project_globals.SCREENING.FOUND_LUMP_SENSITIVITY.name]
        sensitivity.loc[screened_remission] = self.screening_parameters[
            project_globals.SCREENING.REMISSION_SENSITIVITY.name
        ]
        sensitivity.loc[mri] = self.screening_parameters[project_globals.SCREENING.MRI_SENSITIVITY.name]
        sensitivity.loc[ultrasound] = self.screening_parameters[project_globals.SCREENING.ULTRASOUND_SENSITIVITY.name]
        sensitivity.loc[mammogram_ultrasound] = self.screening_parameters[
            project_globals.SCREENING.MAMMOGRAM_ULTRASOUND_SENSITIVITY.name
        ]
        sensitivity.loc[mammogram] = self.screening_parameters[project_globals.SCREENING.MAMMOGRAM_SENSITIVITY.name]

        # Perform screening on those who attended screening
        accurate_results = self.randomness.get_draw(pop.index, 'sensitivity') < sensitivity

        # Screening results for everyone who was screened
        screening_result = pop.loc[:, project_globals.SCREENING_RESULT_MODEL_NAME].copy()
        screening_result.loc[accurate_results] = (
            pop.loc[accurate_results, project_globals.BREAST_CANCER_MODEL_NAME]
            .apply(project_globals.get_screened_state)
        )
        return screening_result

    def _schedule_screening(self, previous_screening: pd.Series, screening_result: pd.Series,
                            family_history: pd.Series) -> pd.Series:
        """Schedules follow up visits."""
        has_had_lcis_dcis = (screening_result != project_globals.NEGATIVE_STATE_NAME)
        annual_screening = has_had_lcis_dcis | family_history
        draw = self.randomness.get_draw(previous_screening.index, 'schedule_next')

        time_to_next_screening = pd.Series(None, previous_screening.index)
        time_to_next_screening.loc[annual_screening] = pd.to_timedelta(
            pd.Series(project_globals.DAYS_UNTIL_NEXT_ANNUAL.ppf(draw), index=draw.index), unit='day'
        ).loc[annual_screening]
        time_to_next_screening.loc[~annual_screening] = pd.to_timedelta(
            pd.Series(project_globals.DAYS_UNTIL_NEXT_BIENNIAL.ppf(draw), index=draw.index), unit='day'
        ).loc[~annual_screening]

        return previous_screening + time_to_next_screening.astype('timedelta64[ns]')

    def is_symptomatic(self, pop: pd.DataFrame):
        return ((pop.loc[:, project_globals.BREAST_CANCER_MODEL_NAME] == project_globals.BREAST_CANCER_STATE_NAME)
                & (self.randomness.get_draw(pop.index, 'symptomatic_presentation')
                   < self.screening_parameters[project_globals.P_SYMPTOMS]))
