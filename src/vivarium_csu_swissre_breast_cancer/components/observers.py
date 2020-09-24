from collections import Counter
import itertools
import typing
from typing import Dict, Iterable, List, Tuple, Union

import pandas as pd
from vivarium_public_health.metrics import (MortalityObserver as MortalityObserver_,
                                            DisabilityObserver as DisabilityObserver_)
from vivarium_public_health.metrics.utilities import (get_output_template, get_group_counts,
                                                      QueryString, to_years, get_person_time,
                                                      get_deaths, get_years_of_life_lost,
                                                      get_years_lived_with_disability, get_age_bins,
                                                      )

from vivarium_csu_swissre_breast_cancer import globals as project_globals, paths

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.event import Event
    from vivarium.framework.population import SimulantData


class ResultsStratifier:
    """Centralized component for handling results stratification.

    This should be used as a sub-component for observers.  The observers
    can then ask this component for population subgroups and labels during
    results production and have this component manage adjustments to the
    final column labels for the subgroups.

    """

    def __init__(self, observer_name: str, has_screening_state: bool = False):
        self.name = f'{observer_name}_results_stratifier'
        self.has_screening_state = has_screening_state

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        """Perform this component's setup."""
        # The only thing you should request here are resources necessary for results stratification.
        self.pipelines = {
            'family_history': builder.value.get_value('family_history.exposure'),
        }
        columns_required = [
            'age',
            'family_history_propensity',
        ]

        def get_age_range_function(age_cohort):
            birth_year_bounds = [2020 - int(year) for year in age_cohort.split('_to_')]
            return lambda: (
                (birth_year_bounds[1] <= self.population_values['age'])
                & (self.population_values['age'] < (birth_year_bounds[0]))
            )

        def get_screening_result_function(state_name):
            return lambda: self.population_values[project_globals.SCREENING_RESULT_MODEL_NAME] == state_name

        self.stratification_levels = {
            'age_cohort': {age_cohort: get_age_range_function(age_cohort)
                           for age_cohort in project_globals.AGE_COHORTS},
            'family_history': {
                'positive': lambda: self.pipeline_values['family_history'] == 'cat1',
                'negative': lambda: self.pipeline_values['family_history'] == 'cat2',
            },
        }

        if self.has_screening_state:
            columns_required.append(project_globals.SCREENING_RESULT_MODEL_NAME)

        self.population_view = builder.population.get_view(columns_required)
        self.pipeline_values = {pipeline: None for pipeline in self.pipelines}
        self.population_values = None
        self.stratification_groups = None

        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 requires_columns=columns_required,
                                                 requires_values=list(self.pipelines.keys()))

        builder.event.register_listener('time_step__cleanup', self.on_timestep_cleanup)

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        self.set_stratification_groups(pop_data.index)

    def on_timestep_cleanup(self, event: 'Event'):
        if self.has_screening_state:
            # Update screening result state
            self.population_values.loc[event.index, project_globals.SCREENING_RESULT_MODEL_NAME] = (
                self.population_view.get(event.index).loc[event.index, project_globals.SCREENING_RESULT_MODEL_NAME]
            )

    def get_all_stratifications(self) -> List[Tuple[Dict[str, str], ...]]:
        """
        Gets all stratification combinations. Returns a List of Stratifications. Each Stratification is represented as a
        Tuple of Stratification Levels. Each Stratification Level is represented as a Dictionary with keys 'metric' and
        'category'. 'metric' refers to the stratification level's name, and 'category' refers to the stratification
        category.

        If no stratification levels are defined, returns a List with a single empty Tuple
        """
        # Get list of lists of metric and category pairs for each metric
        groups = [[{'metric': metric, 'category': category} for category, category_mask in category_maps.items()]
                  for metric, category_maps in self.stratification_levels.items()]
        # Get product of all stratification combinations
        return list(itertools.product(*groups))

    # noinspection PyAttributeOutsideInit
    def set_stratification_groups(self, index: pd.Index):
        stratification_groups = pd.Series('', index=index)

        self.pipeline_values = {name: pipeline(index) for name, pipeline in self.pipelines.items()}
        self.population_values = self.population_view.get(index)

        all_stratifications = self.get_all_stratifications()
        for stratification in all_stratifications:
            stratification_group_name = '_'.join([f'{metric["metric"]}_{metric["category"]}'
                                                  for metric in stratification])
            mask = pd.Series(True, index=index)
            for metric in stratification:
                mask &= self.stratification_levels[metric['metric']][metric['category']]()
            stratification_groups.loc[mask] = stratification_group_name

        self.stratification_groups = stratification_groups

    @staticmethod
    def get_stratification_key(stratification: Iterable[Dict[str, str]]) -> str:
        return ('' if not stratification
                else '_'.join([f'{metric["metric"]}_{metric["category"]}' for metric in stratification]))

    def group(self, pop: pd.DataFrame) -> Iterable[Tuple[Tuple[str, ...], pd.DataFrame]]:
        """Takes the full population and yields stratified subgroups.

        Parameters
        ----------
        pop
            The population to stratify.

        Yields
        ------
            A tuple of stratification labels and the population subgroup
            corresponding to those labels.

        """
        stratification_group = self.stratification_groups.loc[pop.index]
        stratifications = self.get_all_stratifications()
        for stratification in stratifications:
            if self.has_screening_state:
                screening_result = self.population_view.get(pop.index)[project_globals.SCREENING_RESULT_MODEL_NAME]
                for screening_state_name in project_globals.SCREENING_MODEL_STATES:
                    stratification_key = self.get_stratification_key(stratification)
                    if pop.empty:
                        pop_in_group = pop
                    else:
                        pop_in_group = pop.loc[(stratification_group == stratification_key)
                                               & (screening_result == screening_state_name)]
                    yield (f'{stratification_key}_screening_result_{screening_state_name}',), pop_in_group
            else:
                stratification_key = self.get_stratification_key(stratification)
                if pop.empty:
                    pop_in_group = pop
                else:
                    pop_in_group = pop.loc[stratification_group == stratification_key]
                yield (stratification_key,), pop_in_group

    @staticmethod
    def update_labels(measure_data: Dict[str, float], labels: Tuple[str, ...]) -> Dict[str, float]:
        """Updates a dict of measure data with stratification labels.

        Parameters
        ----------
        measure_data
            The measure data with unstratified column names.
        labels
            The stratification labels. Yielded along with the population
            subgroup the measure data was produced from by a call to
            :obj:`ResultsStratifier.group`.

        Returns
        -------
            The measure data with column names updated with the stratification
            labels.

        """
        stratification_label = labels[0]
        measure_data = {f'{k}_{stratification_label}': v for k, v in measure_data.items()}
        return measure_data


class MortalityObserver(MortalityObserver_):

    def __init__(self):
        super().__init__()
        self.stratifier = ResultsStratifier(self.name, True)

    @property
    def sub_components(self) -> List[ResultsStratifier]:
        return [self.stratifier]

    def metrics(self, index: pd.Index, metrics: Dict[str, float]) -> Dict[str, float]:
        pop = self.population_view.get(index)
        pop.loc[pop.exit_time.isnull(), 'exit_time'] = self.clock()

        measure_getters = (
            (get_person_time, ()),
            (get_deaths, (self.causes,)),
            (get_years_of_life_lost, (self.life_expectancy, self.causes)),
        )

        for labels, pop_in_group in self.stratifier.group(pop):
            base_args = (pop_in_group, self.config.to_dict(), self.start_time, self.clock(), self.age_bins)

            for measure_getter, extra_args in measure_getters:
                measure_data = measure_getter(*base_args, *extra_args)
                measure_data = self.stratifier.update_labels(measure_data, labels)
                metrics.update(measure_data)

        the_living = pop[(pop.alive == 'alive') & pop.tracked]
        the_dead = pop[pop.alive == 'dead']
        metrics[project_globals.TOTAL_YLLS_COLUMN] = self.life_expectancy(the_dead.index).sum()
        metrics['total_population_living'] = len(the_living)
        metrics['total_population_dead'] = len(the_dead)

        return metrics


class DisabilityObserver(DisabilityObserver_):

    def __init__(self):
        super().__init__()
        self.stratifier = ResultsStratifier(self.name)

    @property
    def sub_components(self) -> List[ResultsStratifier]:
        return [self.stratifier]

    def on_time_step_prepare(self, event: 'Event'):
        pop = self.population_view.get(event.index, query='tracked == True and alive == "alive"')
        self.update_metrics(pop)

        pop.loc[:, project_globals.TOTAL_YLDS_COLUMN] += self.disability_weight(pop.index)
        self.population_view.update(pop)

    def update_metrics(self, pop: pd.DataFrame):
        for labels, pop_in_group in self.stratifier.group(pop):
            ylds_this_step = get_years_lived_with_disability(pop_in_group, self.config.to_dict(),
                                                             self.clock().year, self.step_size(),
                                                             self.age_bins, self.disability_weight_pipelines,
                                                             self.causes)
            ylds_this_step = self.stratifier.update_labels(ylds_this_step, labels)
            self.years_lived_with_disability.update(ylds_this_step)


class DiseaseObserver:
    """Observes transition counts and person time for a cause."""
    configuration_defaults = {
        'metrics': {
            'disease': {
                'by_age': False,
                'by_year': False,
                'by_sex': False,
            }
        }
    }

    def __init__(self, disease: str, stratify_by_screening_state: str = 'False'):
        self.disease = disease
        self.configuration_defaults = {
            'metrics': {disease: DiseaseObserver.configuration_defaults['metrics']['disease']}
        }
        self.stratifier = ResultsStratifier(self.name, stratify_by_screening_state == 'True')

    @property
    def name(self) -> str:
        return f'{self.disease}_observer'

    @property
    def sub_components(self) -> List[ResultsStratifier]:
        return [self.stratifier]

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        self.config = builder.configuration['metrics'][self.disease].to_dict()
        self.clock = builder.time.clock()
        self.age_bins = get_age_bins(builder)
        self.counts = Counter()
        self.person_time = Counter()

        self.states = project_globals.STATE_MACHINE_MAP[self.disease]['states']
        self.transitions = project_globals.STATE_MACHINE_MAP[self.disease]['transitions']

        self.previous_state_column = f'previous_{self.disease}'
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=[self.previous_state_column])

        columns_required = ['alive', self.disease, self.previous_state_column]
        if self.config['by_age']:
            columns_required += ['age']
        if self.config['by_sex']:
            columns_required += ['sex']
        self.population_view = builder.population.get_view(columns_required)

        builder.value.register_value_modifier('metrics', self.metrics)
        # FIXME: The state table is modified before the clock advances.
        # In order to get an accurate representation of person time we need to look at
        # the state table before anything happens.
        builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        self.population_view.update(pd.Series('', index=pop_data.index, name=self.previous_state_column))

    def on_time_step_prepare(self, event: 'Event'):
        pop = self.population_view.get(event.index)
        # Ignoring the edge case where the step spans a new year.
        # Accrue all counts and time to the current year.
        for labels, pop_in_group in self.stratifier.group(pop):
            for state in self.states:
                # noinspection PyTypeChecker
                state_person_time_this_step = get_state_person_time(pop_in_group, self.config, self.disease, state,
                                                                    self.clock().year, event.step_size, self.age_bins)
                state_person_time_this_step = self.stratifier.update_labels(state_person_time_this_step, labels)
                self.person_time.update(state_person_time_this_step)

        # This enables tracking of transitions between states
        prior_state_pop = self.population_view.get(event.index)
        prior_state_pop[self.previous_state_column] = prior_state_pop[self.disease]
        self.population_view.update(prior_state_pop)

    def on_collect_metrics(self, event: 'Event'):
        pop = self.population_view.get(event.index)
        for labels, pop_in_group in self.stratifier.group(pop):
            for transition in self.transitions:
                # noinspection PyTypeChecker
                transition_counts_this_step = get_transition_count(pop_in_group, self.config, self.disease, transition,
                                                                   event.time, self.age_bins)
                transition_counts_this_step = self.stratifier.update_labels(transition_counts_this_step, labels)
                self.counts.update(transition_counts_this_step)

    def metrics(self, index: pd.Index, metrics: Dict[str, float]):
        metrics.update(self.counts)
        metrics.update(self.person_time)
        return metrics

    def __repr__(self) -> str:
        return f"DiseaseObserver({self.disease})"


class ScreeningObserver:
    """Observes screening appointments scheduled and attended"""
    configuration_defaults = {
        'metrics': {
            'screening': {
                'by_age': False,
                'by_year': False,
                'by_sex': False,
            }
        }
    }

    def __init__(self):
        self.configuration_defaults = {
            'metrics': {'screening': ScreeningObserver.configuration_defaults['metrics']['screening']}
        }
        self.stratifier = ResultsStratifier(self.name)

    @property
    def name(self) -> str:
        return 'screening_observer'

    @property
    def sub_components(self) -> List[ResultsStratifier]:
        return [self.stratifier]

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        self.config = builder.configuration['metrics']['screening'].to_dict()
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.age_bins = get_age_bins(builder)
        self.counts = Counter()

        columns_required = [
            'alive',
            project_globals.ATTENDED_LAST_SCREENING,
            project_globals.PREVIOUS_SCREENING_DATE,
        ]
        if self.config['by_age']:
            columns_required += ['age']
        if self.config['by_sex']:
            columns_required += ['sex']
        self.population_view = builder.population.get_view(columns_required)

        builder.value.register_value_modifier('metrics', self.metrics)
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)

    def on_collect_metrics(self, event: 'Event'):
        pop = self.population_view.get(event.index)
        for labels, pop_in_group in self.stratifier.group(pop):
            for sex in project_globals.SEXES:
                sex_mask = pop_in_group.loc[:, 'sex'] == sex.title()
                scheduled_screening = (pop_in_group.loc[sex_mask, project_globals.PREVIOUS_SCREENING_DATE]
                                       > (self.clock() - self.step_size()))
                attended_screening = scheduled_screening & pop_in_group.loc[:, project_globals.ATTENDED_LAST_SCREENING]
                year_sex = f'in_{self.clock().year}_among_{sex}'
                counts_this_step = self.stratifier.update_labels(
                    {
                        f'{project_globals.SCREENING_SCHEDULED}_{year_sex}': sum(scheduled_screening),
                        f'{project_globals.SCREENING_ATTENDED}_{year_sex}': sum(attended_screening)
                    }, labels
                )
                self.counts.update(counts_this_step)

    def metrics(self, index: pd.Index, metrics: Dict[str, float]):
        metrics.update(self.counts)
        return metrics

    def __repr__(self) -> str:
        return 'ScreeningObserver'


class SampleHistoryObserver:

    configuration_defaults = {
        'metrics': {
            'sample_history_observer': {
                'sample_size': 1000,
                'path': f'{paths.RESULTS_ROOT}/sample_history.hdf'
            }
        }
    }

    @property
    def name(self):
        return "sample_history_observer"

    def __init__(self):
        self.history_snapshots = []
        self.sample_index = None

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.sample_history_parameters = builder.configuration.metrics.sample_history_observer
        self.randomness = builder.randomness.get_stream("sample_history")

        # sets the sample index
        builder.population.initializes_simulants(self.on_initialize_simulants, requires_streams=['sample_history'])

        columns_required = [
            'alive', 'age', 'sex', 'entrance_time', 'exit_time',
            project_globals.BREAST_CANCER_MODEL_NAME,
            project_globals.SCREENING_RESULT_MODEL_NAME,
            'cause_of_death',
            project_globals.PREVIOUS_SCREENING_DATE,
            project_globals.ATTENDED_LAST_SCREENING,
        ] + [f'{state}_event_time' for state in project_globals.BREAST_CANCER_MODEL_STATES]
        self.population_view = builder.population.get_view(columns_required)

        # keys will become column names in the output
        self.pipelines = {
            'family_history': builder.value.get_value('family_history.exposure')
        }

        # record on time_step__prepare to make sure all pipelines + state table
        # columns are reflective of same time
        builder.event.register_listener('time_step__cleanup', self.on_time_step_cleanup)
        builder.event.register_listener('simulation_end', self.on_simulation_end)

    def on_initialize_simulants(self, pop_data):
        sample_size = self.sample_history_parameters.sample_size
        if sample_size is None or sample_size > len(pop_data.index):
            sample_size = len(pop_data.index)
        draw = self.randomness.get_draw(pop_data.index)
        priority_index = [i for d, i in sorted(zip(draw, pop_data.index), key=lambda x:x[0])]
        self.sample_index = pd.Index(priority_index[:sample_size])

    def on_time_step_cleanup(self, event):
        pop = self.population_view.get(self.sample_index)

        pipeline_results = []
        for name, pipeline in self.pipelines.items():
            values = pipeline(self.sample_index)
            values = values.rename(name)
            pipeline_results.append(values)

        record = pd.concat(pipeline_results + [pop], axis=1)
        record.loc[:, 'date'] = self.clock()

        # Get screenings scheduled and attended this timestep
        record.loc[:, 'scheduled_screening'] = (pop.loc[:, project_globals.PREVIOUS_SCREENING_DATE]
                                                > self.clock() - self.step_size())
        record.loc[:, 'attended_screening'] = (record.loc[:, 'scheduled_screening']
                                               & pop.loc[:, project_globals.ATTENDED_LAST_SCREENING])
        del record[project_globals.PREVIOUS_SCREENING_DATE]
        del record[project_globals.ATTENDED_LAST_SCREENING]

        record.index.rename("simulant", inplace=True)
        record.set_index('date', append=True, inplace=True)
        self.history_snapshots.append(record)

    def on_simulation_end(self, event):
        sample_history = pd.concat(self.history_snapshots, axis=0)
        sample_history.to_hdf(self.sample_history_parameters.path, key='trajectories')


def get_state_person_time(pop: pd.DataFrame, config: Dict[str, bool],
                          disease: str, state: str, current_year: Union[str, int],
                          step_size: pd.Timedelta, age_bins: pd.DataFrame) -> Dict[str, float]:
    """Custom person time getter that handles state column name assumptions"""
    base_key = get_output_template(**config).substitute(measure=f'{state}_person_time',
                                                        year=current_year)
    base_filter = QueryString(f'alive == "alive" and {disease} == "{state}"')
    person_time = get_group_counts(pop, base_filter, base_key, config, age_bins,
                                   aggregate=lambda x: len(x) * to_years(step_size))
    return person_time


def get_transition_count(pop: pd.DataFrame, config: Dict[str, bool],
                         disease: str, transition: project_globals.TransitionString,
                         event_time: pd.Timestamp, age_bins: pd.DataFrame) -> Dict[str, float]:
    """Counts transitions that occurred this step."""
    event_this_step = ((pop[f'previous_{disease}'] == transition.from_state)
                       & (pop[disease] == transition.to_state))
    transitioned_pop = pop.loc[event_this_step]
    base_key = get_output_template(**config).substitute(measure=f'{transition}_event_count',
                                                        year=event_time.year)
    base_filter = QueryString('')
    transition_count = get_group_counts(transitioned_pop, base_filter, base_key, config, age_bins)
    return transition_count
