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
                                                      get_age_sex_filter_and_iterables)

from vivarium_csu_swissre_cancer import globals as project_globals, paths
from vivarium_csu_swissre_cancer.components.disease import BreastCancer

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

    def __init__(self, observer_name: str):
        self.name = f'{observer_name}_results_stratifier'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        """Perform this component's setup."""
        # The only thing you should request here are resources necessary for
        # results stratification.
        self.pipelines = {
            # 'pipeline_one': builder.value.get_value('pipeline_one'),
        }
        columns_required = [
            # 'column_one',
        ]
        self.population_view = builder.population.get_view(columns_required)
        self.stratification_levels = {
            # 'C1': {
            #     'high': lambda: self.population_values['column_one'] > 0,
            #     'low': lambda: self.population_values['column_one'] <= 0,
            # },
            # 'P1': {
            #     'high': lambda: self.pipeline_values['pipeline_one'] > 5,
            #     'low': lambda: self.pipelines['pipeline_one'] <= 5,
            # },
        }
        self.pipeline_values = {pipeline: None for pipeline in self.pipelines}
        self.population_values = None
        self.stratification_groups = None

        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 requires_columns=columns_required,
                                                 requires_values=list(self.pipelines.keys()))

    # noinspection PyAttributeOutsideInit
    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        stratification_groups = pd.Series('', index=pop_data.index)

        self.pipeline_values = {name: pipeline(pop_data.index) for name, pipeline in self.pipelines.items()}
        self.population_values = self.population_view.get(pop_data.index)

        all_stratifications = self.get_all_stratifications()
        for stratification in all_stratifications:
            stratification_group_name = '_'.join([f'{metric["metric"]}_{metric["category"]}'
                                                  for metric in stratification])
            mask = pd.Series(True, index=pop_data.index)
            for metric in stratification:
                mask &= self.stratification_levels[metric['metric']][metric['category']]()
            stratification_groups.loc[mask] = stratification_group_name

        self.stratification_groups = stratification_groups

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

    @staticmethod
    def get_stratification_key(stratification: Iterable[Dict[str, str]]) -> str:
        return ('' if not stratification
                else '_'.join([f'{metric["metric"]}_{metric["category"]}' for metric in stratification]))

    def group(self, population: pd.DataFrame) -> Iterable[Tuple[Tuple[str, ...], pd.DataFrame]]:
        """Takes the full population and yields stratified subgroups.

        Parameters
        ----------
        population
            The population to stratify.

        Yields
        ------
            A tuple of stratification labels and the population subgroup
            corresponding to those labels.

        """
        stratification_group = self.stratification_groups.loc[population.index]
        stratifications = self.get_all_stratifications()
        for stratification in stratifications:
            stratification_key = self.get_stratification_key(stratification)
            if population.empty:
                pop_in_group = population
            else:
                pop_in_group = population.loc[stratification_group == stratification_key]
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
        self.stratifier = ResultsStratifier(self.name)

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
            'disease_observer': {
                'by_age': False,
                'by_year': False,
                'by_sex': False,
            }
        }
    }

    def __init__(self, disease: str):
        self.disease = disease
        self.configuration_defaults = {
            'metrics': {f'{disease}_observer': DiseaseObserver.configuration_defaults['metrics']['disease_observer']}
        }
        self.stratifier = ResultsStratifier(self.name)

    @property
    def name(self) -> str:
        return f'disease_observer.{self.disease}'

    @property
    def sub_components(self) -> List[ResultsStratifier]:
        return [self.stratifier]

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        self.config = builder.configuration['metrics'][f'{self.disease}_observer'].to_dict()
        self.clock = builder.time.clock()
        self.age_bins = get_age_bins(builder)
        self.counts = Counter()
        self.person_time = Counter()

        self.states = project_globals.DISEASE_MODEL_MAP[self.disease]['states']
        self.transitions = project_globals.DISEASE_MODEL_MAP[self.disease]['transitions']

        self.previous_state_column = f'previous_{self.disease}'
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=[self.previous_state_column])

        columns_required = ['alive', f'{self.disease}', self.previous_state_column]
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
