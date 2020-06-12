from vivarium_public_health.disease import (DiseaseState as DiseaseState_, DiseaseModel, SusceptibleState,
                                            RateTransition as RateTransition_)

from vivarium_csu_swissre_cancer import globals as project_globals


class RateTransition(RateTransition_):
    def load_transition_rate_data(self, builder):
        if 'transition_rate' in self._get_data_functions:
            rate_data = self._get_data_functions['transition_rate'](builder, self.input_state.cause,
                                                                    self.output_state.cause)
            pipeline_name = f'{self.input_state.cause}_to_{self.output_state.cause}.transition_rate'
        else:
            raise ValueError("No valid data functions supplied.")
        return rate_data, pipeline_name


class DiseaseState(DiseaseState_):

    # I really need to rewrite the state machine code.  It's super inflexible
    def add_transition(self, output, source_data_type=None, get_data_functions=None, **kwargs):
        if source_data_type == 'rate':
            if get_data_functions is None or 'transition_rate' not in get_data_functions:
                raise ValueError('Must supply get data functions for transition_rate.')
            t = RateTransition(self, output, get_data_functions, **kwargs)
            self.transition_set.append(t)
        else:
            t = super().add_transition(output, source_data_type, get_data_functions, **kwargs)
        return t


def BreastCancer():
    susceptible = SusceptibleState(project_globals.BREAST_CANCER_MODEL_NAME)
    lcis = DiseaseState(
        project_globals.LCIS_STATE_NAME,
        cause_type='sequela',
        get_data_functions={
            'disability_weight': lambda *_: 0,
            'excess_mortality_rate': lambda *_: 0,
        },
    )
    dcis = DiseaseState(
        project_globals.DCIS_STATE_NAME,
        cause_type='sequela',
        get_data_functions={
            'disability_weight': lambda *_: 0,
            'excess_mortality_rate': lambda *_: 0,
        },
    )
    breast_cancer = DiseaseState(
        project_globals.BREAST_CANCER_STATE_NAME,
        # TODO remove this once disability weight is resolved
        get_data_functions={
            'disability_weight': lambda *_: 0,
        },
    )

    # Add transitions for Susceptible state
    susceptible.allow_self_transitions()
    susceptible.add_transition(
        lcis,
        source_data_type='rate',
        get_data_functions={
            'incidence_rate': lambda _, builder: builder.data.load(project_globals.BREAST_CANCER.LCIS_INCIDENCE_RATE)
        }
    )
    susceptible.add_transition(
        dcis,
        source_data_type='rate',
        get_data_functions={
            'incidence_rate': lambda _, builder: builder.data.load(project_globals.BREAST_CANCER.DCIS_INCIDENCE_RATE)
        }
    )

    # Add transitions for DCIS state
    dcis.allow_self_transitions()
    dcis.add_transition(
        breast_cancer,
        source_data_type='rate',
        get_data_functions={
            'transition_rate':
                lambda builder, *_: builder.data.load(project_globals.BREAST_CANCER.DCIS_BREAST_CANCER_TRANSITION_RATE)
        }
    )

    # Add transitions for LCIS state
    lcis.allow_self_transitions()
    lcis.add_transition(
        breast_cancer,
        source_data_type='rate',
        get_data_functions={
            'transition_rate':
                lambda builder, *_: builder.data.load(project_globals.BREAST_CANCER.LCIS_BREAST_CANCER_TRANSITION_RATE)
        }
    )

    # Add transitions for Breast Cancer state
    breast_cancer.allow_self_transitions()

    return DiseaseModel('breast_cancer', states=[susceptible, dcis, lcis, breast_cancer])
