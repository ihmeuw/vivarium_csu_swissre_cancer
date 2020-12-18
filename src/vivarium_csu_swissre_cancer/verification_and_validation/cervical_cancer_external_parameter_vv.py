import numpy as np
import pandas as pd

columns = ['age_cohort', 'year', 'input_draw', 'scenario', 'measure']

def get_person_time(sim_result_dir: str):
    disease_state_person_time = (pd
                                 .read_csv(sim_result_dir + 'disease_state_person_time.csv', index_col=0)
                                 .groupby(columns + ['cause'])
                                 .value.sum()
                                 .reset_index())
    person_time = (disease_state_person_time
                   .groupby(columns)
                   .value.sum()
                   .reset_index())
    return disease_state_person_time, person_time

def calc_hrhpv_prevalence(disease_state_person_time: pd.DataFrame, person_time: pd.DataFrame):
    hrhpv_prev = (disease_state_person_time
                  .loc[disease_state_person_time.cause == 'high_risk_hpv']
                  .set_index(columns + ['cause'])
                  .div(person_time.set_index(columns))
                  .reset_index())
    hrhpv_prev['measure'] = 'prevalence'
    hrhpv_prev_summary = (hrhpv_prev
                          .groupby(['age_cohort', 'year', 'scenario', 'measure', 'cause'])
                          .value.describe(percentiles=[.025, .975])
                          .filter(['mean', '2.5%', '97.5%'])
                          .reset_index())
    hrhpv_prev_summary[['mean', '2.5%', '97.5%']] *= 100
    hrhpv_prev_summary['age_midpoint'] = list(map(lambda x, y: y-int(x.split('_to_')[1])+2,
                                                  hrhpv_prev_summary['age_cohort'], hrhpv_prev_summary['year']))
    return hrhpv_prev_summary

def calc_screening_coverage(sim_result_dir: str):
    event_count = (pd
                   .read_csv(sim_result_dir + 'event_count.csv', index_col=0)
                   .groupby(['year', 'input_draw', 'scenario', 'measure'])
                   .value.sum()
                   .reset_index())
    scheduled = (event_count
                 .loc[event_count.measure == 'screening_scheduled_count']
                 .set_index(['year', 'input_draw', 'scenario'])
                 .value)
    attended = (event_count
                .loc[event_count.measure == 'screening_attended_count']
                .set_index(['year', 'input_draw', 'scenario'])
                .value)
    coverage = (100 * attended / scheduled).reset_index()
    coverage_summary = (coverage
                        .groupby(['year', 'scenario'])
                        .value.describe(percentiles=[.025, .975])
                        .filter(['mean', '2.5%', '97.5%'])
                        .reset_index())
    return coverage_summary

def calc_vaccination_coverage(sim_result_dir: str):
    pt = (pd
          .read_csv(sim_result_dir + 'disease_state_person_time.csv', index_col=0)
          .groupby(['year', 'input_draw', 'scenario', 'vaccination_state'])
          .value.sum()
          .reset_index())
    unvaccinated = (pt
                    .loc[pt.vaccination_state == 'not_vaccinated']
                    .set_index(['year', 'input_draw', 'scenario'])
                    .value)
    vaccinated = (pt
                  .loc[pt.vaccination_state == 'vaccinated']
                  .set_index(['year', 'input_draw', 'scenario'])
                  .value)
    coverage = (100 * vaccinated / (unvaccinated + vaccinated)).reset_index()
    coverage_summary = (coverage
                        .groupby(['year', 'scenario'])
                        .value.describe(percentiles=[.025, .975])
                        .filter(['mean', '2.5%', '97.5%'])
                        .reset_index())
    return coverage_summary

def calc_rr_hrHPV(sim_result_dir: str, disease_state_person_time: pd.DataFrame):
    transition_count = (pd
                        .read_csv(sim_result_dir + 'disease_transition_count.csv', index_col=0)
                        .groupby(columns)
                        .value.sum()
                        .reset_index())
    bcc_with_hpv_incidence_count = (transition_count
                                    .loc[transition_count.measure == 'high_risk_hpv_to_benign_cervical_cancer_with_hpv_event_count']
                                    .set_index(['age_cohort', 'year', 'input_draw', 'scenario'])
                                    .value)
    bcc_without_hpv_incidence_count = (transition_count
                                       .loc[transition_count.measure == 'susceptible_to_cervical_cancer_to_benign_cervical_cancer_event_count']
                                       .set_index(['age_cohort', 'year', 'input_draw', 'scenario'])
                                       .value)
    hrhpv_pt = (disease_state_person_time
                .loc[disease_state_person_time.cause == 'high_risk_hpv']
                .set_index(['age_cohort', 'year', 'input_draw', 'scenario'])
                .value)
    susceptible_pt = (disease_state_person_time
                      .loc[disease_state_person_time.cause == 'susceptible_to_cervical_cancer']
                      .set_index(['age_cohort', 'year', 'input_draw', 'scenario'])
                      .value)
    bcc_with_hpv_incidence_rate = bcc_with_hpv_incidence_count / hrhpv_pt
    bcc_without_hpv_incidence_rate = bcc_without_hpv_incidence_count / susceptible_pt
    rr = (bcc_with_hpv_incidence_rate / bcc_without_hpv_incidence_rate).reset_index()
    rr = rr.loc[~((np.isinf(rr.value)) | (np.isnan(rr.value)))]
    return rr

def calc_rr_vaccination(sim_result_dir: str, sink_state: str):
    if sink_state == 'high_risk_hpv':
      transition_count = (pd
                          .read_csv(sim_result_dir + 'disease_transition_count.csv', index_col=0)
                          .query('measure == "susceptible_to_cervical_cancer_to_high_risk_hpv_event_count"')
                          .groupby(['age_cohort', 'year', 'input_draw', 'scenario', 'vaccination_state'])
                          .value.sum())
      pt = (pd
            .read_csv(sim_result_dir + 'disease_state_person_time.csv', index_col=0)
            .query('cause == "susceptible_to_cervical_cancer"')
            .groupby(['age_cohort', 'year', 'input_draw', 'scenario', 'vaccination_state'])
            .value.sum())
    elif sink_state == 'benign_cervical_cancer':
      transition_count = (pd
                          .read_csv(sim_result_dir + 'disease_transition_count.csv', index_col=0)
                          .query('measure == "susceptible_to_cervical_cancer_to_benign_cervical_cancer_event_count"')
                          .groupby(['age_cohort', 'year', 'input_draw', 'scenario', 'vaccination_state'])
                          .value.sum())
      pt = (pd
            .read_csv(sim_result_dir + 'disease_state_person_time.csv', index_col=0)
            .query('cause == "susceptible_to_cervical_cancer"')
            .groupby(['age_cohort', 'year', 'input_draw', 'scenario', 'vaccination_state'])
            .value.sum())
    else: #sink_state == 'benign_cervical_cancer_with_hpv':
      transition_count = (pd
                          .read_csv(sim_result_dir + 'disease_transition_count.csv', index_col=0)
                          .query('measure == "benign_cervical_cancer_to_benign_cervical_cancer_with_hpv_event_count"')
                          .groupby(['age_cohort', 'year', 'input_draw', 'scenario', 'vaccination_state'])
                          .value.sum())
      pt = (pd
            .read_csv(sim_result_dir + 'disease_state_person_time.csv', index_col=0)
            .query('cause == "benign_cervical_cancer"')
            .groupby(['age_cohort', 'year', 'input_draw', 'scenario', 'vaccination_state'])
            .value.sum())
    incidence_rate = (transition_count / pt).reset_index()
    unvaccinated = (incidence_rate
                    .loc[incidence_rate.vaccination_state == 'not_vaccinated']
                    .set_index(['age_cohort', 'year', 'input_draw', 'scenario'])
                    .value)
    vaccinated = (incidence_rate
                  .loc[incidence_rate.vaccination_state == 'vaccinated']
                  .set_index(['age_cohort', 'year', 'input_draw', 'scenario'])
                  .value)
    rr = (unvaccinated / vaccinated).reset_index()
    rr = rr.loc[~((np.isinf(rr.value)) | (np.isnan(rr.value)))]
    return rr

def calc_treatment_coverage(sim_result_dir: str):
    pt = (pd
          .read_csv(sim_result_dir + 'disease_state_person_time.csv', index_col=0)
          .groupby(['year', 'input_draw', 'scenario', 'treatment_state'])
          .value.sum()
          .reset_index())
    untreated = (pt
                 .loc[pt.treatment_state == 'not_treated']
                 .set_index(['year', 'input_draw', 'scenario'])
                 .value)
    treated = (pt
               .loc[pt.treatment_state == 'treated']
               .set_index(['year', 'input_draw', 'scenario'])
               .value)
    coverage = (100 * treated / (untreated + treated)).reset_index()
    coverage_summary = (coverage
                        .groupby(['year', 'scenario'])
                        .value.describe(percentiles=[.025, .975])
                        .filter(['mean', '2.5%', '97.5%'])
                        .reset_index())
    return coverage_summary

def calc_rr_treatment(sim_result_dir: str):
    transition_count = (pd
                        .read_csv(sim_result_dir + 'disease_transition_count.csv', index_col=0)
                        .query('measure == "benign_cervical_cancer_to_invasive_cervical_cancer_event_count" \
                               | measure == "benign_cervical_cancer_with_hpv_to_invasive_cervical_cancer_with_hpv_event_count"')
                        .groupby(['age_cohort', 'year', 'input_draw', 'scenario', 'treatment_state'])
                        .value.sum())
    pt = (pd
          .read_csv(sim_result_dir + 'disease_state_person_time.csv', index_col=0)
          .query('cause == "benign_cervical_cancer" | cause == "benign_cervical_cancer_with_hpv"')
          .groupby(['age_cohort', 'year', 'input_draw', 'scenario', 'treatment_state'])
          .value.sum())
    icc_incidence_rate = (transition_count / pt).reset_index()
    untreated = (icc_incidence_rate
                 .loc[icc_incidence_rate.treatment_state == 'not_treated']
                 .set_index(['age_cohort', 'year', 'input_draw', 'scenario'])
                 .value)
    treated = (icc_incidence_rate
               .loc[icc_incidence_rate.treatment_state == 'treated']
               .set_index(['age_cohort', 'year', 'input_draw', 'scenario'])
               .value)
    rr = (untreated / treated).reset_index()
    rr = rr.loc[~((np.isinf(rr.value)) | (np.isnan(rr.value)))]
    return rr
