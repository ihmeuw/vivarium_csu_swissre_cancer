import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_sim_count_data(sim_result_dir: str):
    columns = ['age_cohort', 'year', 'input_draw', 'scenario', 'measure']
    df = {}
    fnames = ['deaths', 'disease_transition_count', 'disease_state_person_time', 'person_time']
    for fname in fnames:
        df[fname] = pd.read_csv(sim_result_dir + fname + '.csv').iloc[:, 1:]
        # get aggregated results if stratifications like risk or screening exist
        if 'cause' in df[fname].columns:
            df[fname] = df[fname].groupby(columns + ['cause']).value.sum().reset_index()
        else:
            df[fname] = df[fname].groupby(columns).value.sum().reset_index()
    return df

def get_all_causes_death_count(data: pd.DataFrame):
    count = (data
             .groupby(['age_cohort', 'year', 'input_draw', 'scenario', 'measure'])
             .value.sum()
             .reset_index())
    count['cause'] = 'all_causes'
    return count

def get_cervical_cancer_incidence_count(data: pd.DataFrame):
    """
    benign_cervical_cancer_incidence_count = 
    high_risk_hpv_to_benign_cervical_cancer_event_count +
    susceptible_to_cervical_cancer_to_benign_cervical_cancer_event_count
    
    invasive_cervical_cancer_incidence_count = 
    benign_cervical_cancer_to_cervical_cancer_event_count
    """
    idx_columns = ['age_cohort', 'year', 'input_draw', 'scenario']
    hrhpv_to_bcc_event_count = (data
                                .query('measure == "high_risk_hpv_to_benign_cervical_cancer_event_count"')
                                .drop(columns='measure')
                                .set_index(idx_columns))
    s_to_bcc_event_count = (data
                            .query('measure == "susceptible_to_cervical_cancer_to_benign_cervical_cancer_event_count"')
                            .drop(columns='measure')
                            .set_index(idx_columns))
    bcc_incidence_count = (hrhpv_to_bcc_event_count + s_to_bcc_event_count).reset_index()
    bcc_incidence_count['measure'] = 'benign_cervical_cancer_incidence'
    
    icc_incidence_count = data.query('measure == "benign_cervical_cancer_to_cervical_cancer_event_count"')
    icc_incidence_count['measure'] = 'invasive_cervical_cancer_incidence'

    return bcc_incidence_count, icc_incidence_count

def get_measure(data: pd.DataFrame, person_time: pd.DataFrame, measure: str, by_cause=True):
    join_columns = ['cause'] if by_cause else []
    multiplier = 1 if measure == 'prevalence' else 100_000
    result = (data.set_index(['age_cohort', 'year', 'input_draw', 'scenario', 'measure'] + join_columns)
              .div(person_time.drop(columns='measure').set_index(['age_cohort', 'year', 'input_draw', 'scenario']))
              .mul(multiplier)
              .reset_index())
    if measure == 'prevalence':
        result['measure'] = measure 
    result_summary = (result
                      .groupby(['age_cohort', 'year', 'scenario', 'measure'] + join_columns)
                      .value.describe(percentiles=[.025, .975])
                      .filter(['mean', '2.5%', '97.5%'])
                      .reset_index())
    return result_summary

def add_age_midpoint(sim_data: pd.DataFrame, measure: str, forecast_data_dir: str):
    sim_data = sim_data.query('scenario == "baseline"')
    sim_data['age_midpoint'] = list(map(lambda x, y: y-int(x.split('_to_')[1])+2, sim_data['age_cohort'], sim_data['year']))
    if measure == 'acmr':
        forecast_data = (pd
                         .read_csv(forecast_data_dir + 'acmr_forecast.csv')
                         .query('sex == "female"')
                         .drop(columns=['location', 'age_group_id', 'sex_id', 'sex'])
                         .rename(columns={'year_id': 'year'}))
        forecast_data['age_midpoint'] = forecast_data['age'].map(lambda x: int(x.split(' ')[0])+2)
    else:
        forecast_data = pd.read_csv(forecast_data_dir + f'c432_{measure}_forecast.csv').drop(columns='sex')
        forecast_data['age_midpoint'] = forecast_data['age_group'].map(lambda x: int(x.split(' ')[0])+2)
    
    return sim_data, forecast_data

def plot_sim_vs_forecast(sim_data: pd.DataFrame, forecast_data: pd.DataFrame, year: int, measure: str):
    sim_sub = sim_data.loc[sim_data.year == year]
    forecast_sub = forecast_data.loc[forecast_data.year == year]
    
    fig = plt.figure(figsize=(6, 4), dpi=120)
    plt.plot(sim_sub['age_midpoint'], sim_sub['mean'], marker='o', label='sim baseline')
    plt.fill_between(sim_sub['age_midpoint'], sim_sub['2.5%'], sim_sub['97.5%'], alpha=.3)
    plt.plot(forecast_sub['age_midpoint'], forecast_sub['mean'], marker='o', label='forecast')
    plt.fill_between(forecast_sub['age_midpoint'], forecast_sub['lb'], forecast_sub['ub'], alpha=.3)
    plt.title(year)
    plt.xlabel('Age')
    if measure == 'acmr':
        plt.ylabel('Deaths due to all_causes\n (per 100,000 PY)')
    elif measure == 'deaths':
        plt.ylabel(f'Deaths due to {cancer_name}\n (per 100,000 PY)')
    elif measure == 'incidence':
        plt.ylabel(f'Incidence of {cancer_name}\n (Cases per 100,000 PY)')
    else: # measure == 'prevalence'
        plt.ylabel(f'Prevalence of {cancer_name}\n (proportion)')
    plt.legend(loc=(1.05, .1))
    plt.grid()

if __name__ == '__main__':
    cancer_name = 'cervical_cancer'
    master_dir = '/home/j/Project/simulation_science/cancer/'
    sim_result_dir = '/ihme/costeffectiveness/results/vivarium_csu_swissre_cervical_cancer/v1.0_cancer/swissre_coverage/2020_10_16_14_42_07/count_data/'
    forecast_data_dir = master_dir + f'forecast/{cancer_name}/'
    output_dir = master_dir + f'verification_and_validation/{cancer_name}/v1.0_cancer/'

    df = load_sim_count_data(sim_result_dir)
    total_deaths = get_all_causes_death_count(df['deaths'])
    bcc_incidence_count, icc_incidence_count = get_cervical_cancer_incidence_count(df['disease_transition_count'])
    acmr = get_measure(total_deaths, df['person_time'], 'deaths')
    icc_deaths = get_measure(df['deaths'].query('cause == "cervical_cancer"'), df['person_time'], 'deaths')
    icc_incidence = get_measure(icc_incidence_count, df['person_time'], 'incidence', by_cause=False)
    icc_prevalence = get_measure(df['disease_state_person_time'].query('cause == "cervical_cancer"'), df['person_time'], 'prevalence')
    sim_acmr, forecast_acmr = add_age_midpoint(acmr, 'acmr', forecast_data_dir)
    sim_deaths, forecast_deaths = add_age_midpoint(icc_deaths, 'deaths', forecast_data_dir)
    sim_incidence, forecast_incidence = add_age_midpoint(icc_incidence, 'incidence', forecast_data_dir)
    sim_prevalence, forecast_prevalence = add_age_midpoint(icc_prevalence, 'prevalence', forecast_data_dir)

with PdfPages(output_dir + 'acmr.pdf') as pdf:
    for year in range(2020, 2041):
        plot_sim_vs_forecast(sim_acmr, forecast_acmr, year, 'acmr')
        pdf.savefig(bbox_inches='tight')

with PdfPages(output_dir + 'deaths.pdf') as pdf:
    for year in range(2020, 2041):
        plot_sim_vs_forecast(sim_deaths, forecast_deaths, year, 'deaths')
        pdf.savefig(bbox_inches='tight')

with PdfPages(output_dir + 'incidence.pdf') as pdf:
    for year in range(2020, 2041):
        plot_sim_vs_forecast(sim_incidence, forecast_incidence, year, 'incidence')
        pdf.savefig(bbox_inches='tight')

with PdfPages(output_dir + 'prevalence.pdf') as pdf:
    for year in range(2020, 2041):
        plot_sim_vs_forecast(sim_prevalence, forecast_prevalence, year, 'prevalence')
        pdf.savefig(bbox_inches='tight')
