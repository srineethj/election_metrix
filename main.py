import csv
import random

import pandas as pd
import sys
import numpy as np
from scipy.stats import norm
import time
import openai

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.expand_frame_repr', False)  # Avoid line breaks for large frames
pd.set_option('display.float_format', '{:.4f}'.format)

pollster_set = {}


# Unfortunately coconut pilled, as the kids say

def construct_import(data):
    print("---> CONSTRUCTING IMPORT")
    # Define the states of interest
    states_of_interest = [
        'Pennsylvania', 'Michigan', 'Wisconsin', 'North Carolina',
        'Georgia', 'Virginia', 'Arizona', 'Nevada'
    ]
    print("---> ELECTORAL VOTES ASSIGNED")
    electoral_votes = {
        'Alabama': 9,
        'Alaska': 3,
        'Arizona': 11,
        'Arkansas': 6,
        'California': 54,
        'Colorado': 10,
        'Connecticut': 7,
        'Delaware': 3,
        'District of Columbia': 3,
        'Florida': 30,
        'Georgia': 16,
        'Hawaii': 4,
        'Idaho': 4,
        'Illinois': 19,
        'Indiana': 11,
        'Iowa': 6,
        'Kansas': 6,
        'Kentucky': 8,
        'Louisiana': 8,
        'Maine': 2,  # Maine has 2 electoral votes, split by district
        'Maine CD-1': 1,  # Congressional district 1
        'Maine CD-2': 1,  # Congressional district 2
        'Maryland': 10,
        'Massachusetts': 11,
        'Michigan': 15,
        'Minnesota': 10,
        'Mississippi': 6,
        'Missouri': 10,
        'Montana': 4,
        'Nebraska': 4,  # Nebraska has 4 electoral votes, split by district
        'Nebraska CD-2': 1,  # Congressional district 2
        'Nevada': 6,
        'New Hampshire': 4,
        'New Jersey': 14,
        'New Mexico': 5,
        'New York': 28,
        'North Carolina': 16,
        'North Dakota': 3,
        'Ohio': 17,
        'Oklahoma': 7,
        'Oregon': 8,
        'Pennsylvania': 19,
        'Rhode Island': 4,
        'South Carolina': 9,
        'South Dakota': 3,
        'Tennessee': 11,
        'Texas': 40,
        'Utah': 6,
        'Vermont': 3,
        'Virginia': 13,
        'Washington': 12,
        'West Virginia': 4,
        'Wisconsin': 10,
        'Wyoming': 3,
        '**National': 0
    }

    columns_to_keep = [
        'poll_id', 'pollster_id', 'pollster', 'sponsors', 'display_name',
        'numeric_grade', 'methodology', 'transparency_score', 'state',
        'start_date', 'end_date', 'sample_size', 'population', 'party',
        'candidate_id', 'candidate_name', 'pct'
    ]

    bias_data = {
        'Pollster': ['HarrisX', 'UMass Lowell', 'CNN', 'New York Times/Siena', 'Suffolk University', 'FOX News',
                     'SurveyUSA', 'Susquehanna', 'Marist College', 'Univision', 'Emerson College',
                     'Data for Progress', 'CBS News', 'YouGov', 'Siena College', 'Remington Research',
                     'Rasmussen Reports', 'Trafalgar Group', 'InsiderAdvantage'],
        'Average Error': [0.5, 1.2, 1.2, 1.5, 1.7, 1.9, 2.5, 2.6, 2.8, 3.0, 3.5, 3.6, 3.7, 3.7, 4.0, 4.2, 5.1, 5.4,
                          5.8],
        'Error Favored Republicans': [100, 33, 50, 50, 60, 75, 20, 67, 56, 40, 61, 73, 50, 50, 29, 70, 100, 86, 81],
        'Error Favored Democrats': [0, 67, 50, 50, 40, 25, 80, 33, 44, 60, 39, 27, 50, 50, 71, 30, 0, 14, 19]
    }

    DC_dummy_data = {
        'poll_id': [74812, 74812],
        'pollster_id': [241, 241],
        'pollster': ['Ipsos', 'Ipsos'],
        'sponsor_ids': [71, 71],
        'sponsors': ['Reuters', 'Reuters'],
        'display_name': ['Ipsos', 'Ipsos'],
        'pollster_rating_id': [154, 154],
        'pollster_rating_name': ['Ipsos', 'Ipsos'],
        'numeric_grade': [2.8, 2.8],
        'pollscore': [-0.9, -0.9],
        'methodology': ['Probability Panel', 'Probability Panel'],
        'transparency_score': ['', ''],
        'state': ['District of Columbia', 'District of Columbia'],
        'start_date': ['9/12/24', '9/12/24'],
        'end_date': ['9/12/24', '9/12/24'],
        'sponsor_candidate_id': ['', ''],
        'sponsor_candidate': ['', ''],
        'sponsor_candidate_party': ['', ''],
        'endorsed_candidate_id': [142393, 142393],
        'endorsed_candidate_name': ['Kamala Harris', 'Donald Trump'],
        'endorsed_candidate_party': ['DEM', 'REP'],
        'question_id': ['', ''],
        'sample_size': [1107, 1107],
        'population': ['', ''],
        'subpopulation': ['', ''],
        'population_full': ['', ''],
        'tracking': ['', ''],
        'created_at': ['5/19/21 8:57', '5/19/21 8:57'],
        'notes': ['quarter sample', 'quarter sample'],
        'source': ['538', '538'],
        'internal': [8914, 8914],
        'partisan': [2024, 2024],
        'race_id': ['U.S. President', 'U.S. President'],
        'cycle': [0, 0],
        'office_type': ['', ''],
        'seat_number': ['', ''],
        'seat_name': ['', ''],
        'election_date': ['11/5/24', '11/5/24'],
        'stage': ['general', 'general'],
        'nationwide_batch': [False, False],
        'ranked_choice_reallocated': [False, False],
        'ranked_choice_round': ['', ''],
        'party': ['DEM', 'REP'],
        'answer': ['Harris', 'Trump'],
        'candidate_id': [19368, 16640],
        'candidate_name': ['Kamala Harris', 'Donald Trump'],
        'pct': [90, 10]
    }

    dc_poll_df = pd.DataFrame(DC_dummy_data)
    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv(data)
    print("\033[92m---> DATA IMPORTED SUCCESSFULLY\033[0m")
    df_filt = df[columns_to_keep].copy()

    ## add DC dummy data :eyeroll:
    df_filt = df_filt._append(dc_poll_df, ignore_index=True)
    # print(df_filt)
    print("---> APPLIED CORRECTIONS FOR SAFE STATES")

    df_filt['state'] = df_filt['state'].fillna('**National')
    df_filt['sponsors'] = df_filt['sponsors'].fillna('none')
    df_filt['transparency_score'] = df_filt['transparency_score'].fillna('5.0')
    df_filt['start_date'] = pd.to_datetime(df_filt['start_date'], format='%m/%d/%y')
    df_filt['end_date'] = pd.to_datetime(df_filt['end_date'], format='%m/%d/%y')

    df_filt = df_filt[df_filt['party'].isin(['DEM', 'REP'])]
    cutoff_date = pd.to_datetime('2024-01-25')

    df_filt = df_filt[df_filt['end_date'] >= cutoff_date]
    print("---> DATES ADJUSTED")

    df_filt['end_date'] = pd.to_datetime(df_filt['end_date'])
    max_date = df_filt['end_date'].max()

    print("---> LAST UPDATE: " + str(max_date))

    df_filt['candidate_name'] = df_filt['candidate_name'].replace('Joe Biden', 'Kamala Harris')
    df_filt = df_filt[df_filt['candidate_name'].isin(['Donald Trump', 'Kamala Harris'])]

    bias_df = pd.DataFrame(bias_data)

    df_filt = df_filt.merge(bias_df, left_on='pollster', right_on='Pollster', how='left')
    df_filt['corrected_pct'] = df_filt.apply(apply_correction, axis=1)
    df_filt['Pollster'] = df_filt['Pollster'].fillna('unkwn')
    df_filt['Average Error'] = df_filt['Average Error'].fillna('0')
    df_filt['Error Favored Republicans'] = df_filt['Error Favored Republicans'].fillna('0')
    df_filt['Error Favored Democrats'] = df_filt['Error Favored Democrats'].fillna('0')

   # print(df_filt)
    print('---> CREATED poll_data_corrected.csv')
    df_filt.to_csv('poll_data_corrected.csv', index=False)
    df_filt = calculate_weights(df_filt)
    # print(df_filt.loc['weight'])
    # Weighted average calculation
    df_weighted_avg = df_filt.groupby(['state', 'candidate_name']).agg(
        weighted_avg_pct=pd.NamedAgg(column='corrected_pct',
                                     aggfunc=lambda x: np.average(x, weights=df_filt.loc[x.index, 'weight']))
    ).reset_index()

    # Pivot the weighted data to have each candidate's weighted percentage in separate columns
    df_pivot = df_weighted_avg.pivot(index='state', columns='candidate_name', values='weighted_avg_pct').reset_index()

    # Calculate the difference between candidates and assign the winner
    df_pivot['mean_diff'] = df_pivot['Kamala Harris'] - df_pivot['Donald Trump']
    df_pivot['winner'] = np.where(df_pivot['mean_diff'] > 0, 'Kamala Harris', 'Donald Trump')

    # Assign electoral votes to each state
    df_pivot['electoral_votes'] = df_pivot['state'].map(electoral_votes)

    # Calculate win probability based on the margin (using a normal distribution assumption)
    df_pivot['win_percent_harris'] = df_pivot['mean_diff'].apply(lambda x: norm.cdf(x, 0, 6)) * 100
    df_pivot['win_percent_trump'] = (1 - df_pivot['win_percent_harris'] / 100) * 100

    # Determine the leading candidate in each state
    idx_max = df_weighted_avg.groupby('state')['weighted_avg_pct'].idxmax()
    df_winner = df_weighted_avg.loc[idx_max]

    # Map electoral votes to the winning candidate in each state
    df_winner['electoral_votes'] = df_winner['state'].map(electoral_votes)

    # Calculate total electoral votes per candidate
    electoral_tally = df_winner.groupby('candidate_name')['electoral_votes'].sum().reset_index()
    electoral_tally = electoral_tally.rename(columns={'electoral_votes': 'total_electoral_votes'})

    # Output results
    df_winner.to_csv('poll_data_winner.csv', index=False)
    electoral_tally.to_csv('df_winner.csv', index=False)

    print_df(df_pivot)
    sanity_checks(electoral_tally)
    print(electoral_tally)
    return df_pivot


def apply_correction(row):
    if pd.isna(row['Pollster']):
        return row['pct']
    elif row['party'] == 'DEM':
        return row['pct'] + row['Average Error'] * (row['Error Favored Democrats'] / 100)
    elif row['party'] == 'REP':
        return row['pct'] - row['Average Error'] * (row['Error Favored Republicans'] / 100)
    else:
        return row['pct']


def calculate_weighted_sd(poll_data):
    # Example function to calculate weighted standard deviation
    print("---> CALCULATING WEIGHTED SD")
    weighted_sds = []
    for _, row in poll_data.iterrows():
        sample_size = row['sample_size']
        moe = row['margin_of_error']  # Assume you have this data
        sd = moe * np.sqrt(sample_size) / 1.96  # For 95% confidence
        weighted_sds.append(sd)
    return np.mean(weighted_sds)  # Average SD for the model


def calculate_weights(df):
    print("---> CALCULATING WEIGHTS")
    print("")
    # Default sample size if missing
    df['sample_size'] = df['sample_size'].fillna(100)
    df['numeric_grade'] = df['numeric_grade'].fillna(1)

    # Convert end_date to datetime if it's not already
    df['end_date'] = pd.to_datetime(df['end_date'])

    # Calculate days since the earliest end_date for weighting
    min_date = df['end_date'].min()
    df['days_since'] = (df['end_date'] - min_date).dt.days

    # Normalize days_since to use as a weight (more recent dates get higher weights)
    max_days = df['days_since'].max()
    df['recency_weight'] = df['days_since'] / max_days

    # Pollster counts to calculate base weight for each pollster (inverse of frequency)
    pollster_counts = df['pollster'].value_counts()
    df['weight'] = df['pollster'].map(pollster_counts).apply(lambda x: 1 / x)

    # Adjust weight for small or large sample sizes
    df['weight'] *= df['sample_size'] / df['sample_size'].mean()

    # Adjust weight based on numeric_grade, assuming higher grade (3) means higher trustworthiness
    df['weight'] *= df['numeric_grade'] / df['numeric_grade'].mean()

    # Incorporate recency weight
    df['weight'] *= df['recency_weight']

    return df


def sanity_checks(df):
    total_electoral_votes = df['total_electoral_votes'].sum()
    if total_electoral_votes != 538:
        print(f"********** DISCREPANCY DETECTED **********: {538 - total_electoral_votes} VOTES")
    else:
        print("---> \033[92mEC COUNT CHECKS PASSED -> 538 TOTAL\033[0m")
        #print("\033[92m---> DATA IMPORTED SUCCESSFULLY\033[0m")
def generate_electoral_globe(df_pivot, shapefile_path):
    pass
    # # Load US state boundaries from the downloaded shapefile
    # usa_states = gpd.read_file(shapefile_path)
    #
    # # Filter to include only mainland US states
    # usa_states = usa_states[usa_states['admin'] == 'United States of America'].copy()
    #
    # # Ensure both dataframes have matching state names
    # df_pivot['state'] = df_pivot['state'].replace({
    #     'Georgia': 'Georgia',
    #     'Michigan': 'Michigan',
    #     'Wisconsin': 'Wisconsin',
    #     'North Carolina': 'North Carolina',
    #     'Pennsylvania': 'Pennsylvania',
    #     'Arizona': 'Arizona',
    #     'Nevada': 'Nevada',
    #     'Virginia': 'Virginia'
    #     # Add other state mappings if necessary
    # })
    #
    # # Merge the df_pivot data with the usa_states GeoDataFrame on the state name
    # electoral_map = usa_states.merge(df_pivot, left_on='name', right_on='state', how='left')
    #
    # # Set up the color mapping based on the winner column
    # electoral_map['winner_color'] = electoral_map['winner'].map({
    #     'Kamala Harris': 'blue',
    #     'Donald Trump': 'red'
    # })
    #
    # # Set up the projection for the globe
    # projection = ccrs.Orthographic(central_longitude=-100, central_latitude=45)  # Adjust as needed for the US
    #
    # # Plot the electoral map on a globe projection
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': projection})
    #
    # # Add coastlines and country borders for context
    # ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.BORDERS)
    #
    # # Plot states, coloring them by the winner
    # for idx, state in electoral_map.iterrows():
    #     ax.add_geometries([state['geometry']], crs=ccrs.PlateCarree(),
    #                       facecolor=state['winner_color'], edgecolor='black')
    #
    # # Set the title and display the map
    # plt.title('2024 US Electoral Map on a Globe', fontsize=18)
    # plt.show()


def simulate_elections(df_pivot, num_simulations=1000):
    # Initialize results list
    results = []

    # Define the threshold to win the election
    WINNING_THRESHOLD = 270

    # Run simulations
    for _ in range(num_simulations):
        # Initialize electoral votes for candidates
        electoral_votes = {
            'Kamala Harris': 0,
            'Donald Trump': 0
        }

        # Simulate the outcome of each state
        for _, row in df_pivot.iterrows():
            if np.random.rand() < row['win_percent_harris'] / 100:
                electoral_votes['Kamala Harris'] += row['electoral_votes']
            else:
                electoral_votes['Donald Trump'] += row['electoral_votes']

        # Store the result
        results.append(electoral_votes)

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Determine if each outcome leads to a win for either candidate
    df_results['Outcome'] = df_results.apply(
        lambda row: 'Kamala Harris' if row['Kamala Harris'] >= WINNING_THRESHOLD else 'Donald Trump', axis=1)

    # Calculate the percentage chance of each outcome
    df_results_summary = df_results.groupby('Outcome').size().reset_index(name='Counts')
    df_results_summary['Percentage'] = (df_results_summary['Counts'] / num_simulations) * 100

    return df_results_summary


def print_tables(df_summary, top_n=10):
    for i in range(top_n):
        print(f"Table {i + 1}:")
        outcome = df_summary.iloc[i]
        percentage = outcome['Percentage']
        print(f"Percentage chance of outcome: {percentage:.2f}")

        # Create the table for this outcome
        outcome_name = outcome['Outcome']
        df_table = df_summary[df_summary['Outcome'] == outcome_name]

        for _, row in df_table.iterrows():
            print(f"{row['Outcome']:<20} {row['Counts']}")

        print("\n" + "-" * 40 + "\n")


def print_df(df):
    print(df)


def print_swing_states_won(df_pivot, swing_states):
    # Filter df_pivot for swing states
    df_swing = df_pivot[df_pivot['state'].isin(swing_states)]

    # Get swing states won by Kamala Harris
    harris_won = df_swing[df_swing['winner'] == 'Kamala Harris']['state'].tolist()

    # Get swing states won by Donald Trump
    trump_won = df_swing[df_swing['winner'] == 'Donald Trump']['state'].tolist()

    # Print results
    print("DEM SWING        --> ")
    print(harris_won)
    print("")
    print("REP SWING        --> ")
    print(trump_won)

from itertools import product
from collections import Counter



if __name__ == '__main__':
    start_time = time.time()
    data = "https://projects.fivethirtyeight.com/polls/data/president_polls.csv"
    df_pivot = construct_import(data)
    MAX = 1
    for i in range(0, MAX):
        num_simulations = random.randint(5000, 9999)
        print("")
        print("ROUND " + str(i + 1) + " OUT OF " + str(MAX))
        print("SIMULATION START --> ")
        df_summary = simulate_elections(df_pivot, num_simulations)
        print(f"Simulation where n =", num_simulations)
        print(df_summary)
        print("SIMULATION END   --> ")
        print("")
    print("##### COCONUT TREE #####")
    # print(coconut_tree(df_pivot, 1000))
    # Print tables
    # print_tables(df_summary, top_n=10)
    print_swing_states_won(df_pivot, swing_states={'North Carolina', 'Pennsylvania', 'Georgia', 'Wisconsin', 'Michigan',
                                                   'Arizona', 'Minnesota', 'Nevada'})
    shapefile_path = 'states/ne_110m_admin_1_states_provinces.shp'
    # generate_electoral_globe(df_pivot, shapefile_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("")
    print(f"TOTAL PROCESS RUNTIME: {elapsed_time:.2f} seconds")
