import csv
import random

import pandas
from fpdf import FPDF
import json
import datetime
from scipy.stats import cauchy
from scipy.stats import logistic


import sys
from scipy.special import expit

import pandas as pd
import sys
import numpy as np
from scipy.stats import norm
import time
import openai

import geopandas as gpd
import plotly.express as px
import pandas as pd

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import firebase_admin
from firebase_admin import credentials, db

# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.expand_frame_repr', False)  # Avoid line breaks for large frames
pd.set_option('display.float_format', '{:.4f}'.format)

pollster_set = {}

# Unfortunately coconut pilled, as the kids say
coconut_pilled = True

def construct_import(data, safe_state_data):
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

    safe_state_df = pd.read_csv(safe_state_data)
    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv(data)
    print("\033[92m---> DATA IMPORTED SUCCESSFULLY\033[0m")
    df = pd.concat([df, safe_state_df], axis=0, ignore_index=True)
    # print(df)
    df_filt = df[columns_to_keep].copy()

    michigan_polls = df_filt[df_filt['state'] == 'Nevada']
    print(michigan_polls)

    ## add DC dummy data :eyeroll:
    # df_filt = df_filt._append(dc_poll_df, ignore_index=True)
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

    last_poll = df_filt.sort_values(by='end_date', ascending=False).iloc[0]

    # Extract the category, pollster, and end_date from the most recent poll
    category = last_poll['state']
    pollster = last_poll['pollster']
    end_date = last_poll['end_date']

    # Print the results
    print(f"     STATE: \t {category}")
    print(f"     POLLSTER: \t {pollster}")
    print(f"     END DATE: \t {end_date}")

    if coconut_pilled:
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

    # mean_diff_mean = df_pivot['mean_diff'].mean()
    # mean_diff_std = df_pivot['mean_diff'].std()

    # Calculate win probability based on the margin (using a normal distribution assumption)
    # df_pivot['win_percent_harris'] = df_pivot['mean_diff'].apply(lambda x: norm.cdf(x) * 100)
    # df_pivot['win_percent_harris'] = df_pivot['mean_diff'].apply( lambda x: expit((x - mean_diff_mean) / mean_diff_std) * 100)
    # Calculate the standard deviation from the data
    std_dev = df_pivot['mean_diff'].std()

    # Apply the norm.cdf with the calculated standard deviation
    df_pivot['win_percent_harris'] = df_pivot['mean_diff'].apply(lambda x: norm.cdf(x, 0, 4.5)) * 100

    # df_pivot['win_percent_harris'] = df_pivot['mean_diff'].apply(lambda x: cauchy.cdf(x, loc=0, scale=4)) * 100

    # df_pivot['win_percent_harris'] = df_pivot['mean_diff'].apply(lambda x: logistic.cdf(x, loc=0)) * 100

    df_pivot['win_percent_trump'] = (1 - df_pivot['win_percent_harris'] / 100) * 100

    # Determine the leading candidate in each state
    idx_max = df_weighted_avg.groupby('state')['weighted_avg_pct'].idxmax()
    df_winner = df_weighted_avg.loc[idx_max]

    # Map electoral votes to the winning candidate in each state
    df_winner['electoral_votes'] = df_winner['state'].map(electoral_votes)

    # Calculate total electoral votes per candidate
    electoral_tally = df_pivot.groupby('winner')['electoral_votes'].sum().reset_index()

    # electoral_tally = df_winner.groupby('candidate_name')['electoral_votes'].sum().reset_index()
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


# def calculate_weights(df):
#     print("---> CALCULATING WEIGHTS")
#     print("")
#     # Default sample size if missing
#     df['sample_size'] = df['sample_size'].fillna(500)
#     df['numeric_grade'] = df['numeric_grade'].fillna(1)
#
#     # Convert end_date to datetime if it's not already
#     df['end_date'] = pd.to_datetime(df['end_date'])
#
#     # Calculate days since the earliest end_date for weighting
#     min_date = df['end_date'].min()
#     df['days_since'] = (df['end_date'] - min_date).dt.days
#
#     # Normalize days_since to use as a weight (more recent dates get higher weights)
#     max_days = df['days_since'].max()
#     df['recency_weight'] = df['days_since'] / max_days
#
#     # Pollster counts to calculate base weight for each pollster (inverse of frequency)
#     pollster_counts = df['pollster'].value_counts()
#     df['weight'] = df['pollster'].map(pollster_counts).apply(lambda x: 1 / x)
#
#     # Adjust weight for small or large sample sizes
#     df['weight'] *= df['sample_size'] / df['sample_size'].mean()
#
#     # Adjust weight based on numeric_grade, assuming higher grade (3) means higher trustworthiness
#     df['weight'] *= df['numeric_grade'] / df['numeric_grade'].mean()
#
#     # Incorporate recency weight
#     df['weight'] *= df['recency_weight']
#
#     return df

def calculate_weights(df):
    print("---> CALCULATING WEIGHTS")
    print("")
    # Default sample size if missing
    df['sample_size'] = df['sample_size'].fillna(250)
    df['numeric_grade'] = df['numeric_grade'].fillna(1.5)

    # Convert end_date to datetime if it's not already
    df['end_date'] = pd.to_datetime(df['end_date'])

    # Calculate days since the earliest end_date for weighting
    reference_date = df['end_date'].min()
    df['days_from_reference'] = (reference_date - df['end_date']).dt.days

    decay_factor = 0.00125
    # decay_factor = 1
    df['recency_weight'] = np.exp(-decay_factor * df['days_from_reference'])

    df['weight'] = df['recency_weight']

    # Pollster counts to calculate base weight for each pollster (inverse of frequency)
    # pollster_counts = df['pollster'].value_counts()
    # df['weight'] = df['pollster'].map(pollster_counts).apply(lambda x: 1 / x)

    # Adjust weight for small or large sample sizes
    df['weight'] *= df['sample_size'] / df['sample_size'].mean()

    # Adjust weight based on numeric_grade, assuming higher grade (3) means higher trustworthiness
    # df['weight'] *= df['numeric_grade'] / df['numeric_grade'].mean()

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


# def generate_electoral_globe(df_pivot, shapefile_path):
#     pass
#     # # Load US state boundaries from the downloaded shapefile
#     # usa_states = gpd.read_file(shapefile_path)
#     #
#     # # Filter to include only mainland US states
#     # usa_states = usa_states[usa_states['admin'] == 'United States of America'].copy()
#     #
#     # # Ensure both dataframes have matching state names
#     # df_pivot['state'] = df_pivot['state'].replace({
#     #     'Georgia': 'Georgia',
#     #     'Michigan': 'Michigan',
#     #     'Wisconsin': 'Wisconsin',
#     #     'North Carolina': 'North Carolina',
#     #     'Pennsylvania': 'Pennsylvania',
#     #     'Arizona': 'Arizona',
#     #     'Nevada': 'Nevada',
#     #     'Virginia': 'Virginia'
#     #     # Add other state mappings if necessary
#     # })
#     #
#     # # Merge the df_pivot data with the usa_states GeoDataFrame on the state name
#     # electoral_map = usa_states.merge(df_pivot, left_on='name', right_on='state', how='left')
#     #
#     # # Set up the color mapping based on the winner column
#     # electoral_map['winner_color'] = electoral_map['winner'].map({
#     #     'Kamala Harris': 'blue',
#     #     'Donald Trump': 'red'
#     # })
#     #
#     # # Set up the projection for the globe
#     # projection = ccrs.Orthographic(central_longitude=-100, central_latitude=45)  # Adjust as needed for the US
#     #
#     # # Plot the electoral map on a globe projection
#     # fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': projection})
#     #
#     # # Add coastlines and country borders for context
#     # ax.add_feature(cfeature.COASTLINE)
#     # ax.add_feature(cfeature.BORDERS)
#     #
#     # # Plot states, coloring them by the winner
#     # for idx, state in electoral_map.iterrows():
#     #     ax.add_geometries([state['geometry']], crs=ccrs.PlateCarree(),
#     #                       facecolor=state['winner_color'], edgecolor='black')
#     #
#     # # Set the title and display the map
#     # plt.title('2024 US Electoral Map on a Globe', fontsize=18)
#     # plt.show()

def generate_electoral_globe(df_pivot, shapefile_path):
    # Load US state boundaries from the downloaded shapefile
    usa_states = gpd.read_file(shapefile_path)

    # Filter to include only mainland US states
    usa_states = usa_states[usa_states['admin'] == 'United States of America'].copy()

    # Ensure both dataframes have matching state names
    df_pivot['state'] = df_pivot['state'].replace({
        'Georgia': 'Georgia',
        'Michigan': 'Michigan',
        'Wisconsin': 'Wisconsin',
        'North Carolina': 'North Carolina',
        'Pennsylvania': 'Pennsylvania',
        'Arizona': 'Arizona',
        'Nevada': 'Nevada',
        'Virginia': 'Virginia'
        # Add other state mappings if necessary
    })

    # Merge the df_pivot data with the usa_states GeoDataFrame on the state name
    electoral_map = usa_states.merge(df_pivot, left_on='name', right_on='state', how='left')

    # Add a color column for the winner
    electoral_map['winner_color'] = electoral_map['winner'].map({
        'Kamala Harris': 'blue',
        'Donald Trump': 'red'
    })

    # Convert GeoDataFrame to GeoJSON format for Plotly
    geojson_data = electoral_map.__geo_interface__

    # Create an interactive choropleth map with Plotly
    fig = px.choropleth_mapbox(
        electoral_map,
        geojson=geojson_data,
        locations='state',                # Column in df_pivot to match with the geojson 'state'
        featureidkey='properties.name',   # The state names in the GeoDataFrame
        color='winner',                   # Color states by the winner
        hover_name='state',               # Display state name on hover
        hover_data={
            'state': True,                # Show state names
            'win_percent_harris': True,   # Show win percentage for Kamala Harris on hover
            'win_percent_trump': True,    # Show win percentage for Donald Trump on hover
            'winner': True                # Show winner on hover
        },
        color_discrete_map={
            'Kamala Harris': 'blue',
            'Donald Trump': 'red'
        },
        center={"lat": 37.0902, "lon": -95.7129},  # Center on the US
        mapbox_style="carto-positron",             # Map style
        zoom=3,                                   # Initial zoom level
        opacity=0.6                               # Set opacity for better visibility
    )

    # Set title and layout
    fig.update_layout(
        title_text='2024 US Electoral Map with Win Percentages',
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    # Show the figure
    fig.show()




def simulate_elections(df_pivot, num_simulations=1000):
    # Initialize results list
    results = []
    harris_swing_state_wins = {state: 0 for state in
                               ['Pennsylvania', 'Wisconsin', 'Arizona', 'Georgia', 'Michigan', 'Nevada',
                                'North Carolina']}
    trump_swing_state_wins = {state: 0 for state in
                              ['Pennsylvania', 'Wisconsin', 'Arizona', 'Georgia', 'Michigan', 'Nevada',
                               'North Carolina']}

    # Define the threshold to win the election
    WINNING_THRESHOLD = 270

    # Run simulations
    z = 0
    for _ in range(num_simulations):
        z = z + 1
        # print("SIMULATION: " + str(z))
        # sys.stdout.write("\033[F")
        # Initialize electoral votes for candidates
        electoral_votes = {
            'Kamala Harris': 0,
            'Donald Trump': 0
        }

        # Simulate the outcome of each state
        for _, row in df_pivot.iterrows():
            if np.random.rand() < row['win_percent_harris'] / 100:
                electoral_votes['Kamala Harris'] += row['electoral_votes']
                if row['state'] in harris_swing_state_wins:
                    harris_swing_state_wins[row['state']] += 1
            else:
                electoral_votes['Donald Trump'] += row['electoral_votes']
                if row['state'] in trump_swing_state_wins:
                    trump_swing_state_wins[row['state']] += 1

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

    final_harris_swing_wins = [state for state, count in harris_swing_state_wins.items() if count > num_simulations / 2]
    final_trump_swing_wins = [state for state, count in trump_swing_state_wins.items() if count > num_simulations / 2]

    summary_table = pd.DataFrame({
        'Candidate': ['Donald Trump', 'Kamala Harris'],
        'Swing States Won': [final_trump_swing_wins, final_harris_swing_wins]
    })

    df_results_summary = pd.concat([df_results_summary, summary_table], axis=1)
    df_results_summary = df_results_summary.drop('Candidate', axis=1)

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


def add_final_electoral_results(df_pivot, df_summary):
    # Calculate total electoral votes for Harris and Trump
    total_harris_votes = df_pivot[df_pivot['winner'] == 'Kamala Harris']['electoral_votes'].sum()
    total_trump_votes = df_pivot[df_pivot['winner'] == 'Donald Trump']['electoral_votes'].sum()

    # Get the probabilities from df_summary (simulated outcomes)
    harris_percent = df_summary.loc[df_summary['Outcome'] == 'Kamala Harris', 'Percentage'].values[0]
    trump_percent = df_summary.loc[df_summary['Outcome'] == 'Donald Trump', 'Percentage'].values[0]

    # print("##### OUTCOMES")
    # print(harris_percent)
    # Determine the final winner only if the percent difference is greater than 5%
    if abs(harris_percent - trump_percent) > 3:
        final_winner = 'Kamala Harris' if harris_percent > trump_percent else 'Donald Trump'
    else:
        final_winner = 'tie'

    # Create the row for the final electoral results
    # date = datetime.datetime.now().strftime('%Y%m%d')
    date = datetime.datetime.now().strftime('%Y%m%d')
    electoral_results = {
        'state': 'Electoral',
        'electoral_votes': 0,
        'Donald Trump': total_trump_votes,
        'Kamala Harris': total_harris_votes,
        'win_percent_harris': harris_percent,
        'win_percent_trump': trump_percent,
        'mean_diff': 0,
        'winner': final_winner,
        'date': date
    }

    try:
        historical_df = pd.read_csv("historical_data.csv")
    except FileNotFoundError:
        historical_df = pd.DataFrame(columns=['date', 'percent_win_harris', 'percent_win_trump'])

    current_date = datetime.datetime.now().strftime('%m-%d')
    new_row = {'date': current_date, 'percent_win_harris': harris_percent, 'percent_win_trump': trump_percent}

    historical_df = historical_df._append(new_row, ignore_index=True)
    historical_df.to_csv("historical_data.csv", index=False)


    # print(electoral_results)

    # Append the final results to the DataFrame
    electoral_df = pd.DataFrame([electoral_results])

    # Append the final results to the DataFrame
    df_pivot = pd.concat([df_pivot, electoral_df], ignore_index=True)

   #  df_pivot = df_pivot._append(electoral_results, ignore_index=True)

    return df_pivot

def historical():
    # Read the historical CSV file
    csv_file = 'historical_data.csv'
    df = pd.read_csv(csv_file)

    # Initialize the JSON structure for historical data
    historical_data = {"Historical": {}}

    # Loop through the DataFrame and build the historical JSON structure
    for index, row in df.iterrows():
        date = row['date']
        harris_percent = float(row['percent_win_harris'])  # Convert float to int if needed
        trump_percent = float(row['percent_win_trump'])  # Convert float to int if needed

        # Create an entry for the date
        historical_data["Historical"][date] = {
            "Harris": harris_percent,
            "Trump": trump_percent
        }

    return historical_data

def df_pivot_to_json(df_pivot):
    result_dict = {}

    # Iterate through each row of df_pivot
    for _, row in df_pivot.iterrows():
        state = row['state']

        # Create a dictionary for each state's data
        state_data = {
            "Donald Trump": row['Donald Trump'],
            "Kamala Harris": row['Kamala Harris'],
            "electoral_votes": row['electoral_votes'],
            "mean_diff": row['mean_diff'],
            "win_percent_harris": row['win_percent_harris'],
            "win_percent_trump": row['win_percent_trump'],
            "winner": row['winner']
        }

        # Add the state's data to the result dictionary
        result_dict[state] = state_data

    # Convert the result dictionary to a JSON string
    json_output = json.dumps(result_dict, indent=2)

    # Optionally save to a file
    with open('electoral_results.json', 'w') as json_file:
        json_file.write(json_output)

    return json_output

from itertools import product
from collections import Counter

if __name__ == '__main__':
    cred = credentials.Certificate('metrix-d84a6-firebase-adminsdk-mzl7o-370ba512ad.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://metrix-d84a6-default-rtdb.firebaseio.com/'
    })

    ref = db.reference('')

    try:
        data = "https://projects.fivethirtyeight.com/polls/data/president_polls.csv"
        safe_data = "safe_state_data.csv"
        df_pivot = construct_import(data, safe_data)
        df_pivot.to_csv('df_pivot.csv', index=False)
        MAX = 1
        avg_df_summary = pandas.DataFrame
        for i in range(0, MAX):
            num_simulations = random.randint(100000, 100000)
            # print("Loading" + ": " + str(i))
            print("")
            print("ROUND " + str(i + 1) + " OUT OF " + str(MAX))
            print("SIMULATION START --> ")
            df_summary = simulate_elections(df_pivot, num_simulations)
            print(f"Simulation where n =", num_simulations)
            print(df_summary)
            avg_df_summary = df_summary
            print("SIMULATION END   --> ")
            print("")
            # Cursor up one line

        new_pivot = add_final_electoral_results(df_pivot, avg_df_summary)
        json_output = df_pivot_to_json(new_pivot)
        # print("##### COCONUT TREE #####")
        # print(coconut_tree(df_pivot, 1000))
        # Print tables
        # print_tables(df_summary, top_n=10)
        print_swing_states_won(df_pivot, swing_states={'North Carolina', 'Pennsylvania', 'Georgia', 'Wisconsin', 'Michigan',
                                                       'Arizona', 'Michigan', 'Nevada'})
        shapefile_path = 'states/ne_110m_admin_1_states_provinces.shp'
        # generate_electoral_globe(df_pivot, shapefile_path)
        end_time = time.time()
        # elapsed_time = end_time - start_time
        print("")
        # print(f"TOTAL PROCESS RUNTIME: {elapsed_time:.2f} seconds")
        print("Sample JSON")
        historical_data = historical()
        print(json_output)

        # with open('x.json', 'w') as json_file:
        #     json.dump(json_output, json_file, indent=4)
        date = datetime.datetime.now().strftime('%Y%m%d')
        # Create the filename with the date
        filename = f'database/x_{date}.json'

        with open(filename, 'w') as f:
            f.write(json_output)

        with open(filename, 'r') as json_file:
            data = json.load(json_file)
            # print("unslay")
            # data["Electoral"]["date"] = datetime.datetime.now().strftime("%B %d, %Y")
            data["Electoral"]["date"] = datetime.datetime.now().strftime("%B %d, %Y %I:%M %p")
            data['Historical'] = historical_data['Historical']
            print(data)
            # print(data)
            # print("Historical data:")
            # print(historical_data)
            # print(data)

        with open("database/last_output.json", 'w') as j:
            q = json.dumps(data, indent=4)
            j.write(q)

        x = input("WRITE TO FIREBASE? Y/N")
        if x.upper() == "Y":
            print("Writing to Firebase")
            ref.set(data)
            print("Operation successful")
        else:
            print("")
            print("DID NOT WRITE TO FIREBASE")

        print(df_summary)
        exit(0)


    except Exception as e:
        print("ERROR: ---------> " + (str(e)))
        exit(1)