import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def cluster(cohort_submissions: dict) -> list:
    """
    Splits given dict into clusters of 4 based on their ranked complexity

    The 'remainder problem' of needing to have 4 submissions per cluster,
    regardless of number of submissions, is solved here by duplicating
    submission IDs across clusters to supplement any clusters with less than 4,
    with no clusters containing more than 1 submission_ID that also
    appears in another cluster, unless there are fewer than 3 total
    submissions, or exactly 5 submissions, in the given cohort.

    Input: dictionary of a single cohort containing nested dictionary
    with 'submission_id' as first level key,
    and 'complexity' as one of the inner keys
    Output: JSON object of clusters:
    {1: [list of submission_ids], 2: [list of submission_ids]}
    """

    # Generate DataFrame from dict
    df = pd.DataFrame.from_dict(cohort_submissions, orient="index")

    #Reset index to put ID column first
    df = df.reset_index()
    df = df.rename(columns={'index': 'ID'})

    # Empty dictionary to store the groupings
    groups = {}

    # Instantiate scaler
    scaler = StandardScaler()

    # Pull out features
    features = df.drop(df.columns[0], axis=1)

    # Scale data
    norm_x = scaler.fit_transform(features)

    # Turn into df
    df_norm_x = pd.DataFrame(norm_x)

    # Instantiate model - groups of 4
    nn = NearestNeighbors(n_neighbors=4, algorithm='kd_tree')

    # Counter to use as key for groups in dictionary
    counter = 1

    # Grab a copy of the df before taking it apart to deal with the remainder problem
    df_copy = df

    # While loop that takes the top user and creates a group with the its three closest users
    # Drops grouped users and continues until there are less than 12 users left to group
    # Remainder problem will be dealt with after the while loop runs
    while len(df_norm_x) >4:
        # Fit the nearest neighbors model
        nn.fit(df_norm_x)

        # Find nearest neighbors
        array_2 = nn.kneighbors([df_norm_x.iloc[0].values], return_distance=False)

        # Put story_id list into groups dictionary
        groups[counter] = [df[df.columns[0]][item] for item in array_2[0]]

        # Increment the counter
        counter += 1

        # Drop the users you have already grouped
        # From both df's that you are using
        df_norm_x = df_norm_x.drop(array_2[0])
        df = df.drop(array_2[0])

        # Reset the index
        # For both datasets that you are using
        df_norm_x.reset_index(inplace= True, drop= True)
        df.reset_index(inplace= True, drop= True)

    # Drop the remainders from copy of df to find most similar
    for i in range(len(df)):
        df_copy = df_copy[df_copy[df_copy.columns[0]] != int(df.iloc[i][0])]

    # Do preproccesing done above
    df_copy.reset_index(inplace=True, drop=True)
    features_copy = df_copy.drop(df.columns[0], axis=1)
    norm_x_copy = scaler.fit_transform(features_copy)
    df_norm_x_copy = pd.DataFrame(norm_x_copy)
    
    # Fit to KNN model
    nn.fit(df_norm_x_copy)
    
    # TODO: Finish remainder problem, going to get it working with perfect numbers first

    #Make dictionary into JSON object to be passed back to the web team
    json_groups = json.dumps(groups, default=numpy_convert)

    return json_groups

def numpy_convert(o):
    """
    Input: Dictionary containing numpy int64 type integars to be converted to python int64
    for the purpose of making a numpy object
    Output: Dictionary with only python int64 type for a json object
    """
    if isinstance(o, np.int64): return int(o)  
    raise TypeError


async def batch_cluster(submissions: dict) -> json:
    """
    Generates a return JSON object of clusters for all provided cohorts.

    Input: dictionary of all cohort submissions
    Output: JSON object of nested lists of submission IDs by cluster, by cohort

    To test locally in isolation as an async function, run the following code:
    import asyncio
    asyncio.run(batch_cluster(submissions_json))
    """

    # Initiate cluster dictionary
    cluster_dict = {}

    # Iterate through cohorts to get clusters, and
    # add each to cluster_dict
    for cohort_id in submissions:
        clusters = cluster(submissions[cohort_id])
        cluster_dict[cohort_id] = clusters

    # Convert dict back to JSON
    cluster_json = json.dumps(cluster_dict)

    return cluster_json
