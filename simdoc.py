import os, sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import math

# simdoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here



"""
COSINE SIMILARITY
"""

#    In part 2 we calculate the average Cosine Similarity (CS) in 4 different functions:

# 1. First CS of each vector of the same topic (1. crude and crude and 2. grain and grain)
#    between each other, averaged over the entire topic. Values are 0-1: the closer to 1 the more similar
#    the vectors/documents are (nested for loop).

# 2. Calculate the average CS of each vector of a specific topic compared to every vector of the other topic
#    (3. crude to grain and 4. grain to crude), averaged over the entire topic. First vector cosine similarity
#    to the second vector, first to third, first to fourth, etc. Then the average of all those values.

#    So: All documents to each other in the same topic. Then all documents to each other in the other topic.
#    Then all of topic 1 to all of topic 2 (and vice versa). We should get the same result :)




def dataframe_from_file(vectorfile):
    """
    With the output-file generated from "gendoc.py" we build a dataframe.
    """
    df = pd.read_csv(vectorfile, index=False) # argument is path to output-file

    #cut_df = df.drop("Unnamed: 0", axis=1) # dropping index first column
    return df




def cs_crude(vectorfile):
    """
    Calculates CS of each vector in topic crude between every vector in crude, averaged over entire topic sub-folder.
    """

    # Create a list of results, save each CS result.
    cs_result = []
    # In order to get different topics, we use slices from the main list (557 docs for grain),
    # then build dataframe from those slices (from cut_df) and we don't have to analyze
    # the file, but directly the dataframe:
    # grain_dataframe = cut_df(slice1)
    # crude_dataframe = cut_df(slice2)
    cut_df = dataframe_from_file(vectorfile)

    # Then we do nested for loops with the sliced dataframe.

    # Nested loop for CS for the matrix:
    for index in range(0, 578):
    # Here we get the first vector for getting CS:
        vector1 = cut_df[index:index + 1]

        for inner_index in range(0, 578):
        # Here we get vector 2
            vector2 = cut_df[inner_index:inner_index + 1]
        # Now we do CS between vector 1 and vector 2
            value = cosine_similarity(vector1, vector2, dense_output=True)
            cs_result.append(value[0][0])

    # Average = the CS values of each comparison divided with the sum of all comparisons.
    average_cs_crude = sum(cs_result) / len(cs_result)
    print("Average cosine similarity of topic crude: ")
    print(average_cs_crude)
    return average_cs_crude




def cs_grain(vectorfile):
    """
    Same procedure as in the cs_crude function, but here we calculate the CS of each vector in topic grain
    compared to every vector of topic grain, averaged over the entire topic grain.
    """

    # Create a list of results, save each CS result.
    cs_result = []
    # In order to get different topics, we use slices from the main list (557 docs for grain),
    # then build dataframe from those slices (from cut_df) and we don't have to analyze
    # the file, but directly the dataframe:
    # grain_dataframe = cut_df(slice1)
    # crude_dataframe = cut_df(slice2)
    cut_df = dataframe_from_file(vectorfile)

    # Then we do nested for loops with the sliced dataframe.

    # Nested loop for CS for the matrix:
    for index in range(578, 1160):
    # Here we get the first vector for getting CS:
        vector1 = cut_df[index:index + 1]

        for inner_index in range(578, 1160):
        # Here we get vector 2
            vector2 = cut_df[inner_index:inner_index + 1]
        # Now we do CS between vector 1 and vector 2
            value = cosine_similarity(vector1, vector2, dense_output=True)
            cs_result.append(value[0][0])

    # Average = the CS values of each comparison divided with the sum of all comparisons.
    average_cs_grain = sum(cs_result) / len(cs_result)
    print("Average cosine similarity of topic grain: ")
    print(average_cs_grain)
    return average_cs_grain




def cs_crude_to_grain(vectorfile):
    """
    We compute the average CS of each vector in crude compared to every vector
    in grain, averaged over the entire topic.
    """
    cs_result = []
    cut_df = dataframe_from_file(vectorfile)

    for index in range(0, 578):
        vector1 = cut_df[index:index+1]

        for innerindex in range(578, 1160):
            vector2 = cut_df[innerindex:innerindex+1]
        # Now we do CS between vectors from crude and vectors from grain
            value = cosine_similarity(vector1, vector2, dense_output=True)
            cs_result.append(value[0][0])

    average_cs_crude_to_grain = sum(cs_result) / len(cs_result)
    print("Average cosine similarity of topic crude compared to grain: ")
    print(average_cs_crude_to_grain)
    return average_cs_crude_to_grain





def cs_grain_to_crude(vectorfile):
    """
    We compute the average CS of each vector in grain compared to every vector
    in crude, averaged over the entire topic.
    """
    cs_result = []
    cut_df = dataframe_from_file(vectorfile)

    for index in range(578, 1160):
        vector1 = cut_df[index:index + 1]

        for innerindex in range(0, 578):
            vector2 = cut_df[innerindex:innerindex + 1]
            # Now we do CS between vectors from crude and vectors from grain
            value = cosine_similarity(vector1, vector2, dense_output=True)
            cs_result.append(value[0][0])

    average_cs_grain_to_crude = sum(cs_result) / len(cs_result)
    print("Average cosine similarity of topic grain compared to crude: ")
    print(average_cs_grain_to_crude)
    return average_cs_grain_to_crude





parser = argparse.ArgumentParser(description="Compute some similarity statistics.")
parser.add_argument("vectorfile", type=str,
                    help="The name of the input file for the matrix data.")

args = parser.parse_args()

print("Reading matrix from {}.".format(args.vectorfile))


dataframe_from_file(args.vectorfile)
cs_crude(args.vectorfile)
cs_grain(args.vectorfile)
cs_crude_to_grain(args.vectorfile)
cs_grain_to_crude(args.vectorfile)
