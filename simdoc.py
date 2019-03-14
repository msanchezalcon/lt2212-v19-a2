import os, sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import math

# simdoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here
parser = argparse.ArgumentParser(description="Compute some similarity statistics.")
parser.add_argument("vectorfile", type=str,
                    help="The name of the input  file for the matrix data.")

args = parser.parse_args()

print("Reading matrix from {}.".format(args.vectorfile))

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
    With the outputfile generated from "gendoc.py" we build a dataframe.
    """
    df = pd.read_csv(vectorfile) # argument is path to outputfile
    drop_df = df.drop(df.columns[0], axis=1)
    print(df.columns[0])
    array = np.array(drop_df)

    #cut_df = df.drop("Unnamed: 0", axis=1) # dropping index first column
    return array


def selecting_input_data(vectorfile):
    """
    Separating crude files from grain files
    """
    crude_files = []
    grain_files = []
    together = []
    df = pd.read_csv(vectorfile)
    for label in df[df.columns[0]]:
        together.append(label)
        if "crude" in label:
            crude_files.append(label)
        else:
            grain_files.append(label)

    return len(crude_files), len(grain_files), len(together)


def cs_crude(whatever_array, len_crude):
    """
    Calculates CS of each vector in topic crude between every vector in crude, averaged over entire topic sub-folder.
    """

    # Create a list of results, save each CS result.
    cs_result = []


    # Nested loop for CS for the matrix:
    for index in range(0, len_crude):
    # Here we get the first vector for getting CS:
        vector1 = whatever_array[index:index + 1]

        for inner_index in range(0, len_crude):
        # Here we get vector 2
            vector2 = whatever_array[inner_index:inner_index + 1]
        # Now we do CS between vector 1 and vector 2
            value = cosine_similarity(vector1, vector2, dense_output=True)
            cs_result.append(value[0][0])

    # Average = the CS values of each comparison divided with the sum of all comparisons.
    average_cs_crude = sum(cs_result) / len(cs_result)
    print("Average cosine similarity of topic crude: ")
    print(average_cs_crude)
    return average_cs_crude




def cs_grain(whatever_array, len_grain, total_len):
    """
    Same procedure as in the cs_crude function, but here we calculate the CS of each vector in topic grain
    compared to every vector of topic grain, averaged over the entire topic grain.
    """
    cs_result = []
  
    for index in range(len_grain, total_len):
        vector1 = whatever_array[index:index + 1]

        for inner_index in range(len_grain, total_len):
            vector2 = whatever_array[inner_index:inner_index + 1]
            value = cosine_similarity(vector1, vector2, dense_output=True)
            cs_result.append(value[0][0])

    average_cs_grain = sum(cs_result) / len(cs_result)
    print("Average cosine similarity of topic grain: ")
    print(average_cs_grain)
    return average_cs_grain




def cs_crude_to_grain(whatever_array, len_crude, len_grain, total_len):
    """
    We compute the average CS of each vector in crude compared to every vector
    in grain, averaged over the entire topic.
    """
    cs_result = []

    for index in range(0, len_crude):
        vector1 = whatever_array[index:index+1]

        for innerindex in range(len_grain, total_len):
            vector2 = whatever_array[innerindex:innerindex+1]
        # Now we do CS between vectors from crude and vectors from grain
            value = cosine_similarity(vector1, vector2, dense_output=True)
            cs_result.append(value[0][0])

    average_cs_crude_to_grain = sum(cs_result) / len(cs_result)
    print("Average cosine similarity of topic crude compared to grain: ")
    print(average_cs_crude_to_grain)
    return average_cs_crude_to_grain





def cs_grain_to_crude(whatever_array, len_grain, len_crude, total_len):
    """
    We compute the average CS of each vector in grain compared to every vector
    in crude, averaged over the entire topic.
    """
    cs_result = []

    for index in range(len_grain, total_len):
        vector1 = whatever_array[index:index + 1]

        for innerindex in range(0, len_crude):
            vector2 = whatever_array[innerindex:innerindex + 1]
            # Now we do CS between vectors from crude and vectors from grain
            value = cosine_similarity(vector1, vector2, dense_output=True)
            cs_result.append(value[0][0])

    average_cs_grain_to_crude = sum(cs_result) / len(cs_result)
    print("Average cosine similarity of topic grain compared to crude: ")
    print(average_cs_grain_to_crude)
    return average_cs_grain_to_crude







whatever_array = dataframe_from_file(args.vectorfile)
len_crude, len_grain, total_len = selecting_input_data(args.vectorfile)
cs_crude(whatever_array, len_crude)
cs_grain(whatever_array, len_grain, total_len)
cs_crude_to_grain(whatever_array, len_crude, len_grain, total_len)
cs_grain_to_crude(whatever_array, len_crude, len_grain, total_len)
