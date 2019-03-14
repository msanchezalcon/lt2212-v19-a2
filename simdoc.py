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
COSINE SIMILARITY (CS)
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



# We add new arguments for whatever two sets of data we choose to compare (for CS purposes).
# In this assignment each set of data will correspond to the vector files from crude and grain:
parser = argparse.ArgumentParser(description="Compute some similarity statistics.")
parser.add_argument("first_vectorfile", type=str,
                    help="The name of the input file for the matrix data.")
parser.add_argument("second_vectorfile", type=str,
                    help="The name of the input file for the matrix data.")

args = parser.parse_args()

# We read the data from both text data files:
vec_1 = np.loadtxt(args.first_vectorfile, dtype='i', delimiter=',')
vec_2 = np.loadtxt(args.second_vectorfile, dtype='i', delimiter=',')


def CS(data1, data2=None):
    """
    Calculates CS of each vector in every topic with those of their same topic and those of a different topic,
    averaged over all topics.
    """
    cs = []
    for index, v1 in enumerate(data1):
        if data2 is not None:
            for v2 in data2:
                cs.append(cosine_similarity([v1], [v2]))
        else:
            for v2 in data1[index + 1:]:
                cs.append(cosine_similarity([v1], [v2]))
    average_cs = sum(cs)/len(cs)
    return average_cs


# We run the previous CS function to compute all 4 results: between same topics and between the different ones:
#1
first_topic_same = CS(vec_1)
second_topic_same = CS(vec_2)
#2
first_to_second = CS(vec_1, vec_2)
second_to_first =CS(vec_2, vec_1)



# We print all results:
print("Reading matrix from {}.".format(args.first_vectorfile))
print("Reading matrix from {}.".format(args.second_vectorfile))

#
print("Average cosine similarity within same topic, topic 1: ", first_topic_same)

print("Average cosine similarity within same topic, topic 2: ", second_topic_same)

print("Average cosine similarity to other topic (topic 1 to topic 2): ", first_to_second)
print("Average cosine similarity to other topic (topic 2 to topic 1): ", second_to_first)
