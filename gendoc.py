import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

# gendoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here

parser = argparse.ArgumentParser(description="Generate term-document matrix.")
parser.add_argument("-T", "--tfidf", action="store_true", help="Apply tf-idf to the matrix.")
parser.add_argument("-S", "--svd", metavar="N", dest="svddims", type=int,
                    default=None,
                    help="Use TruncatedSVD to truncate to N dimensions")
parser.add_argument("-B", "--base-vocab", metavar="M", dest="basedims",
                    type=int, default=None,
                    help="Use the top M dims from the raw counts before further processing")
parser.add_argument("foldername", type=str,
                    help="The base folder name containing the two topic subfolders.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the matrix data.")

args = parser.parse_args()

print("Loading data from directory {}.".format(args.foldername))

if not args.basedims:
    print("Using full vocabulary.")
else:
    print("Using only top {} terms by raw count.".format(args.basedims))

if args.tfidf:
    print("Applying tf-idf to raw counts.")

if args.svddims:
    print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))

# THERE ARE SOME ERROR CONDITIONS YOU MAY HAVE TO HANDLE WITH CONTRADICTORY
# PARAMETERS.

print("Writing matrix to {}.".format(args.outputfile))

"""
Notes on pandas
"""
# df = pd.read_csv('ransom.csv') read csv files. we can print the dataframe print(df)
# inspecting data frame df.head()
# df.info()
# selects a column in the file (suspect): suspect = credit_records['suspect'] or credit_records.suspect
# select rows: you access by index like in lists



#Opening the files, preprocessing the text. Lowercase, strip punctuation.
def vocabulary_list(directory, m=None): # add arg.foldername in the main when we call this function in the main bc the argument name directory is arbitrary
    """
    Returns vocabulary list containing all words in all documents in every
    subfolder
    """
    vocab_list = []
    for subfolder in os.listdir(directory):
        path_to_subfolder = os.path.join(directory, subfolder)
        for file in os.listdir(path_to_subfolder):
            path_to_file = os.path.join(path_to_subfolder, file)
            with open(path_to_file, "r", encoding="utf8") as f:
                text = f.read()
                remove_punctuation = re.sub(r"[^\w\s\d]", "", text).lower()
                clean_words = remove_punctuation.split(" ")
                for word in clean_words:
                    if word not in vocab_list:
                        vocab_list.append(word)
    return vocab_list



def preprocessing_and_labels(directory, m=None):
    """
    Creates a main dictionary with the topic + document name as key and
    subdictionaries with words as keys and word counts values as values
    """
    label_maker = {} # main dict
    for topic in os.listdir(directory):
        path_to_subfolder = os.path.join(directory, topic) # we join the path from main folder to every subfolder and subsequently to every text doc inside them
        vocab = vocabulary_list(directory, m)
        vocab_dict = dict.fromkeys(vocab, 0) # this makes a dict from every word from vocab list and count of 0 to start with

        for file in os.listdir(path_to_subfolder):
            path_to_file = os.path.join(path_to_subfolder, file)
            counts = vocab_dict.copy() # smaller dict

            with open(path_to_file, "r", encoding="utf8") as f:
                text = f.read()
                remove_punctuation =  re.sub(r"[^\w\s\d]", "", text).lower()
                clean_words = remove_punctuation.split(" ")
                for word in vocab:
                    if word not in counts:
                        counts[word] = 1
                    else:
                        counts[word] += 1
            label_maker[topic+" "+file] = counts # joining the name of subfolder and the doc in one single name and make it the key
    return label_maker



def vector_builder(directory, m=None):
    """
    Create a vector or lists of appended values/counts from previously created
    dictionary.
    Every vector (little list) will be converted into an array and then into
    a dataframe from pandas.
    """
    # this main vector list will be later converted into a np.array() which is basically a list of lists
    bigdict = preprocessing_and_labels(directory, m) # contains all smaller dictionaries which are each doc

    for label in bigdict.keys():
        main_vector_list = []
        for word, count in bigdict[label].items():
            main_vector_list.append(count)
        bigdict[label] = main_vector_list


    return bigdict



#create a dataframe from the dictionaries?
#OR convert the dictionaries into vectors and every vector goes into an array
#then make a dataframe from the array
def matrix_builder(directory, m=None):
    """
    Convert main_vector_list into an array by calling np.array() and then
    convert this array into a dataframe from pandas.
    """
    main_vector = vector_builder(directory, m)
    # matrix_array = np.array(main_vector)
    columns = vocabulary_list(directory, m)
    matrix_dataframe = pd.DataFrame.from_dict(main_vector, orient='index',  dtype=None, columns=columns)

    return matrix_dataframe
    print(matrix_dataframe)





#def cosine_similarity_same_topic():



#def consine_similarity_other_topic():


#import pdb;pdb.set_trace()
vocabulary_list(args.foldername, args.basedims)
preprocessing_and_labels(args.foldername, args.basedims)
vector_builder(args.foldername, args.basedims)
matrix_builder(args.foldername, args.basedims)
