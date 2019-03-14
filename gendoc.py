import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
import re
import nltk
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
# gendoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here


"""
Notes on pandas
"""
# df = pd.read_csv('ransom.csv') read csv files. we can print the dataframe print(df)
# inspecting data frame df.head()
# df.info()
# selects a column in the file (suspect): suspect = credit_records['suspect'] or credit_records.suspect
# select rows: you access by index like in lists



"""
Part 1
"""

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
                remove_punctuation = re.sub(r"/[^\s]+", "", text).lower()
                clean_words = remove_punctuation.split()
                for word in clean_words:
                    vocab_list.append(word)
    vocab_dict = dict(nltk.FreqDist(vocab_list))
    s=[(k,vocab_dict[k]) for k in sorted(vocab_dict, key=vocab_dict.get,reverse=True)]
    if m:
        s = s[:m]

    return [k for k,v in s]

def preprocessing(filenames):
    preprocess_text = []
    for f in filenames:
        file = open(f, "r", encoding = "utf8")
        text = file.read()
        remove_punctuation = re.sub(r"/[^\s]+", "", text).lower()
        preprocess_text.append(remove_punctuation)
    return preprocess_text


def preprocessing_and_labels(directory, m=None):
    """
    Creates a main dictionary with the topic + document name as key and
    subdictionaries with words as keys and word counts values as values
    """
    label_maker = {} # main dict
    vocab = vocabulary_list(directory, m)
    vocab_dict = dict.fromkeys(vocab, 0)
    for topic in os.listdir(directory):
        path_to_subfolder = os.path.join(directory, topic) # we join the path from main folder to every subfolder and subsequently to every text doc inside them
        # this makes a dict from every word from vocab list and count of 0 to start with

        for file in os.listdir(path_to_subfolder):
            path_to_file = os.path.join(path_to_subfolder, file)
            counts = vocab_dict.copy() # smaller dict

            with open(path_to_file, "r", encoding="utf8") as f:
                text = f.read()
                remove_punctuation = re.sub(r"[^\w\s\d]", "", text).lower()
                clean_words = remove_punctuation.split()
                for word in clean_words:
                    if word in vocab:
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



def matrix_builder(directory, m=None):
    """
    Convert main_vector (which is a dictionary) into a dataframe from pandas.
    """
    main_vector = vector_builder(directory, m)
    # matrix_array = np.array(main_vector)
    columns = vocabulary_list(directory, m) # this is a lists with all words
    matrix_dataframe = pd.DataFrame.from_dict(main_vector, orient='index',  dtype=None, columns=columns)
    duplicates_list = matrix_dataframe[matrix_dataframe.duplicated()].index.tolist()
    matrix_dataframe = matrix_dataframe.drop_duplicates()
    print("These duplicated vectors have been dropped: ")
    for duplicate in duplicates_list:
        print(duplicate)
    return matrix_dataframe, main_vector, columns


def file_creator(dataframe):
    """
    Prints dataframe into a new file
    """
    dataframe.to_csv(args.outputfile, header=True)
    print(dataframe)




"""
Part 2
"""
#def cosine_similarity_same_topic():



#def cosine_similarity_other_topic():


#import pdb;pdb.set_trace()


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
documents = []
for topic in os.listdir(args.foldername):
    path_to_subfolder = os.path.join(args.foldername,
                                     topic)  # we join the path from main folder to every subfolder and subsequently to every text doc inside them
    # this makes a dict from every word from vocab list and count of 0 to start with

    for file in os.listdir(path_to_subfolder):
        documents.append(os.path.join(path_to_subfolder, file))
files_data = preprocessing(documents)
dataframe, main_vector, columns = matrix_builder(args.foldername, args.basedims) # dataframes are excel files. you can output info very easily
main_array = np.array(list(main_vector.values()),dtype=float)
main_labels = [x for x in main_vector.keys()]
file_creator(dataframe)
print("Writing matrix to {}.".format(args.outputfile))

print("Loading data from directory {}.".format(args.foldername))

if not args.basedims:
    print("Using full vocabulary.")
else:
    print("Using only top {} terms by raw count.".format(args.basedims))

if args.tfidf:
    print("Applying tf-idf to raw counts.")
    tfidfconverter = TfidfVectorizer(max_features = args.basedims)
    tfidfvector = tfidfconverter.fit_transform(files_data).toarray()
    tfidf_df = pd.DataFrame(data=tfidfvector,
                            index= [x for x in documents],
                            columns = tfidfconverter.get_feature_names())
    tfidf_df.to_csv("output.csv", encoding="utf8")

if args.svddims:
    print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))
    svd = TruncatedSVD(n_components=args.svddims)
    svdT = svd.fit_transform(main_array)
    svdT_df = pd.DataFrame(data=svdT,
                           index = main_labels,
                           columns = [i for i in range(0,args.svddims)])
    svdT_df.to_csv("svdt_countvector.csv", encoding="utf-8")
    if args.tfidf:
        svd = TruncatedSVD(n_components=args.svddims)
        svdT = svd.fit_transform(tfidfvector)
        svdT_df = pd.DataFrame(data=svdT,
                               index=[x for x in documents],
                               columns=[i for i in range(0, args.svddims)])
        svdT_df.to_csv("svdt_tfidf.csv", encoding="utf-8")
# THERE ARE SOME ERROR CONDITIONS YOU MAY HAVE TO HANDLE WITH CONTRADICTORY
# PARAMETERS.
