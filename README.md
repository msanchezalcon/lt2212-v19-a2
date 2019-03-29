# LT2212 V19 Assignment 2

From Asad Sayeed's statistical NLP course at the University of Gothenburg.

My name: MIRIAM SANCHEZ ALCON


## Results and discussion


### Vocabulary restriction.
Top 50 words.
I chose this little amount of data to be able to see noticeable differences, since the corpus is very large in comparison. By restricting the the vocabulary to the top 50, the program uses the most common words, thus making a much bigger difference in the classification task.

### Result table

(in a separate pdf file called "result_table.pdf")


### The hypothesis in your own words
In this assigment we are comparing the similarity among documents in terms of their words. This comparison is done through cosine similarity, which is a value between 0-1. The closer the cosine similarity between two documents is to 1, the more similar they are and therefore should be classificed as same topic, or at least related. The more amount of data is filtered in, the more occurrence of different words, and less of the previous ones (occurrence flattens). By restricting vocabulary size and setting the tf-idf we find the most relevant words to a topic. In the same way, by truncating we reduce the size of vectors with the same consequence.


### Discussion of trends in results in light of the hypothesis
The cosine similarities shows how picking a much smaller amout of data plays a role in the results, being the numbers higher than when choosing full vocabulary.
Applying SVD and tf-idf doesn't affect the results in my table, just reducing the amount of data. I doubt my results are correct because tf-idf should change the results to a much higher cosine similarity, since we are picking the words that are most relevant to the topic.

## Bonus answers

The first issue that comes to my mind is something we mentioned earlier in our NLP course. When classifying and comparing word occurrences is important to have enough context. This means, comparing words that occur together, as in bigrams and trigrams, in order to get a much more accurate result. Words in isolation give very little information in most cases, specially in search engines or machine translation, in which the occurrence of words is based in conditional probability.
Another issue would be not considering the relationship between words belonging to the same stem or root, and therefore are particular for a specific topic in the same way. Therefore, the use of stemming and lemmatization would account for those similarities which oterwhise would be overlooked. On the other hand, considering words that might look similar but don't refer to the same topic would be a mistake, even though their stem might look similar.
In conclusion, taking into account more context in the occurrence of words would make this experiment much more precise and accurate.
