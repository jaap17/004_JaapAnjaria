import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]


sentences = []
labels = []
with open("bbc-text-small.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
            sentence = sentence.replace("  ", " ")
        sentences.append(sentence)


print(len(sentences))
print(sentences[48])
print(labels[48])


import pandas as pd

dataset = pd.read_csv('bbc-text-small.csv')

dataset.head()

labels = dataset.category
data = dataset.text

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

#print(tokenizer.word_counts)


#Example 2
sentences = [
    'At DDU, we are Learning ML in semester 7 !!!',
    'I love ml subject',
    'Prof Brijesh Bhatt and prof. Hariom Pandya are faculties for the Ml subject',
    'Do you think ML is amazing?'
]

##########################################

# Try with words that the tokenizer wasn't fit to
test_data = [
    'i really love CC subject',
    'Do you think BDA and IP are amazing subjects?'
]

###############################################


#Observations: subject/subjects, ML/mL/ml, i/I  prof/Prof.  !!!
tokenizer2 = Tokenizer()
tokenizer2.fit_on_texts(sentences)

tokenizer2.word_counts
print(tokenizer2.texts_to_sequences(test_data))

