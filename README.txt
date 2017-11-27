This model has been built using Multinomial Naive Bayes Classifier. First the data from the training text file is extracted.  The labels are extracted and stripped of relations i.e. (e1,e2) or (e2,e1). Then it is encoded using a LabelEncoder. All the words from e1 to e2 in a sentence are considered for training the classifier. The textual data is converted into numerical data using 'Bag of Words', where the words and their ngrams become the features. The words are also lemmatized and stemmed for better prediction. The test data is used to test the accuracy of the classifier which is around 72.5%. The predicted labels are stored in a file named "output.txt".

Libraries required are-

nltk.punkt
nltk.wordnet
scikit-learn (sklearn)
beautifulsoup4 (bs4)
numpy
 

