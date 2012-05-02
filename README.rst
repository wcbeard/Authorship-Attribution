The file 'classify.py' defines a class 'data_builder' which initializes with a folder name, and extracts text from all the pre-processed text files in the folder that end with '_pcd.txt.' Calling the 'pos_vectorize' method will then extract the tagword features from the already extracted text and turn them into vectors for the SVM. The vectors will then be stored in the class variable 'vec_list.'

The 'clf_data' is a class that acts as a container for the data after the data has been gathered in a data_builder. On initializing, it splits the data from vec_list into training and test data. 

The 'subs_xval' function takes as argument a data_builder and integer 'iters', automatically converts it to a clf_data container, and performs subsample cross-validation 'iters' times. It prints the Fscore, precision and recall. The main function does this as is.

This file uses LinearSVC from the 3rd party `scikits learn module`_, and POS tagger, WordNetLemmatizer and PorterStemmer from the NLTK (natural language toolkit).

.. _`scikits learn module`: http://scikit-learn.org/stable/>