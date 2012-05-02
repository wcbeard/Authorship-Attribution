===============================
Authorship Attribution with SVM
===============================

*Final project for CS 6601: Artificial Intelligence*

This project contains a procedure which takes text files (whose filename is named after the author), and learns the author's style, paragraph by paragraph, in order to make predictions on unseen paragraphs.

The file ``classify.py`` defines a class ``data_builder`` which initializes with a folder name, and extracts text from all the pre-processed text files in the folder that end with ``_pcd.txt.`` Calling the ``pos_vectorize()`` method will then extract the tagword features from the already extracted text and turn them into vectors for the SVM. The vectors will then be stored in the class variable ``vec_list.``

The ``clf_data`` is a class that acts as a container for the data to be classified, after the data has been gathered in a data_builder. On initializing, it splits the data from ``vec_list`` into training and test data. 

The ``subs_xval`` function takes as argument a ``data_builder`` instance and integer ``iters``, automatically converts it to a ``clf_data`` container, and performs subsample cross-validation ``iters`` times. It prints the Fscore, precision and recall.

This project uses `scikit learn`_ module's SVM, and POS tagger, WordNetLemmatizer and PorterStemmer from the NLTK_ (natural language toolkit), and makes plots with matplotlib_.

The script ``plotter.py`` plots histograms of my existing dataset, and imports a list of the authorship probabilities for each paragraph.

.. _`scikit learn`: http://scikit-learn.org/stable/
.. _NLTK: http://www.nltk.org/
.. _matplotlib: http://matplotlib.sourceforge.net/