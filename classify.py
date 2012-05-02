#!/usr/bin/env python
from __future__ import division
import os, re, json, time
from glob import glob
from collections import defaultdict
from operator import itemgetter
from copy import deepcopy
from random import shuffle
from math import ceil, log
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import pylab as pl

__author__ = 'Chris Beard'
__email__ = 'wbeard3@gatech.edu'

#================
#Helper functions
#================

def sbk6(d, rev=False):
    """
    Sort dictionary by key...'fastest' way
    http://writeonly.wordpress.com/2008/08/30/sorting-dictionaries-by-value-in-python-improved/
    """
    return sorted(d.iteritems(), key=itemgetter(0), reverse=rev)

def posh(tag):
    """
    Wrapper for interactive POS tag help
    """
    return nltk.help.upenn_tagset(tag.upper())

def full_name(f, folder):
    """
    Returns full name of a file f located in folder
    """
    return os.path.abspath(os.path.join(folder, f))


#===================
#Decorator Functions
#===================

def timeit(method):
# from http://www.zopyx.com/blog/a-python-decorator-for-measuring-the-execution-time-of-methods"
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%s() %2.2f sec\n' % (method.__name__, te-ts)
        return result

    return timed

def read_and_write(meth):
    """Decorator function"""

    def wrapper(*args, **kwargs):
        """
        Wrapper function returned by read_and_write decorator. Takes target
        method's args as args (including filename), opens file, extracts data,
        calls target method on the data, then writes the data to new processed
        file with '_pcd.txt' suffix.
        ...don't use this in current version
        """
        data, fname, args = '', args[-1], list(args)
        fname_out = fname[:-4] + '_pcd.txt'
        if os.path.isfile(fname):
            with open(fname) as f:
                data = f.read()
        else:
            print('"%s" is not a file!' % fname)

        args[-1] = data
        print 'Calling function "{0:>s}"'.format(meth.__name__)
        data_output = meth(*args, **kwargs)
        print 'Now going to do something with output as part of the wrapper function. I.e., write to file.'
        with open(fname_out, 'w') as f:
            f.write(data_output)

    return wrapper


#============================
#Feature Processing Functions
#============================

exceptions = set('vb vbd vbg vbn vbp vbz jj jjr jjs nn nns nnp nnps cd'.upper().split())

def feat_form_dict(para, f=lambda x: x):
    """
    Takes a [paragraphs of (word, POS_tag)'s], and returns a
    dict of tag-words and counts
    Tag-word is in form 'POS-word': {'the-DT': 3, ...}
    """
    d = defaultdict(lambda: 0)
    l = WordNetLemmatizer()
    f = l.lemmatize
    # p = PorterStemmer()
    # f = p.stem
    for word, tag in para:
        feature = (f(word.lower()) if tag not in exceptions else '') + '-' + tag
        d[feature] += 1
    return dict(d)

def normalize(l, ln=2):
    """Assuming only positive numbers. Only works with future division."""
    norm = sum([i**ln for i in l]) ** (1/ln)
    return [i/norm for i in l]


#============
#Data classes
#============

class clf_data(object):
    """
    Container to separate and hold training and test data.
    __init__ function by default splits the data into
    10%% test set and 90%% training set (shuffled)
    """
    def __init__(self, vec, split=.1):
        """
        vec: [(label, [data_vector]), (lab2, [data2_vec])]
        """
        self.vec = deepcopy(vec)
        shuffle(self.vec)
        self.n = len(vec)
        labels, vectors = zip(*self.vec)
        if split:
            cutoff = int(ceil(self.n * split))
            self.train_dat = vectors[:-cutoff]
            self.test_dat  = vectors[-cutoff:]
            self.train_lab = labels[:-cutoff]
            self.test_lab  = labels[-cutoff:]
        else:
            self.train_dat = vectors
            self.test_dat  = []
            # Coauthored papers will have labels, but they'll be ambiguous and wrong...
            self.train_lab = labels
            self.test_lab  = []


class data_builder(object):
    
    def __init__(self, foldername, filename=None):
        """
        Initialize with either the name of a folder or a file to extract the data from.
        If initializing with filename from a different directory, call with both folder
        and filename as separate args.
        """
        super(data_builder, self).__init__()
        self.folder = foldername if os.path.isdir(foldername) else os.path.dirname(foldername)
        self.fname =  [f for f in glob(self.folder + '/*_pcd.txt')] if not filename else [os.path.join(self
        .folder, filename)]
        self.pos_data = {}  # {author: [[par1== (word1, POS), ...], [par2...], ...]}
        self.master_tags = {}  # comprehensive dict with every seen tag-word; each val==0
        self.tag_word_vecs = {}  # {author: [(par1==0,1,0,1), (par2), ...], author2:...}
        self.author_ind = {}  # {author1: 1, author2: 2...}
        self.vec_list = []  # [(lab1, [vec1]), (lab2, ...]
        self.extract_data()


    @timeit
    def extract_data(self, files=None):
        """
        Gets POS data from initialized file list. Stores list of ['word', 'POS']
        pairs as values of self.pos_data, with the authors as keys
        """
        print 'Extracting data from files...'
        files = (files,) if files else [f for f in self.fname]
        print files

        for full_fname in files:
            fname = os.path.basename(full_fname)
            print ' ' * 4 + 'Opening file', fname
            with open(full_fname) as f:
                f_data = f.read()
            author_name = fname[:-4] #  assuming it's .txt
            self.author_ind[author_name] = len(self.author_ind) + 1
            pos_fname = os.path.join(self.folder, author_name + '_pos.txt')

            if os.path.exists(pos_fname):
                with open(pos_fname) as f:
                    pos_words = json.loads(f.read())
            else:
                # POS tagging is the most time consuming part, so best to write and read
                # after the first time
                f_data = re.split(r'[\n\r]{2,}', f_data)
                pos_words = [nltk.pos_tag(re.findall(r"\b\w+\b", f)) for f in f_data if f]
                with open(pos_fname, 'w') as f:
                    json.dump(pos_words, f)
            self.pos_data[author_name] = pos_words


    @timeit
    def pos_vectorize(self):
        """
        Create vectors from the POS tagging data. This will be input into the SVM.
        It will first be necessary to make a master dict with all entries, and then
        go one by one through the authors and paragraphs, turning them into sparse
        vectors.
        """
        if not self.pos_data:
            print 'No tagged data to vectorize'
            return

        #Build master tag dict
        master_tags = dict()
        for author in self.pos_data:
            print ' Adding info from author "%s" into master tag dict' % author
            for para in self.pos_data[author]:
                master_tags.update(feat_form_dict(para))
        self.master_tags = {key:0 for key in master_tags}

        #Process individual paragraphs
        st = time.time()
        for author in self.pos_data:
            self.tag_word_vecs[author] = []
            print ' Processing tag vectors for author "%s"' % author
            vec_fname = os.path.join(self.folder, author + '_svec.txt')

            for para in self.pos_data[author]:
                para_dict = self.master_tags.copy()
                author_pos_dict = feat_form_dict(para)
                types = len(author_pos_dict)  # Number of tagword types: used for relative freq.
                para_dict.update(author_pos_dict)
                data_vector = [v for (k,v) in sbk6(para_dict)]  #raw word count
                # data_vector = [v/types for (k,v) in sbk6(para_dict)]  #relative freq
                # data_vector = [log(1.+v/types) for (k,v) in sbk6(para_dict)]  #logRel freq
                data_vector = normalize(data_vector, 2)
                self.tag_word_vecs[author].append(data_vector)

            print 'Dumping to', vec_fname  # just to be able to see what's inside
            with open(vec_fname, 'w') as f:
                json.dump(self.tag_word_vecs[author], f)

        print 'Labeling vectors'
        for auth in self.tag_word_vecs:
            print 'Adding vec for', auth
            for vec in self.tag_word_vecs[auth]:
                self.vec_list.append( (self.author_ind[auth], vec) )
        tt = time.time() - st
        print 'Completed in %2.5f seconds' % tt

    # @timeit
    # @read_and_write
    # def pre_process(self, data):
    #     """
    #     Preprocesses given file in filename. Should be called with filename, data arguments.
    #     """
    #     # Turn all digits to '0'
    #     data = re.sub(r"(\S?\d+\S?)", '0', data)
    #     # Only keep words of length 2+
    #     data = re.findall(r"\b\w+\b", data)
    #     data = ' '.join(data)
    #     return data


def batch_test(vec_list, iters=1, target=1):
    """"
    Takes vector list, performs subsampling cross-validation.
    """
    Prec, Rec, fscores = [], [], []
    num_correct, total_count = 0, 0
    global Y, y_, cm
    Y = []  # y-labels
    y_ = []  # predicted labels

    for _ in range(iters):
        dat = clf_data(vec_list)
        clf = LinearSVC()
        clf = clf.fit(dat.train_dat, dat.train_lab)

        pred = clf.predict(dat.test_dat)
        pos = [1 if dat.test_lab[i]==target else 0 for i in range(len(pred))]
        guess_pos = [1 if pred[i]==target else 0 for i in range(len(pred))]
        tpos = [1 if (dat.test_lab[i]==target and pred[i]==target) else 0 for i in range(len(pred))]
        prec, rec = sum(tpos) / (sum(guess_pos) + 1e-10), sum(tpos) / (sum(pos) + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)

        Y.extend(dat.test_lab)
        y_.extend(pred)
        fscores.append(f1)
        Prec.append(prec)
        Rec.append(rec)
#        print 'Precision: %2.2f, Recall: %2.2f, Fscore: %2.2f' % (prec, rec, f1)
    print 'Overall Fscore: %2.4f' % (sum(fscores) / len(fscores))
    print 'Overall Precision: %2.4f' % (sum(Prec) / len(Prec))
    print 'Overall Recall: %2.4f' % (sum(Rec) / len(Rec))
    cm = confusion_matrix(Y, y_)
    # pl.matshow(cm)
    # pl.title('Confusion matrix')
    # pl.colorbar()
    print cm
    print ''
    return Y, y_, cm

@timeit
def subs_xval(myclass, iters=10):
    """
    Subsampling cross-validation iters times, over all the authors in
    a data_builder class instance.
    """
    Yauth = []
    yauth = []
    for author in myclass.author_ind:
        if author == 'starner_pcd': continue
        print '%s:' % author[:-4]
        Y, y, _ = batch_test(myclass.vec_list, iters, myclass.author_ind[author])
        Yauth.extend(Y)
        yauth.extend(y)
    return Yauth, yauth, confusion_matrix(Yauth, yauth)

@timeit
def batch_coauth(myclass, iters=1):
    """
    Pseudo-percentage testing for coauthored papers
    """
    coauth = [a for a in myclass.author_ind if a[:5] == 'comb_'][0]
    auths = [a for a in myclass.author_ind if a != coauth]
    auths_labs = [myclass.author_ind[a] for a in auths]  # numerical labels for each author: i.e., [2, 3]
    label_map = {lab:auths_labs.index(lab) for lab in auths_labs}  # maps author labels to [0,1]

    print coauth
    print auths_labs
    print auths
    print label_map

    test_vec = [(lab, vec) for lab, vec in myclass.vec_list if lab == myclass.author_ind[coauth]]
    train_vec = [(lab, vec) for lab, vec in myclass.vec_list if lab in auths_labs]

    Pred = [0] * len(test_vec)  # Prediction list, init'd to 0's
    test_dat = clf_data(test_vec, 0)

    for _ in range(iters):
        train_data = clf_data(train_vec, .5)
        clf = LinearSVC()
        clf = clf.fit(train_data.train_dat, train_data.train_lab)

        # won't have labels, so we won't be able to get fscores
        pred = clf.predict(test_dat.train_dat)
        pred = [label_map[a] for a in pred]  #predictions will be for author 0, or author 1
        Pred = [i+j for i,j in zip(pred, Pred)]  #running sum for each classifier

    print "Done looking at coauthors"
    return Pred


@timeit
def main():
    global myclass, dat, clf, pred, coauth_class
    myclass = data_builder("test")
    myclass.pos_vectorize()
    Y, y_, cm = subs_xval(myclass, 1)

    # Confusion matrix
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    for i, cas in enumerate(cm):
        for j, c in enumerate(cas):
            if c>0:
                pl.text(j-.2, i+.2, c, fontsize=14)

    pl.xlabel('Actual author')
    pl.ylabel('Predicted author')
    pl.savefig('conf_mat.pdf')
    # coauth_class = data_builder('test', 'comb_StarnerAshbrook.txt')
    # coauth_class.pos_vectorize()
    # coauth_class = data_builder('test', 'comb_StarnerLee.txt')
    # coauth_class.pos_vectorize()
    # coauth_class = data_builder('test', 'comb_starnerMann.txt')
    # coauth_class.pos_vectorize()


if __name__ == '__main__':
    main()

