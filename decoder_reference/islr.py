## goal of this file: build out the ISLR class in a similar way to smlr.py + add in slr_learning_var2.m converted to Python


# ---------------------------------------------------------------------

#### my plan of action

## first, i want the shell --> empty class with __init__ and method names that just pass

## then, i want to create the standalone function of slr_learning_var2, checked against the matlab version

## then, i want to fit with the recycling loop

## decision_function, predict_proba, and predict

## integrate into PyDecNef scripts

# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# importing things that were imported in smlr.py
# ---------------------------------------------------------------------


from __future__ import print_function # i actually don't think we need this

import numpy 
import scipy # or this
import scipy.optimize # or this? but tbd
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


def slr_learning_var2(label, X, nlearn=1000, amax=1e8):
    """single sparse logistic regression fit (SLR-VAR).
     Python port of slr_learning_var2.m (placeholder - we need to go in and build this)."""
    
    # this is not inside the class because we don't need to save the objects

# ---------------------------------------------------------------------
# now for reference, looking at what decnef matlab imports (decnef_biclsfy_islrvar.m)
# ---------------------------------------------------------------------
# Required inputs:
#   x_train : [Nsamp_tr, Nfeat]  training data (trials x voxels)
#   t_train : [Nsamp_tr, 1]      training labels
#   x_test  : [Nsamp_te, Nfeat]  test data (trials x voxels)
#   t_test  : [Nsamp_te, 1]      test labels
#             (the original header says [Nsamp_te, Nfeat], which is a typo)
#   niter   : number of recycling iterations (number of base classifiers)
#
# Optional inputs (name : type : allowed values : default):
#   scale_mode  : string  : 'all', 'each', 'stdall', 'stdeach', 'none' : 'each'
#   mean_mode   : string  : 'all', 'each', 'none'                      : 'each'
#   ax0         : real    : initial relevance parameters               : [] (all ones)
#   nlearn      : integer : [1, inf]   SLR updates per base classifier : 1000
#   nstep       : integer : [1, inf]   how often progress is saved     : 100
#   amax        : real    : [0, inf]   voxel pruning threshold         : 1e8
#   usebias     : boolean : add a bias term                            : 1 (True)
#   norm_sep    : boolean : normalize train and test separately        : 0 (False)
#   displaytext : boolean : print progress                             : 1 (True)
#   invhessian  : boolean : which SLR math branch to use               : 0 (False)

# ---------------------------------------------------------------------


#### of these mandatory inputs, what do we need to include?

# we need niter, which is the number of recycling iterations (core of the ISLR setting - no matlab default, so have to choose one or later have fit pick it by CR) --> for clarity's sake, im going to call this outer_iter and set a default of 10
# we need nlearn, which are the SLR updates per base classifier (default 1000) --> for clarity's sake, im going to call this inner_iter
# we need amax, pruning threshold, which controls how sparse each fit is (default 1e8)
# we eed usebiase -->our pipeline expects a bias, default (true)
# to match matlab, we could have scale_mode and mean_mode which default to each --> need these to reproduce matlab normalization when validating; however, if we set these to none, we could rely on PyDecNef's z-scoring instead
# verbose: the python name for displaytext

#### what can we exclude?

# invhessian: chooses between the 2 math branches inside slr_learning_var2 --> iSLR uses invhessian = 0 by default; we can probably just port this one branch, but come back to this to check what the difference in the branches are
# nstep: just controls how often progress is saved
# ax0: initial relevance parameters --> we can just use the defautl of all ones
# norm_sep: this normalizes test and training data separately --> we don't want to do this though because we don't want to normalize test data with its own statistics; we want to normalize test data with training statistics (the model's weights were learned on that scale). notmalizing test data with its own statistics changes the scale , and with a single rela-time volume it would turn every voxel into zero.



class ISLR(BaseEstimator, ClassifierMixin):
    
    """Iterative Sparse Logistic Regression (iSLR) classifier.

    Binary classifier that combines several sparse logistic regression
    (SLR-VAR) fits by "iterative recycling": each iteration trains a new
    SLR on only the voxels not selected by earlier iterations, and the
    final weight vector is the sum of all iterations' weights.

    The API is compatible with scikit-learn classifiers (fit, predict,
    predict_proba, decision_function).

    Parameters:
        outer_iter : int (default 10)
            Number of recycling iterations (number of base classifiers).
            Called niter in the MATLAB version.
        inner_iter : int (default 1000)
            Number of SLR updates per base classifier.
            Called nlearn in the MATLAB version.
        amax : float (default 1e8)
            Pruning threshold. A voxel is dropped once its relevance
            parameter exceeds this value.
        usebias : bool (default True)
            If True, add a bias (intercept) term.
        scale_mode : str (default 'each')
            How to scale features before fitting. 'each' divides each voxel
            by its maximum absolute value in the training data (as in the
            MATLAB version); 'none' skips scaling.
        mean_mode : str (default 'each')
            How to center features before fitting. 'each' subtracts each
            voxel's training mean; 'none' skips centering.
        verbose : bool (default True)
            If True, print progress during fitting.
            Called displaytext in the MATLAB version.

    Attributes (set by fit):
        coef_ : array, shape = [1, n_features]
            Summed weights across all recycling iterations. Zero for voxels
            never selected.
        intercept_ : array, shape = [1]
            Bias term added to the decision function.
        classes_ : array, shape = [2]
            The two labels, in the order used by predict_proba's columns.

    References:
        Hirose S, Nambu I, Naito E (2015). An empirical solution for
        over-pruning with a novel ensemble-learning method for fMRI decoding.
        J Neurosci Methods, 239, 238-245.

        Yamashita O, Sato M, Yoshioka T, Tong F, Kamitani Y (2008). Sparse
        estimation automatically selects voxels relevant for the decoding
        of fMRI activity patterns. NeuroImage, 42(4), 1414-1429.

        Ported from decnef_biclsfy_islrvar.m and slr_learning_var2.m
        (SLR toolbox, Okito Yamashita, ATR CNS; DecNef modifications by
        Hugo Six).
    """
    
    
    def __init__(self, outer_iter = 10, inner_iter = 1000, amax = 1e8, usebias=True,     scale_mode = 'each', mean_mode = 'each', verbose = True):
        self.outer_iter=outer_iter
        self.inner_iter=inner_iter
        self.amax=amax
        self.usebias=usebias
        self.scale_mode=scale_mode
        self.mean_mode=mean_mode
        self.verbose=verbose
        
        
    def fit(self, X, y):
        
        """Fit the iSLR model to the training data.

        (im just putting this as a placeholder - the recycling loop will go here.)
        
        the goal of this is to normalize the data, run the recycling loop calling                  slr_learning_var2 once per outer iteration,  saves the summed weights, bias and              labels on the object.

        Parameters:
            X : array-like, shape = [n_samples, n_features]
                Training data, where each sample is a trial (or volume)
                and each feature is a voxel.
            y : array-like, shape = [n_samples]
                Labels, one per sample (trial).

        Returns:
            self : object
                The fitted model.
        """
        return self

        
    def decision_function(self, X):
        """Compute the score for each sample in X.

        (im just putting this as a placeholder -will compute X @ weights + bias.)
        
        the goal of this is to compute the score for each trial (voxel values x weights,            summed + bias)

        Parameters:
            X : array-like, shape = [n_samples, n_features]
                Data to classify, where each sample is a trial (or volume)
                and each feature is a voxel.

        Returns:
            scores : array, shape = [n_samples]
                Score per sample. Positive values lean toward classes_[1],
                negative values lean toward classes_[0].
        """

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        (im just putting this as a placeholder - will apply the logistic function to the
        scores from decision_function)
        
        the goal of this is to turn each score into probabilities with the logistic                 function, giving one row per trial with the probabiliy of each label --> what               pydecnef uses for neurofeedback

        Parameters:
            X : array-like, shape = [n_samples, n_features]
                Data to classify, where each sample is a trial (or volume)
                and each feature is a voxel.

        Returns:
            P : array, shape = [n_samples, 2]
                Probability of each label per sample. Columns follow the
                order of self.classes_, so P[:, 0] is the probability of
                classes_[0] and P[:, 1] is the probability of classes_[1].
                Each row sums to 1.
        """        
        
    def predict(self, X):
        """Predict class labels for samples in X.

        (im just putting this as a placeholder - will pick the label with he higher                 probability from predict_proba.)
        
        the goal of this is to pick the more likely label for each trial


        Parameters:
            X : array-like, shape = [n_samples, n_features]
                Data to classify, where each sample is a trial (or volume)
                and each feature is a voxel.

        Returns:
            C : array, shape = [n_samples]
                Predicted class label per sample.
        """










