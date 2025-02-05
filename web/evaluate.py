# -*- coding: utf-8 -*-
"""
 Evaluation functions
"""
import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from web.datasets.analogy import fetch_google_analogy, fetch_wordrep, fetch_msr_analogy, fetch_semeval_2012_2
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_MTurk, fetch_RG65, fetch_RW, fetch_TR9856
from web.datasets.categorization import fetch_AP, fetch_battig, fetch_BLESS, fetch_ESSLI_1a, fetch_ESSLI_2b, fetch_ESSLI_2c
from web.analogy import *
import scipy
import pandas as pd

logger = logging.getLogger(__name__)

def calculate_purity(y_true, y_pred):
    """
    Calculate purity for given true and predicted cluster labels.

    Parameters
    ----------
    y_true: array, shape: (n_samples, 1)
      True cluster labels

    y_pred: array, shape: (n_samples, 1)
      Cluster assingment.

    Returns
    -------
    purity: float
      Calculated purity.
    """
    assert len(y_true) == len(y_pred)
    true_clusters = np.zeros(shape=(len(set(y_true)), len(y_true)))
    pred_clusters = np.zeros_like(true_clusters)
    for id, cl in enumerate(set(y_true)):
        true_clusters[id] = (y_true == cl).astype("int")
    for id, cl in enumerate(set(y_pred)):
        pred_clusters[id] = (y_pred == cl).astype("int")

    M = pred_clusters.dot(true_clusters.T)
    return 1. / len(y_true) * np.sum(np.max(M, axis=1))


def evaluate_categorization(mapping, X, y, method="all", seed=None):
    """
    Evaluate embeddings on categorization task.

    Parameters
    ----------
    mapping: Mapping from word to representation

    X: vector, shape: (n_samples, )
      Vector of words.

    y: vector, shape: (n_samples, )
      Vector of cluster assignments.

    method: string, default: "all"
      What method to use. Possible values are "agglomerative", "kmeans", "all.
      If "agglomerative" is passed, method will fit AgglomerativeClustering (with very crude
      hyperparameter tuning to avoid overfitting).
      If "kmeans" is passed, method will fit KMeans.
      In both cases number of clusters is preset to the correct value.

    seed: int, default: None
      Seed passed to KMeans.

    Returns
    -------
    purity: float
      Purity of the best obtained clustering.

    Notes
    -----
    KMedoids method was excluded as empirically didn't improve over KMeans (for categorization
    tasks available in the package).
    """

    assert method in ["all", "kmeans", "agglomerative"], "Uncrecognized method"

    vectors = mapping(X)
    ids = np.random.RandomState(seed).choice(range(len(X)), len(X), replace=False)

    # Evaluate clustering on several hyperparameters of AgglomerativeClustering and
    # KMeans
    best_purity = 0

    if method == "all" or method == "agglomerative":
        best_purity = calculate_purity(y[ids], AgglomerativeClustering(n_clusters=len(set(y)),
                                                                       affinity="euclidean",
                                                                       linkage="ward").fit_predict(vectors[ids]))
        logger.debug("Purity={:.3f} using affinity={} linkage={}".format(best_purity, 'euclidean', 'ward'))
        for affinity in ["cosine", "euclidean"]:
            for linkage in ["average", "complete"]:
                purity = calculate_purity(y[ids], AgglomerativeClustering(n_clusters=len(set(y)),
                                                                          affinity=affinity,
                                                                          linkage=linkage).fit_predict(vectors[ids]))
                logger.debug("Purity={:.3f} using affinity={} linkage={}".format(purity, affinity, linkage))
                best_purity = max(best_purity, purity)

    if method == "all" or method == "kmeans":
        purity = calculate_purity(y[ids], KMeans(random_state=seed, n_init=10, n_clusters=len(set(y))).
                                  fit_predict(vectors[ids]))
        logger.debug("Purity={:.3f} using KMeans".format(purity))
        best_purity = max(purity, best_purity)

    return best_purity


def evaluate_analogy(analogy, X, y, category=None, batch_size=300):
    """
    Simple method to score embedding using SimpleAnalogySolver

    Parameters
    ----------
    predict : Prediction method

    X : array-like, shape (n_samples, 3)
      Analogy questions.

    y : array-like, shape (n_samples, )
      Analogy answers.

    batch_size : int, default: 100
      Increase to increase memory consumption and decrease running time

    category : list, default: None
      Category of each example, if passed function returns accuracy per category
      in addition to the overall performance.
      Analogy datasets have "category" field that can be supplied here.

    Returns
    -------
    result: dict
      Results, where each key is for given category and special empty key "" stores
      summarized accuracy across categories
    """

    assert category is None or len(category) == y.shape[0], "Passed incorrect category list"

    solver = AnalogySolver(analogy=analogy, batch_size=batch_size)
    y_pred = solver.predict(X)

    if category is not None:
        results = OrderedDict({"all": np.mean(y_pred == y)})
        count = OrderedDict({"all": len(y_pred)})
        correct = OrderedDict({"all": np.sum(y_pred == y)})
        for cat in category:
            results[cat] = np.mean(y_pred[category == cat] == y[category == cat])
            count[cat] = np.sum(category == cat)
            correct[cat] = np.sum(y_pred[category == cat] == y[category == cat])

        return pd.concat([pd.Series(results, name="accuracy"),
                          pd.Series(correct, name="correct"),
                          pd.Series(count, name="count")],
                         axis=1)
    else:
        return np.mean(y_pred == y)


def evaluate_similarity(similarity, X, y):
    """
    Calculate Spearman correlation between cosine similarity of the model
    and human rated similarity of word pairs

    Parameters
    ----------
    similarity: Method to measure pairwise similarities.
        Should take a (B x 2) matrix (X) of words and return a B vector of similarities.
        The format is similarity(X[0], X[1]) ~ y.

    X: array, shape: (n_samples, 2)
      Word pairs

    y: vector, shape: (n_samples,)
      Human ratings

    Returns
    -------
    cor: float
      Spearman correlation
    """
    scores = similarity(X)
    return scipy.stats.spearmanr(scores, y).correlation


def get_categorization_tasks():
    return {
        "AP": fetch_AP(),
        "BLESS": fetch_BLESS(),
        "Battig": fetch_battig(),
        "ESSLI_2c": fetch_ESSLI_2c(),
        "ESSLI_2b": fetch_ESSLI_2b(),
        "ESSLI_1a": fetch_ESSLI_1a()
    }


def get_similarity_tasks():
    return {
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "WS353R": fetch_WS353(which="relatedness"),
        "WS353S": fetch_WS353(which="similarity"),
        "SimLex999": fetch_SimLex999(),
        "RW": fetch_RW(),
        "RG65": fetch_RG65(),
        "MTurk": fetch_MTurk(),
#        "TR9856": fetch_TR9856(),  CAN NOT FETCH
    }


def get_analogy_tasks():
    return {
        "Google": fetch_google_analogy(),
        "MSR": fetch_msr_analogy()
    }


def evaluate_tasks_with(tasks, method):
    result = {}
    for name, data in tasks.items():
        result[name] = method(data.X, data.y)
    return result

def evaluate_with(analogy=None, similarity=None, mapping=None):
    """
    Parameters
    ----------
    analogy: Method to answer analogy queries.
        Should take a (B x 3) matrix (X) of words and return a B vector of words (y).
        The format of X is X[0] is to X[1] as X[2] is to y.
        Analogy tests will be skipped if omitted. 

    similarity: Method to measure pairwise similarities.
        Should take a (B x 2) matrix (X) of words and return a B vector of similarities.
        The format is similarity(X[0], X[1]) = y.
        Similarity tests will be skipped if omitted.

    mapping: Method to map words to representations.
        Should take a B vector of words and return a (B x D) matrix of word representations.
        Categorization tests will be skipped if omitted.
    Returns
    -------
    results: pandas.DataFrame
      DataFrame with results, one per column.
    """

    results = {}

    if similarity:
        logger.info("Calculating similarity benchmarks")
        results['similarity'] = evaluate_tasks_with(
            get_similarity_tasks(), 
            lambda X, y: evaluate_similarity(similarity, X, y)
        )

    if analogy:
        logger.info("Calculating analogy benchmarks")
        results['analogy'] = evaluate_tasks_with(
            get_analogy_tasks(), 
            lambda X, y: evaluate_analogy(analogy, X, y)
        )
    
    if mapping:
        logger.info("Calculating categorization benchmarks")
        results['categorization'] = evaluate_tasks_with(
            get_categorization_tasks(),
            lambda X, y: evaluate_categorization(mapping, X, y)
        )

    
    flattened = {}
    for task_category, task_results in results.items():
        for task, res in task_results.items():
            assert(task not in flattened)
            flattened[task] = res

    return flattened


def evaluate_many(method_dictionary):
    """
    Parameters
    ----------
    analogy: Method to answer analogy queries.
        Should take a (B x 3) matrix (X) of words and return a B vector of words (y).
        The format of X is X[0] is to X[1] as X[2] is to y.
        Analogy tests will be skipped if omitted. 

    similarity: Method to measure pairwise similarities.
        Should take a (B x 2) matrix (X) of words and return a B vector of similarities.
        The format is similarity(X[0], X[1]) = y.
        Similarity tests will be skipped if omitted.

    mapping: Method to map words to representations.
        Should take a B vector of words and return a (B x D) matrix of word representations.
        Categorization tests will be skipped if omitted.
    Returns
    -------
    results: pandas.DataFrame
      DataFrame with results, one per column.
    """

    results = {}

    for name, methods in method_dictionary.items():
        results[name] = evaluate_with(
            analogy=methods.get('analogy', None),
            similarity=methods.get('similarity', None), 
            mapping=methods.get('mapping', None)
        )

    return results
