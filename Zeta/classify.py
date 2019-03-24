import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cosine, canberra
import pickle
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, pairwise_distances, paired_distances, cosine_similarity, manhattan_distances
from sklearn.neighbors import DistanceMetric

def read_dtm_single(input, zeta_res, dtm_sinlge_path, z):

    x_train = []
    for index, row in input.iterrows():

        data = pd.read_csv(dtm_sinlge_path + str(row["folder"]) + "/" + str(row["id"]),
                           sep="\t", names=["word", "count"], skiprows=1, dtype={"word": str, "count": np.int16})
        values = []
        for index1, row1 in data.iterrows():

            if row1["word"] in list(zeta_res.index):
                value = row1["count"] * zeta_res.loc[row1["word"], z]
                values.append(value)

        x_train.append([np.mean(values)])

    return x_train


def classify_base_zeta(zeta_res, focus_train, counter_train, focus_test, counter_test, dtm_sinlge_path):

    scores = {}
    for z in ["sd0", "sd2"]:

        y_train = len(focus_train)*[0]+len(counter_train)*[1]
        y_test = len(focus_test)*[0]+len(counter_test)*[1]

        x_train1 = read_dtm_single(focus_train ,zeta_res, dtm_sinlge_path, z)
        x_train2 = read_dtm_single(counter_train, zeta_res, dtm_sinlge_path, z)
        x_test1 = read_dtm_single(focus_test, zeta_res, dtm_sinlge_path, z)
        x_test2 = read_dtm_single(counter_test, zeta_res, dtm_sinlge_path, z)

        x_train = np.nan_to_num(np.array(x_train1 + x_train2))
        x_test = np.nan_to_num(np.array(x_test1 + x_test2))

        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        scores.update({z: f1_score(y_test, y_pred, average="macro")})

    return scores


def estimate_word_genres_ft_single(filename, model, stoplist, word_fields1, word_fields2, pca, t_mode, metric):
    print(pca)
    data = pd.read_csv(filename, sep="\t", names=["word", "count"], skiprows=1, dtype={"word": str, "count": np.int16})
    stop = open(stoplist, "r").read()
    stop = stop.split("\n")

    data = data[~data["word"].isin(stop)]

    vecs = [model.get_word_vector(str(x)) for x in list(data["word"])]
    if pca != "no_pca":
        
        vecs = pca.transform(vecs)
    vecs = np.nan_to_num(np.array(vecs))
    #word_fields1 = np.nan_to_num(np.array(word_fields1))
    #word_fields2 = np.nan_to_num(np.array(word_fields2))
    if t_mode == True:
        
        if metric == "manhattan":
            dist1 = np.mean(manhattan_distances(vecs, word_fields1), axis=1)
            dist2 = np.mean(manhattan_distances(vecs, word_fields2), axis=1)
            
        if metric == "cosine":
            dist1 = np.mean(cosine_similarity(vecs, word_fields1), axis=1)
            dist2 = np.mean(cosine_similarity(vecs, word_fields2), axis=1)
            
        if metric == "euclidean":
            dist1 = np.mean(euclidean_distances(vecs, word_fields1), axis=1)
            dist2 = np.mean(euclidean_distances(vecs, word_fields2), axis=1)
            
        if metric == "canberra":
            d = DistanceMetric.get_metric("canberra")
            dist1 = np.mean(d.pairwise(vecs, word_fields1), axis=1)
            dist2 = np.mean(d.pairwise(vecs, word_fields2), axis=1)
            
    else:
        
        if metric == "manhattan":
            dist1 = np.min(manhattan_distances(vecs, word_fields1), axis=1)
            dist2 = np.min(manhattan_distances(vecs, word_fields2), axis=1)
            
        if metric == "cosine":
            dist1 = np.min(cosine_similarity(vecs, word_fields1), axis=1)
            dist2 = np.min(cosine_similarity(vecs, word_fields2), axis=1)
            
        if metric == "euclidean":
            dist1 = np.min(euclidean_distances(vecs, word_fields1), axis=1)
            dist2 = np.min(euclidean_distances(vecs, word_fields2), axis=1)
            
        if metric == "canberra":
            d = DistanceMetric.get_metric("canberra")
            dist1 = np.min(d.pairwise(vecs, word_fields1), axis=1)
            dist2 = np.min(d.pairwise(vecs, word_fields2), axis=1)
        
       
    dist = dist1 - dist2
    dist = np.mean(dist)
    dist = np.mean(np.multiply(dist, np.array(data["count"])))

    return [dist]


def estimate_word_genres_ft(input, model, dtm_single_path, word_fields,stoplist, pca, t_mode, metric):

    results = []
    if len(list(word_fields.columns)) < 150:
        word_fields1 = np.array(word_fields[word_fields["category"] == 0].iloc[:, 0:2])
        word_fields2 = np.array(word_fields[word_fields["category"] == 1].iloc[:, 0:2])
    else:
        word_fields1 = np.array(word_fields[word_fields["category"] == 0].iloc[:, 1:301])
        word_fields2 = np.array(word_fields[word_fields["category"] == 1].iloc[:, 1:301])
    
    for index, row in input.iterrows():

        path = dtm_single_path + str(row["folder"]) + "/" + str(row["id"])

        results.append(estimate_word_genres_ft_single(path, model, stoplist, word_fields1, word_fields2, pca, t_mode, metric))

    return results


def classify_fasttext(focus_train, counter_train, focus_test, counter_test, dtm_sinlge_path, model_ft, word_fields, mode, stoplist, pca, t_mode, metric):

    y_train = len(focus_train) * [0] + len(counter_train) * [1]
    y_test = len(focus_test) * [0] + len(counter_test) * [1]


    x_focus_train = estimate_word_genres_ft(focus_train, model_ft, dtm_sinlge_path, word_fields,stoplist, pca, t_mode, metric)
    x_focus_test = estimate_word_genres_ft(focus_test, model_ft, dtm_sinlge_path, word_fields,stoplist, pca, t_mode, metric)
    x_counter_train = estimate_word_genres_ft(counter_train, model_ft, dtm_sinlge_path, word_fields,stoplist, pca, t_mode, metric)
    x_counter_test = estimate_word_genres_ft(counter_test, model_ft, dtm_sinlge_path, word_fields,stoplist, pca, t_mode, metric)


    x_train = np.nan_to_num(np.array(x_focus_train + x_counter_train))
    x_test = np.nan_to_num(np.array(x_focus_test + x_counter_test))

    clf = LogisticRegression(solver='lbfgs')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return {"fasttext_"+str(mode): f1_score(y_test, y_pred, average="macro")}


def estimate_word_genres_w2v(input, model, dtm_single_path, word_fields, stoplist, pca, t_mode, metric):

    x = []
    stop = open(stoplist, "r").read()
    stop = stop.split("\n")

    if len(list(word_fields.columns)) < 150:
        word_fields1 = np.array(word_fields[word_fields["category"] == 0].iloc[:, 0:2])
        word_fields2 = np.array(word_fields[word_fields["category"] == 1].iloc[:, 0:2])
    else:
        word_fields1 = np.array(word_fields[word_fields["category"] == 0].iloc[:, 1:301])
        word_fields2 = np.array(word_fields[word_fields["category"] == 1].iloc[:, 1:301])

    for index, row in input.iterrows():
        vecs = []
        c = []
        data = pd.read_csv(dtm_single_path + str(row["folder"]) + "/" + str(row["id"]),
                           sep="\t", names=["word", "count"], skiprows=1, dtype={"word": str, "count": np.int16})
        data = data[~data["word"].isin(stop)]

        for index1, row1 in data.iterrows():
            try:
                vecs.append(model[str(row1["word"])])
                c.append(int((row1["count"])))
            except KeyError:
                pass
        if pca != "no_pca":
            vecs = pca.transform(vecs)
        vecs = np.nan_to_num(np.array(vecs))
        if t_mode:

            if metric == "manhattan":
                dist1 = np.mean(manhattan_distances(vecs, word_fields1), axis=1)
                dist2 = np.mean(manhattan_distances(vecs, word_fields2), axis=1)

            if metric == "cosine":
                dist1 = np.mean(cosine_similarity(vecs, word_fields1), axis=1)
                dist2 = np.mean(cosine_similarity(vecs, word_fields2), axis=1)

            if metric == "euclidean":
                dist1 = np.mean(euclidean_distances(vecs, word_fields1), axis=1)
                dist2 = np.mean(euclidean_distances(vecs, word_fields2), axis=1)

            if metric == "canberra":
                d = DistanceMetric.get_metric("canberra")
                dist1 = np.mean(d.pairwise(vecs, word_fields1), axis=1)
                dist2 = np.mean(d.pairwise(vecs, word_fields2), axis=1)

        else:

            if metric == "manhattan":
                dist1 = np.min(manhattan_distances(vecs, word_fields1), axis=1)
                dist2 = np.min(manhattan_distances(vecs, word_fields2), axis=1)

            if metric == "cosine":
                dist1 = np.min(cosine_similarity(vecs, word_fields1), axis=1)
                dist2 = np.min(cosine_similarity(vecs, word_fields2), axis=1)

            if metric == "euclidean":
                dist1 = np.min(euclidean_distances(vecs, word_fields1), axis=1)
                dist2 = np.min(euclidean_distances(vecs, word_fields2), axis=1)

            if metric == "canberra":
                d = DistanceMetric.get_metric("canberra")
                dist1 = np.min(d.pairwise(vecs, word_fields1), axis=1)
                dist2 = np.min(d.pairwise(vecs, word_fields2), axis=1)

        dist = dist1 - dist2
        dist = np.mean(dist)
        dist = np.mean(np.multiply(dist, np.array(c)))

        x.append([dist])

    return x


def classify_w2v(focus_train, counter_train, focus_test, counter_test, dtm_sinlge_path, model_w2v, word_fields, mode, stoplist, pca, t_mode, metric):

    y_train = len(focus_train) * [0] + len(counter_train) * [1]
    y_test = len(focus_test) * [0] + len(counter_test) * [1]

    x_train1 = estimate_word_genres_w2v(focus_train, model_w2v, dtm_sinlge_path, word_fields, stoplist, pca, t_mode, metric)
    x_train2 = estimate_word_genres_w2v(counter_train, model_w2v, dtm_sinlge_path, word_fields, stoplist, pca, t_mode, metric)
    x_test1 = estimate_word_genres_w2v(focus_test, model_w2v, dtm_sinlge_path, word_fields, stoplist, pca, t_mode, metric)
    x_test2 = estimate_word_genres_w2v(counter_test, model_w2v, dtm_sinlge_path, word_fields, stoplist, pca, t_mode, metric)

    x_train = np.nan_to_num(np.array(x_train1 + x_train2))
    x_test = np.nan_to_num(np.array(x_test1 + x_test2))

    clf = LogisticRegression(solver='lbfgs')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return {"w2v_"+str(mode): f1_score(y_test, y_pred, average="macro")}
