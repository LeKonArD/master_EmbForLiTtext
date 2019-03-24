import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation, MeanShift, dbscan, DBSCAN, Birch
import umap

def bend_embedding_space(metric, word_fields):
    if len(word_fields) > 300:
        comp = 300
    else:
        comp = len(word_fields)-1

    transformer = umap.UMAP(n_components = 2,metric=metric)
    transformer.fit(np.array(word_fields.iloc[:,1:301]), y=word_fields["category"])
    vecs = transformer.transform(word_fields.iloc[:,1:301])
    pca_res = pd.DataFrame(vecs)
    pca_res["category"] = word_fields["category"]
    
    return pca_res, transformer

def bend_embedding_space2(metric, word_fields):

    transformer = umap.UMAP(n_components = 300,metric=metric)
    transformer.fit(word_fields.iloc[:,1:302], y=word_fields["category"])
    vecs = transformer.transform(word_fields.iloc[:,1:302])
    pca_res = pd.DataFrame(vecs)
    pca_res["category"] = word_fields["category"]
    
    return pca_res, transformer

    
def get_ft_field(zeta_res, model, zeta_scope, mode, ft_fields, meth):

    if mode == 0:
        words = zeta_res.index[zeta_res[zeta_scope] > 0]
    else:
        words = zeta_res.index[zeta_res[zeta_scope] < 0]

    vecs = [model.get_word_vector(str(x)) for x in words]
    word_matrix = np.matrix(vecs)

    if meth == "MS":
        clu = MeanShift(n_jobs=-1)
        
    if meth == "AP":
        if mode == 0:
            clu = AffinityPropagation(preference=zeta_res[zeta_scope][zeta_res[zeta_scope] > 0])
        else:
            clu = AffinityPropagation(preference=zeta_res[zeta_scope][zeta_res[zeta_scope] < 0])
        
    if meth == "Birch":
        clu = Birch(n_clusters=None)
        
    clu.fit_predict(word_matrix)
    try:
        cluster_frame1 = pd.DataFrame(clu.cluster_centers_)
    except:
        cluster_frame1 =  pd.DataFrame(clu.subcluster_centers_)
    
    cluster_frame1["Category"] = mode

    ft_fields.put(cluster_frame1)
    ft_fields.close()


def get_ft_fields(zeta_res, model, zeta_scope, meth):

    
    ratio = len(zeta_res.index[zeta_res[zeta_scope] > 0]) / len(zeta_res.index[zeta_res[zeta_scope] < 0])
    print(ratio)
    words = zeta_res.index[zeta_res[zeta_scope] > 0]
    vecs = [model.get_word_vector(str(x)) for x in words]
    word_matrix = np.matrix(vecs)
    
    if meth == "MS":
        clu = MeanShift(bandwidth=1, n_jobs=-1)       
    if meth == "AP":
        clu = AffinityPropagation(preference=zeta_res[zeta_scope][zeta_res[zeta_scope] > 0])
    if meth == "Birch":
        clu = Birch(n_clusters=None)
        

    clu.fit_predict(word_matrix)
    try:
        cluster_frame1 = pd.DataFrame(clu.cluster_centers_)
    except:
        cluster_frame1 =  pd.DataFrame(clu.subcluster_centers_)
        
    cluster_frame1["category"] = 0

    words = zeta_res.index[zeta_res[zeta_scope] < 0]
    vecs = [model.get_word_vector(str(x)) for x in words]
    word_matrix = np.matrix(vecs)
    
    if meth == "MS":
        clu = MeanShift(bandwidth=1, n_jobs=-1)       
    if meth == "AP":
        clu = AffinityPropagation(preference=zeta_res[zeta_scope][zeta_res[zeta_scope] < 0])
    if meth == "Birch":
        clu = Birch(n_clusters=None)

    clu.fit_predict(word_matrix)
    try:
        cluster_frame2 = pd.DataFrame(clu.cluster_centers_)
    except:
        cluster_frame2 =  pd.DataFrame(clu.subcluster_centers_)
   
    cluster_frame2["category"] = 1

    cluster_frame = pd.concat([cluster_frame1, cluster_frame2]).reset_index()

    return cluster_frame


def get_w2v_field(zeta_res, model, zeta_scope, mode):

    if mode == 0:
        words = zeta_res.index[zeta_res[zeta_scope] > 0]
    else:
        words = zeta_res.index[zeta_res[zeta_scope] < 0]

    vecs = []
    for word in words:
        try:
            vecs.append(model[word])

        except KeyError:
            pass

    word_matrix = np.matrix(vecs)
    if mode == 0:
        clu = AffinityPropagation(preference=zeta_res[zeta_scope][zeta_res[zeta_scope] > 0])
    else:
        clu = AffinityPropagation(preference=zeta_res[zeta_scope][zeta_res[zeta_scope] < 0])
    clu.fit_predict(word_matrix)
    cluster_frame = pd.DataFrame(clu.cluster_centers_)
    cluster_frame["Category"] = mode

    return cluster_frame


def get_w2v_fields(zeta_res, model, zeta_scope, meth):

    ratio = len(zeta_res.index[zeta_res[zeta_scope] > 0]) / len(zeta_res.index[zeta_res[zeta_scope] < 0])
    print(ratio)
    words = zeta_res.index[zeta_res[zeta_scope] > 0]
    vecs = []
    for word in words:
        try:
            vecs.append(model[word])
        except:
            pass
    word_matrix = np.matrix(vecs)
    
    if meth == "MS":
        clu = MeanShift(bandwidth=1, n_jobs=-1)       
    if meth == "AP":
        clu = AffinityPropagation(preference=zeta_res[zeta_scope][zeta_res[zeta_scope] > 0])
    if meth == "Birch":
        clu = Birch(n_clusters=None)
        

    clu.fit_predict(word_matrix)
    try:
        cluster_frame1 = pd.DataFrame(clu.cluster_centers_)
    except:
        cluster_frame1 =  pd.DataFrame(clu.subcluster_centers_)
        
    cluster_frame1["category"] = 0

    words = zeta_res.index[zeta_res[zeta_scope] < 0]
    vecs = []
    for word in words:
        try:
            vecs.append(model[word])
        except:
            pass
    word_matrix = np.matrix(vecs)
    
    if meth == "MS":
        clu = MeanShift(bandwidth=1, n_jobs=-1)       
    if meth == "AP":
        clu = AffinityPropagation(preference=zeta_res[zeta_scope][zeta_res[zeta_scope] < 0])
    if meth == "Birch":
        clu = Birch(n_clusters=None)

    clu.fit_predict(word_matrix)
    try:
        cluster_frame2 = pd.DataFrame(clu.cluster_centers_)
    except:
        cluster_frame2 =  pd.DataFrame(clu.subcluster_centers_)
   
    cluster_frame2["category"] = 1

    cluster_frame = pd.concat([cluster_frame1, cluster_frame2]).reset_index()

    return cluster_frame
    return cluster_frame
