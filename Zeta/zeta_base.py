import pandas as pd
import numpy as np


def unify_segments(seg_a, seg_b):

    full_data = pd.concat([seg_a, seg_b], axis=1, sort=False).fillna(0)
    seg_a = full_data.iloc[:,: len(seg_a.columns)]
    seg_b = full_data.iloc[:, len(seg_a.columns):]

    return seg_a, seg_b


def get_segements(meta_focus_train, meta_counter_train, dtm_path, seg_size, stopword_list):


    chunks_focus_train = segmenter(meta_focus_train, dtm_path, seg_size, stopword_list)
    chunks_counter_train = segmenter(meta_counter_train, dtm_path, seg_size, stopword_list)

    chunks_focus_train = chunks_focus_train[(chunks_focus_train.T > 3).any()]
    chunks_counter_train = chunks_counter_train[(chunks_counter_train.T > 3).any()]

    chunks_focus_train, chunks_counter_train = unify_segments(chunks_focus_train, chunks_counter_train)

    bina_focus_train = chunks_focus_train.applymap(lambda x: bina(x)).mean(axis=1)
    bina_counter_train = chunks_counter_train.applymap(lambda x: bina(x)).mean(axis=1)

    rel_focus_train = (chunks_focus_train.div(chunks_focus_train.sum(axis=1), axis=0))
    rel_counter_train = (chunks_counter_train.div(chunks_counter_train.sum(axis=1), axis=0))

    return bina_focus_train, bina_counter_train,\
           rel_focus_train, rel_counter_train, \
           chunks_focus_train, chunks_counter_train,


def segmenter(meta, dtm_path, seg_size, stopword_list):

    chunks = pd.DataFrame()
    counter = 0
    threshold = int(seg_size / 250)

    for index, row in meta.iterrows():

        data = pd.read_csv(dtm_path+str(row["folder"])+"/"+str(row["id"]),
                           sep="\t", names=["seg", "num", "word", "count"],
                           skiprows=1, dtype={"seg": np.int32, "num": np.int32, "word":str, "count": np.int16})

        data.index = data["word"]

        segs = np.unique(data["seg"])



        for seg in segs:

            if counter == 0:
                chunk = pd.DataFrame()

            if counter % threshold == 0 and counter != 0:

                chunk = chunk.sum(axis=1)

                chunks = pd.concat([chunks, chunk], axis=1, sort=False)
                chunk = pd.DataFrame()

            chunk = pd.concat([chunk, data["count"][data["seg"] == seg]], axis=1, sort=False)

            counter += 1
    stoplist = open(stopword_list,"r").read()
    stoplist = stoplist.split("\n")
    print(len(chunks))
    chunks = chunks[~chunks.index.isin(stoplist)]
    print(len(chunks))
    chunks = chunks.fillna(0).astype(np.int16)

    return chunks

def bina(x):

    if x == 0:
        return 0
    else:
        return 1

