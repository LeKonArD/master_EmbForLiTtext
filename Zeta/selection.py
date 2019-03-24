import pandas as pd
from sklearn.model_selection import KFold

def get_folds_2(x1, x2, random_state, run_num):
    x1_ind = list(x1.index)
    x2_ind = list(x2.index)
    splitter = KFold(n_splits=5, random_state=random_state)
    s1 = list(splitter.split(x1_ind))
    s2 = list(splitter.split(x2_ind))

    return x1.iloc[s1[run_num][0],:], x1.iloc[s1[run_num][1],:], x2.iloc[s2[run_num][0],:], x2.iloc[s2[run_num][1],:]

def get_folds(x1, x2, random_state, run_num):
    x1_ind = list(x1.index)
    x2_ind = list(x2.index)
    splitter = KFold(n_splits=40, random_state=random_state)
    s1 = list(splitter.split(x1_ind))
    s2 = list(splitter.split(x2_ind))

    return x1.iloc[s1[run_num][0],:], x1.iloc[s1[run_num][1],:], x2.iloc[s2[run_num][0],:], x2.iloc[s2[run_num][1],:]


def weight(x, frame):
    w = len(frame) / len(frame[frame["genre"] == x])
    return w


def case_1_select(focus, counter, meta_path, random_state, run_num):

    meta_full = pd.read_csv(meta_path, sep="\t", index_col=0)

    meta_focus = meta_full[meta_full["genre"] == focus][:100]
    meta_counter = meta_full[meta_full["genre"] == counter][:100]

    if len(meta_focus) > len(meta_counter):

        meta_focus = meta_focus[:len(meta_counter)]

    else:

        meta_counter = meta_counter[:len(meta_focus)]

    meta_focus_test, meta_focus_train, meta_counter_test, meta_counter_train = get_folds(meta_focus,
                                                                                         meta_counter,
                                                                                         random_state,
                                                                                         run_num)

    return meta_focus_train, meta_counter_train, meta_focus_test, meta_counter_test


def case_2_select(focus, meta_path, random_state, run_num):

    meta_full = pd.read_csv(meta_path, sep="\t", index_col=0)

    meta_focus = meta_full[meta_full["genre"] == focus].sample(150, random_state=45)
    meta_counter = meta_full[meta_full["genre"] != focus].sample(150, random_state=45)

    if len(meta_focus) > len(meta_counter):

        meta_focus = meta_focus[:len(meta_counter)]

    else:

        meta_counter = meta_counter[:len(meta_focus)]

    meta_focus_test, meta_focus_train, meta_counter_test, meta_counter_train = get_folds(meta_focus,
                                                                                         meta_counter,
                                                                                         random_state,
                                                                                         run_num)

    return meta_focus_train, meta_counter_train, meta_focus_test, meta_counter_test


def case_3_select(focus, reihe, meta_path, random_state, run_num):

    meta_full = pd.read_csv(meta_path, sep="\t", index_col=0)

    meta_focus = meta_full[meta_full["genre"] == focus]
    meta_focus = meta_focus[meta_focus["reihenname"] == reihe]

    meta_counter = meta_full[meta_full["genre"] != focus]

    if len(meta_focus) > len(meta_counter):

        meta_focus = meta_focus[:len(meta_counter)]

    else:

        meta_counter = meta_counter[:len(meta_focus)]

    meta_focus_test, meta_focus_train, meta_counter_test, meta_counter_train = get_folds(meta_focus,
                                                                                         meta_counter,
                                                                                         random_state,
                                                                                         run_num)
    return meta_focus_train, meta_counter_train, meta_focus_test, meta_counter_test


def case_45_select(focus, counter, meta_path, random_state, run_num):

    meta_full = pd.read_csv(meta_path, sep="\t", index_col=0)

    meta_focus = meta_full[meta_full["reihenname"] == focus]
    meta_counter = meta_full[meta_full["reihenname"] == counter]

    if len(meta_focus) > len(meta_counter):

        meta_focus = meta_focus[:len(meta_counter)]

    else:

        meta_counter = meta_counter[:len(meta_focus)]

    meta_focus_test, meta_focus_train, meta_counter_test, meta_counter_train = get_folds(meta_focus,
                                                                                         meta_counter,
                                                                                         random_state,
                                                                                         run_num)

    return meta_focus_train, meta_counter_train, meta_focus_test, meta_counter_test


def case_6_select(meta_path, random_state, run_num):

    meta_full = pd.read_csv(meta_path, sep="\t", index_col=0)

    meta_focus = meta_full[meta_full["genre"].isin(["Liebes", "Heimat", "Arzt", "Adels", "Familien"])].sample(200, random_state=45)
    meta_counter = meta_full[meta_full["genre"].isin(["Abenteuer", "Krimi", "SciFi", "Western"])].sample(200, random_state=45)

    if len(meta_focus) > len(meta_counter):

        meta_focus = meta_focus[:len(meta_counter)]

    else:

        meta_counter = meta_counter[:len(meta_focus)]

    meta_focus_test, meta_focus_train, meta_counter_test, meta_counter_train = get_folds(meta_focus,
                                                                                         meta_counter,
                                                                                         random_state,
                                                                                         run_num)

    return meta_focus_train, meta_counter_train, meta_focus_test, meta_counter_test

def case_78_select(focus, counter, reihe, meta_path, random_state, run_num):
    print(reihe)
    meta_full = pd.read_csv(meta_path, sep="\t", index_col=0)

    meta_focus = meta_full[meta_full["reihenname"] == reihe]
    meta_counter = meta_full[meta_full["reihenname"] == reihe]

    meta_focus = meta_focus[meta_focus["author"] == focus]
    meta_counter = meta_counter[meta_counter["author"] == counter]


    if len(meta_focus) > len(meta_counter):

        meta_focus = meta_focus[:len(meta_counter)]

    else:

        meta_counter = meta_counter[:len(meta_focus)]

    meta_focus_test, meta_focus_train, meta_counter_test, meta_counter_train = get_folds_2(meta_focus,
                                                                                         meta_counter,
                                                                                         random_state,
                                                                                         run_num)

    return meta_focus_train, meta_counter_train, meta_focus_test, meta_counter_test

def case_910_select(focus, reihe, meta_path, random_state, run_num):

    meta_full = pd.read_csv(meta_path, sep="\t", index_col=0)

    meta_focus = meta_full[meta_full["reihenname"] == reihe]
    meta_counter = meta_full[meta_full["reihenname"] == reihe]

    meta_focus = meta_focus[meta_focus["author"] == focus]
    meta_counter = meta_counter[meta_counter["author"] != focus]


    if len(meta_focus) > len(meta_counter):

        meta_focus = meta_focus[:len(meta_counter)]

    else:

        meta_counter = meta_counter[:len(meta_focus)]

    meta_focus_test, meta_focus_train, meta_counter_test, meta_counter_train = get_folds_2(meta_focus,
                                                                                         meta_counter,
                                                                                         random_state,
                                                                                         run_num)

    return meta_focus_train, meta_counter_train, meta_focus_test, meta_counter_test
