import argparse
from selection import *
from zeta_base import *
from py_zeta import *
from classify import *
from estimate_genre_fields import *
import fastText as ft

final_results = []
class Args:
    case = 1
    focus = "Liebes"
    reihenname = ""
    counter = "Arzt"
    meta = "meta_master.csv"
    dtm_path = "seq_dtm/"
    dtm_single_path = "dtm_single/"
    seg_size = 10000
    logaddition = 0.21
    ft_model = "cc.de.300.bin"
    ft_model_trained = "/mnt/data/kallimachos/dnb/romangeschichte-dnb/dnb_embeddings.ftz"
    w2v_model = "german.model"
    stoplist = "/home/konle/zeta/stopwords_zeta.txt"
    t_mode = False
    metric = "euclidean"
    random_state = 10
    run_num = 1
    meth = "MS"
    calc = False
args=Args()

if args.case == 1:
        print("select")
        focus_train, counter_train, focus_test, counter_test = case_1_select(args.focus, args.counter,
                                                                             args.meta, args.random_state, args.run_num)
        
focus_test = focus_test[:70]
counter_test =  counter_test[:70]


bina_focus_train, bina_counter_train, \
    rel_focus_train, rel_counter_train, \
    abs_focus_train, abs_counter_train = \
        get_segements(focus_train, counter_train,
                      args.dtm_path, args.seg_size, args.stoplist)
        
        
sd0, sd2, sg0, sg2, dg0, dg2, devprops1, devprops2 = calculate_scores(bina_focus_train, bina_counter_train,
                                                                          abs_focus_train, abs_counter_train,
                                                                          args.logaddition, args.seg_size)

zeta_res = pd.DataFrame()
zeta_res["sd0"] = sd0
zeta_res["sd2"] = sd2
zeta_res["sg0"] = sg0
zeta_res["sg2"] = sg2
zeta_res["dg0"] = dg0
zeta_res["dg2"] = dg2
zeta_res["devprops_1"] = devprops1
zeta_res["devprops_2"] = devprops2
zeta_res.index = list(abs_focus_train.index)


clf_res_base = classify_base_zeta(zeta_res, focus_train, counter_train, focus_test, counter_test, args.dtm_single_path)
print(clf_res_base)
final_results.append(clf_res_base)

model_ft = ft.load_model(args.ft_model_trained)


for m in ["MS","AP","Birch"]:
    for p in [True,False]:
        for me in ["euclidean","manhattan","cosine"]:
            word_fields = get_ft_fields(zeta_res, model_ft, "sd2", m)
            if p:

                word_fields, pca = bend_embedding_space(me, word_fields)
            else:
                pca = "no_pca"

            for c in [False,True]:
                clf_ft_zeta_trained = classify_fasttext(focus_train, counter_train, focus_test, counter_test, args.dtm_single_path, model_ft, word_fields, "trained", args.stoplist, pca, c, me)
                print(clf_ft_zeta_trained)
                final_results.append(clf_ft_zeta_trained)

                
del model_ft
model_ft = ft.load_model(args.ft_model)

for m in ["MS","AP","Birch"]:
    for p in [True,False]:
        for me in ["euclidean","manhattan","cosine"]:
            word_fields = get_ft_fields(zeta_res, model_ft, "sd2", m)
            if p:

                word_fields, pca = bend_embedding_space(me, word_fields)
            else:
                pca = "no_pca"

            for c in [False,True]:
                clf_ft_zeta_trained = classify_w2v(focus_train, counter_train, focus_test, counter_test, args.dtm_single_path, model_ft, word_fields, "trained", args.stoplist, pca, c, me)
                print(clf_ft_zeta_trained)
                final_results.append(clf_ft_zeta_trained)
                
del model_ft
model_w2v = KeyedVectors.load_word2vec_format(args.w2v_model, binary=True, unicode_errors="ignore")

for m in ["MS","AP","Birch"]:
    for p in [True,False]:
        for me in ["euclidean","manhattan","cosine"]:
            word_fields = get_w2v_fields(zeta_res, model_w2v, "sd2", m)
            if p:

                word_fields, pca = bend_embedding_space(me, word_fields)
            else:
                pca = "no_pca"

            for c in [False,True]:
                clf_ft_zeta_trained = classify_w2v(focus_train, counter_train, focus_test, counter_test, args.dtm_single_path, model_w2v, word_fields, "trained", args.stoplist, pca, c, me)
                print(clf_ft_zeta_trained)
                final_results.append(clf_ft_zeta_trained)
out = ""
for r in final_results:
    out+=str(r)
    
with open("vorstudie.txt","w") as f:
    f.write(out)
