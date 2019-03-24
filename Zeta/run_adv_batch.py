import argparse
from selection import *
from zeta_base import *
from py_zeta import *
from classify import *
from estimate_genre_fields_real import *
from gensim.models import Word2Vec, KeyedVectors
import fastText as ft
import re

if __name__ == "__main__":
	print("start")
	out = ""
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-case', type=int, default=None)
	parser.add_argument('-focus', type=str, default="ff")
	parser.add_argument('-reihenname', type=str, default="ff")
	parser.add_argument('-counter', type=str, default="ff")
	parser.add_argument('-run_num', type=int, default=None)
	parser.add_argument('-meta', type=str, default="meta_master.csv")
	parser.add_argument('-dtm_path', type=str, default="seq_dtm/")
	parser.add_argument('-dtm_single_path', type=str, default="dtm_single/")
	parser.add_argument('-seg_size', type=int, default=10000)
	parser.add_argument('-logaddition', type=float, default=0.21)
	parser.add_argument('-ft_model', type=str, default="cc.de.300.bin")
	parser.add_argument('-ft_model_trained', type=str, default="dnb_embeddings.bin")
	parser.add_argument('-w2v_model', type=str, default="german.model")
	parser.add_argument('-stoplist', type=str, default="stopwords_zeta.txt")
	parser.add_argument('-random_state', type=int, default=32)
	args = parser.parse_args()
	args.focus = re.sub("\?"," ",args.focus)
	args.counter = re.sub("\?"," ",args.counter)
	args.reihenname = re.sub("\?"," ",args.reihenname)

	if args.case == 1:
		print("select")
		focus_train, counter_train, focus_test, counter_test = case_1_select(args.focus, args.counter,
                                                                             args.meta, args.random_state, args.run_num)

	if args.case == 2:
		print("select")
		focus_train, counter_train, focus_test, counter_test = case_2_select(args.focus, args.meta,
		                                                                     args.random_state, args.run_num)

	if args.case == 3:
		print("select")
		focus_train, counter_train, focus_test, counter_test = case_3_select(args.focus, args.reihenname,
		                                                                     args.meta, args.random_state, args.run_num)

	if args.case == 4:
		print("select")
		focus_train, counter_train, focus_test, counter_test = case_45_select(args.focus, args.counter,
		                                                                      args.meta, args.random_state, args.run_num)

	if args.case == 5:
		print("select")
		focus_train, counter_train, focus_test, counter_test = case_45_select(args.focus, args.counter,
		                                                                      args.meta, args.random_state, args.run_num)

	if args.case == 6:
		print("select")
		focus_train, counter_train, focus_test, counter_test = case_6_select(args.meta, args.random_state, args.run_num)

	if args.case == 7:
		print("select")
		focus_train, counter_train, focus_test, counter_test = case_78_select(args.focus,
		                                                                     args.counter, args.reihenname,
		                                                                     args.meta, args.random_state, args.run_num)

	if args.case == 8:
		print("select")
		focus_train, counter_train, focus_test, counter_test = case_78_select(args.focus,
		                                                                     args.counter, args.reihenname,
		                                                                     args.meta, args.random_state, args.run_num)

	if args.case == 9:
		print("select")
		focus_train, counter_train, focus_test, counter_test = case_910_select(args.focus, args.reihenname, args.meta,
		                                                                       args.random_state, args.run_num)

	if args.case == 10:
		print("select")
		focus_train, counter_train, focus_test, counter_test = case_910_select(args.focus, args.reihenname, args.meta,
		                                                                       args.random_state, args.run_num)
	final_res = []

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
	final_res.append(clf_res_base)
	print(clf_res_base)
	for r in final_res:
		out+=str(r)+"\n\n"
					    
	with open(str(args.case)+"_"+str(args.focus)+"_"+str(args.counter)+"_"+str(args.run_num)+".txt","w") as f:
		f.write(out)

	model_ft = ft.load_model(args.ft_model_trained)

	for m in ["Birch"]:
		for p in [True]:
			for me in ["euclidean"]:
				word_fields = get_ft_fields(zeta_res, model_ft, "sd2", m)
				print(word_fields)
				if p:
					word_fields, pca = bend_embedding_space(me, word_fields)
					if pca == "error":
						out+="error\n"
						print("error")
						continue
				else:
					pca = "no_pca"

				for c in [False]:
					if pca == "error":
						continue
					clf_ft_zeta_trained = classify_fasttext(focus_train, counter_train, focus_test, counter_test, args.dtm_single_path, model_ft, word_fields, "trained", args.stoplist, pca, c, me)
					print(clf_ft_zeta_trained)
					final_res.append(clf_ft_zeta_trained)
					for r in final_res:
						out+=str(r)+"\n"
					    
					with open(str(args.case)+"_"+str(args.focus)+"_"+str(args.counter)+"_"+str(args.run_num)+".txt","w") as f:
						f.write(out)

	del model_ft
	del word_fields
	del pca

	model_ft = ft.load_model(args.ft_model)

	for m in ["Birch"]:
		for p in [True]:
			for me in ["euclidean"]:
				word_fields = get_ft_fields(zeta_res, model_ft, "sd2", m)
				print(word_fields)
				if p:
					word_fields, pca = bend_embedding_space(me, word_fields)
					if pca == "error":
                                                out+="error\n"
                                                continue

				else:
					pca = "no_pca"

				for c in [False]:
					if pca == "error":
						continue
					clf_ft_zeta_trained = classify_fasttext(focus_train, counter_train, focus_test, counter_test, args.dtm_single_path, model_ft, word_fields, "trained", args.stoplist, pca, c, me)
					print(clf_ft_zeta_trained)
					final_res.append(clf_ft_zeta_trained)
					for r in final_res:
						out+=str(r)+"\n"
					    
					with open(str(args.case)+"_"+str(args.focus)+"_"+str(args.counter)+"_"+str(args.run_num)+".txt","w") as f:
						f.write(out)

	del model_ft
	del word_fields
	del pca

	model_w2v = KeyedVectors.load_word2vec_format(args.w2v_model, binary=True, unicode_errors="ignore")

	for m in ["Birch"]:
		for p in [True]:
			for me in ["euclidean"]:
				word_fields = get_w2v_fields(zeta_res, model_w2v, "sd2", m)
				if p:

					word_fields, pca = bend_embedding_space(me, word_fields)
					if pca == "error":
                                                out+="error\n"
                                                continue

				else:
					pca = "no_pca"

				for c in [False]:
					if pca == "error":
						continue
					clf_ft_zeta_trained = classify_w2v(focus_train, counter_train, focus_test, counter_test, args.dtm_single_path, model_w2v, word_fields, "trained", args.stoplist, pca, c, me)
					print(clf_ft_zeta_trained)
					final_res.append(clf_ft_zeta_trained)
					for r in final_res:
						out+=str(r)+"\n"
					    
					with open(str(args.case)+"_"+str(args.focus)+"_"+str(args.counter)+"_"+str(args.run_num)+".txt","w") as f:
						f.write(out)
					
	del model
	del word_fields
	del pca

	out = ""
	for r in final_res:
		out+=str(r)+"\n"
	    
	with open(str(args.case)+"_"+str(args.focus)+"_"+str(args.counter)+"_"+str(args.run_num)+".txt","w") as f:
		f.write(out)
