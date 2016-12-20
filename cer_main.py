#coding=utf-8
"""
My version of CNN Sentence classification Model
@author: cer
@forked_from: Yoon Kim
"""

from cer_model import CNN_Sen_Model, make_idx_data_cv

if __name__ == '__main__':
    model = CNN_Sen_Model(conf_file="model.json")
    model.build_model()
    r = range(0, 2)
    for i in r:
        datasets = make_idx_data_cv(model.revs, model.word_idx_map, i, max_l=56,k=300, filter_h=5)
        perf = model.train(datasets)
        print "cv: " + str(i) + ", perf: " + str(perf)
