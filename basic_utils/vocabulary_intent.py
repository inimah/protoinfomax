from __future__ import print_function
import numpy
import os
import sys
sys.path.append(os.getcwd())
from simple_tokenizer import tokenizeSimple


def read_word_vectors(filename):
    wdmap = dict()
    W = []
    zeros_init = [float(0.)] * 100
    wdmap['</s>'] = 0
    W.append(zeros_init)
    wdmap['<unk>'] = 1
    W.append(zeros_init)
    curr = 2
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            items = line.split('\t')
            if len(items) == 2 and items[0] not in wdmap:

                print('items w2v:', items)
                sys.stdout.flush()
                
                wdmap[items[0]] = curr
                W.append([float(ii) for ii in items[1].split()])
                curr += 1
                
        f.close()
    print('wdmap', len(wdmap))
    sys.stdout.flush()
    print('len(W)', len(W))
    sys.stdout.flush()
    return wdmap, W


def read_word2idx(infile):
    word2idx = dict()
    with open(infile, 'r') as fi:
        for line in fi:
            wd, idx = line[:-1].split('\t')
            word2idx[wd] = int(idx)
        fi.close()
    return word2idx


def get_word_info(params):
    word2idx = dict()
    w2v = []
    word2idx, w2v = read_word_vectors(params['w2vfile'])
    print("word2idx size:", len(word2idx))

    print("enrich_word_info_with_train_file....")
    enrich_word_info_with_train_file(word2idx,
                                     w2v,
                                     params['training_dir'],
                                     params['training_list'],
                                     params)

    print("After combined with train file")
    print("word2idx size:", len(word2idx))

    print("enrich_word_info_with_dev_file....")
    enrich_word_info_with_train_file(word2idx,
                                     w2v,
                                     params['dev_dir'],
                                     '../data/T1/Test/workspace_t1',
                                     params)
    print("After combined with dev file")
    print("word2idx size:", len(word2idx))

    print("enrich_word_info_with_test_file....")
    enrich_word_info_with_train_file(word2idx,
                                     w2v,
                                     params['testing_dir'],
                                     '../data/T2/Test/workspace_t2',
                                     params)
    print("After combined with test file")
    print("word2idx size:", len(word2idx))
    
    for i in range(len(w2v)):
        if len(w2v[i]) != params['emb_size']:
            raise Exception("wordvec idx %d has a dimension of %d" 
                            % (i, len(w2v[i])))
    w2v = numpy.array(w2v)
    return word2idx, w2v


def enrich_word_info_with_train_file(word2idx,
                                     w2v,
                                     training_wksp_dir,
                                     training_wksp_list,
                                     params):

    training_wksp_dir = training_wksp_dir
    training_wksp_list = training_wksp_list

    print("training_wksp_dir:", training_wksp_dir)
    print("training_wksp_list:", training_wksp_list)

    with open(training_wksp_list, 'r') as fi:

        wksp_ids = fi.readlines()
        wksp_ids = [wid.split('\t')[0] for wid in wksp_ids]

        if training_wksp_dir in ['../data/T1/test/', '../data/T2/test/']:
            wksp_tr_files = [os.path.join(training_wksp_dir, 
                                          wksp.strip()+'.train')
                             for wksp in wksp_ids]
            wksp_tr_files.extend([os.path.join(training_wksp_dir, 
                                          wksp.strip()+'.test')
                             for wksp in wksp_ids])
        else:
            wksp_tr_files = [os.path.join(training_wksp_dir,
                                          wksp.strip()+'.train')
                             for wksp in wksp_ids]
            
        fi.close()

    print("wksp_tr_files:", wksp_tr_files)
    sys.stdout.flush()

    for wkspfile in wksp_tr_files:
        fi = open(wkspfile, 'r')
        for line in fi:
            line = line.strip()
            items = line.split('\t')

            #print("items:", items)
            #sys.stdout.flush()

            if len(items) == 2:
                text, lb = items
                textwds = tokenizeSimple(text, params['max_length'])
                textids = []
                for wd in textwds:
                    if wd in word2idx:
                        textids.append(word2idx[wd])
                    else:
                        word2idx[wd] = len(word2idx)
                        if w2v is not None:
                            w2v.append(((numpy.random.rand(params['emb_size'])
                                       - 0.5) * 2).tolist())
    

    return word2idx, w2v


def write_word2idx(word2idx, outfile):
    with open(outfile, 'w') as fo:
        for wd in word2idx:
            fo.write(wd+'\t'+str(word2idx[wd])+'\n')
        fo.close()
    return
