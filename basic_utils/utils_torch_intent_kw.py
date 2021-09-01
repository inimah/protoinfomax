import os
import sys
sys.path.append(os.getcwd())
import math
#from workspace_intent_kw import workspace
from workspace.workspace_intent_kw2 import workspace
import numpy as np
from utils.cal_methods import HistogramBinning, TemperatureScaling, evaluate, cal_results


def search_best_threshold(params, valid_output_info):
    dataset_best_thresholds = []
    dataset_best_values = []

    #print('****************')
    #print('search_best_threshold')

    bestT = 0
    bestV = 1
    best_frr = 0
    best_far = 1

    offsize = len([conf for pred, gt, conf in valid_output_info
                  if gt == params['offtopic_label']])
    insize = len([conf for pred, gt, conf in valid_output_info
                 if gt != params['offtopic_label']])

    if offsize == 0:
        offsize = offsize+1

    if insize == 0:
        insize = insize+1

    print('offsize, insize', offsize, insize)
    sorted_valid_output_info = sorted(valid_output_info, key=lambda x: x[2])

    accepted_oo = offsize
    rejected_in = 0.0
    threshold = 0.0
    ind = 0
    # sorted based on low confidence of prediction
    for pred, gt, conf in sorted_valid_output_info[:-1]:

        #print("pred, gt, conf:", (pred, gt, conf))
        threshold = (sorted_valid_output_info[ind][2] + 
                     sorted_valid_output_info[ind+1][2])/2.0
        #print("threshold: ", threshold)
        if gt != params['offtopic_label']:
            rejected_in += 1.0
        else:
            accepted_oo -= 1.0


        frr = rejected_in / insize
      
        far = accepted_oo / offsize
        dist = math.fabs(frr - far)
       

        if dist < bestV:
            bestV = dist
            bestT = threshold
            best_frr = frr
            best_far = far
        ind += 1

      
        #print('bestT, bestV, bestFAR, bestFRR', 
        #      bestT, bestV, best_far, best_frr)

    return bestT, bestV


def get_results(params, output_info, threshold):

    #print('****************')
    #print('get_results funct.')

    total_gt_ontopic_utt = len([gt for pred, gt, conf in output_info
                               if gt != params['offtopic_label']])
    total_gt_offtopic_utt = len(output_info) - total_gt_ontopic_utt

    if total_gt_ontopic_utt == 0:
        total_gt_ontopic_utt = total_gt_ontopic_utt+1

    if total_gt_offtopic_utt == 0:
        total_gt_offtopic_utt = total_gt_offtopic_utt+1


    #print("total_gt_ontopic_utt:", total_gt_ontopic_utt)
    #print("total_gt_offtopic_utt:", total_gt_offtopic_utt)

    print("threshold:", threshold)

    accepted_oo = 0.0
    rejected_in = 0.0
    correct_domain_label = 0.0
    correct_wo_thr = 0.0
    correct_w_thr = 0.0

    for pred, gt, conf in output_info:
        #print("pred, gt, conf:", (pred, gt, conf))

        if conf < threshold:
            pred1 = params['offtopic_label']
        else:
            pred1 = pred

        #print("pred1:", pred1)

        if gt == params['offtopic_label'] and pred1 != gt:
            accepted_oo += 1
            #print("if gt == params['offtopic_label'] and pred1 != gt")

        elif gt != params['offtopic_label'] and pred1 == params['offtopic_label']:
            rejected_in += 1
            #print("elif gt != params['offtopic_label'] and pred1 == params['offtopic_label']")
        else:
            correct_domain_label += 1
            #print("else")

        if gt != params['offtopic_label'] and pred == gt:
            correct_wo_thr += 1
            #print("if gt != params['offtopic_label'] and pred == gt")

        if gt != params['offtopic_label'] and pred1 == gt:
            correct_w_thr += 1
            #print("if gt != params['offtopic_label'] and pred1 == gt")

     
    far = accepted_oo / total_gt_offtopic_utt
    frr = rejected_in / total_gt_ontopic_utt
    eer = 1 - correct_domain_label / len(output_info)
    ontopic_acc_ideal = correct_wo_thr / total_gt_ontopic_utt
    ontopic_acc = correct_w_thr / total_gt_ontopic_utt


    print("eer, far, frr, ontopic_acc_ideal, ontopic_acc:", (eer, far, frr, ontopic_acc_ideal, ontopic_acc))

    return eer, far, frr, ontopic_acc_ideal, ontopic_acc


def compute_values(params, experiment, result_data, epoch, idx2word, desc_str):

    t_macro_avg_eer = 0.0
    t_macro_avg_far = 0.0
    t_macro_avg_frr = 0.0

    t_macro_avg_acc_ideal = 0.0
    t_macro_avg_acc = 0.0


    print("Epoch-%s-%s" %(desc_str, epoch))
    sys.stdout.flush()

    preds, all_dataset, probs, gts, avg_conf_ood = experiment.run_testing_epoch(params, result_data)
    print(all_dataset[0])

    thesholds, _ = search_best_threshold(params, preds)

    print("thesholds best:", thesholds)

    test_eer, test_far, test_frr, test_ontopic_acc_ideal, \
            test_ontopic_acc = get_results(params, preds, 
                                           thesholds)
    print('test(eer, far, frr, ontopic_acc_ideal, ontopic_acc) %.3f, %.3f, %.3f, %.3f, %.3f' %
              (test_eer, test_far, test_frr,
               test_ontopic_acc_ideal,
               test_ontopic_acc))
    
    #n_probs = len(probs)
    #error, ece, mce, loss = cal_results(probs, gts)


    t_macro_avg_eer += test_eer
    t_macro_avg_far += test_far
    t_macro_avg_frr += test_frr
    t_macro_avg_acc_ideal += test_ontopic_acc_ideal
    t_macro_avg_acc += test_ontopic_acc


    return t_macro_avg_eer, t_macro_avg_far, t_macro_avg_frr, \
        t_macro_avg_acc_ideal, t_macro_avg_acc, preds, all_dataset, avg_conf_ood


def compute_values_eval(params, experiment, result_data, desc_str):

    t_macro_avg_eer = 0.0
    t_macro_avg_far = 0.0
    t_macro_avg_frr = 0.0

    t_macro_avg_acc_ideal = 0.0
    t_macro_avg_acc = 0.0



    preds, all_dataset, probs, gts, avg_conf_ood = experiment.run_testing_epoch(params, result_data)
    print(all_dataset[0])

    thesholds, _ = search_best_threshold(params, preds)

    print("thesholds best:", thesholds)

    test_eer, test_far, test_frr, test_ontopic_acc_ideal, \
            test_ontopic_acc = get_results(params, preds, 
                                           thesholds)
    print('test(eer, far, frr, ontopic_acc_ideal, ontopic_acc) %.3f, %.3f, %.3f, %.3f, %.3f' %
              (test_eer, test_far, test_frr,
               test_ontopic_acc_ideal,
               test_ontopic_acc))
    
    #n_probs = len(probs)
    #error, ece, mce, loss = cal_results(probs, gts)


    t_macro_avg_eer += test_eer
    t_macro_avg_far += test_far
    t_macro_avg_frr += test_frr
    t_macro_avg_acc_ideal += test_ontopic_acc_ideal
    t_macro_avg_acc += test_ontopic_acc


    return t_macro_avg_eer, t_macro_avg_far, t_macro_avg_frr, \
        t_macro_avg_acc_ideal, t_macro_avg_acc, preds, all_dataset, avg_conf_ood, probs, gts
        
def get_data(params, dir_, file_list, role):
    workspaces = []
    with open(file_list) as fi:
        i = 0
        for wid in fi:
            wid = wid.strip().split('\t')[0]
            print("workspace name:", wid)
            workspaces.append(workspace(wid, params, dir_, role))
            print('get_data:', i)
            sys.stdout.flush()
            i += 1
    return workspaces
