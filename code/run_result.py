# -*- coding: utf-8 -*-
"""
Function to produce one result from our paper, consisting of 100 runs of each algorithm/configuration
"""
import os
import re
from scipy.stats import ttest_1samp
import datetime
import numpy as np
from run_mlrun import RunMLRun
import _pickle as cPickle
import bz2
from run_globals import DATA_DIR, RESULTS_DIR

startTime = datetime.datetime.now()


def run_result(runs, evl, model, f_select, data_name, label_name, table=""):
    # Make a folder for each table to keep things organized
    table_path = os.path.join(RESULTS_DIR, table)
    if not (os.path.exists(table_path)):
        os.mkdir(table_path)

    # Gets the path of the training data and label files, testing data and label files
    if evl == "cv":
        test_data = os.path.join(DATA_DIR, data_name + "_holdout.csv")
        test_label = os.path.join(DATA_DIR, label_name + "_holdout.csv")
        train_data = os.path.join(DATA_DIR, data_name + "_non_holdout.csv")
        train_label = os.path.join(DATA_DIR, label_name + "_non_holdout.csv")
    else:
        train_data = os.path.join(DATA_DIR, data_name + "_entire.csv")
        train_label = os.path.join(DATA_DIR, label_name + "_entire.csv")
        test_data = os.path.join(DATA_DIR,
                                 'X_tillwk4_overlap_canbind.csv')  # X data matrix over CAN-BIND, only overlapping
        # features with STAR*D, subjects who have qids sr until at least week 4
        if evl == "extval_resp":
            test_label = os.path.join(DATA_DIR,
                                      'y_wk8_resp_qids_sr_canbind.csv')  # y matrix from canbind, with subjects as
            # above, targeting week 8 qids sr response
        elif evl == "extval_rem":
            test_label = os.path.join(DATA_DIR,
                                      'y_wk8_rem_qids_sr_canbind.csv')  # y matrix from canbind, with subjects as
            # above, targeting week 8 qids sr remission
        elif evl == "extval_rem_randomized":  # A control to make sure our extval_rem results are robust, with the
            # targets scrambled randomly
            test_label = os.path.join(DATA_DIR,
                                      'y_wk8_randomized_qids_sr_canbind.csv')  # y matrix from canbind, with subjects
            # as above, with targets scrambled
        else:
            raise ValueError("Invalid evaluation type (ev) provided. Must be in cv, extval_rem, extval_resp, "
                             "or extval_rem_randomized")

    # Set n_splits, how many fold the cross-validation used for training should be, as well as ensemble_n,
    # the number of models to train to use as an ensemble for prediction. Set to Nie et al's values.
    ensemble_n = 30
    n_splits = 10

    # Create numpy arrays to store all the results
    accus = np.zeros(runs)
    bal_accus = np.zeros(runs)
    aucs = np.zeros(runs)
    senss = np.zeros(runs)
    specs = np.zeros(runs)
    precs = np.zeros(runs)
    f1s = np.zeros(runs)
    ppvs = np.zeros(runs)
    npvs = np.zeros(runs)
    tps = np.zeros(runs)
    fps = np.zeros(runs)
    tns = np.zeros(runs)
    fns = np.zeros(runs)
    feats = np.zeros(runs)  # Average number of the average number of features used per classifier trained

    # Create filename based on parameters
    result_filename = "{}_{}_{}_{}_{}_{}_{}".format(evl, model, runs, data_name, label_name, f_select,
                                                    datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    # Make a dir for each result
    result_dir = os.path.join(table_path, result_filename)
    if not (os.path.exists(result_dir)):
        os.mkdir(result_dir)

    for i in range(runs):
        accus[i], bal_accus[i], aucs[i], senss[i], specs[i], precs[i], f1s[i], ppvs[i], npvs[i],  feats[i], impt, confus_mat, run_clfs \
            = RunMLRun(train_data, train_label, test_data, test_label, f_select, model, ensemble_n, n_splits)

        tps[i] = confus_mat['tp']
        fps[i] = confus_mat['fp']
        tns[i] = confus_mat['tn']
        fns[i] = confus_mat['fn']

        if i == 0:
            # Initialize impts now as number of features can change
            impts = np.empty([runs, np.size(impt)], dtype=float)

        impts[i, :] = impt

        # Save this run's models into a cPickle with bz2
        models_filename = os.path.join(result_dir, f"run_{i}") + '.pbz2'
        with bz2.BZ2File(models_filename, 'w') as f2:
            cPickle.dump(run_clfs, f2)

        print("Finished run: " + str(i + 1) + " of " + str(runs) + "\n")

    # Process feature importance
    avg_impts = np.mean(impts, axis=0)
    std_impts = np.std(impts, axis=0)

    sorted_features = np.argsort(avg_impts)[::-1]
    top_31_features = sorted_features[0:31]  # In descending importance, first is most important
    with open(train_data) as f:
        feature_names = f.readline().split(',')

    # Write output file
    f = open(os.path.join(table_path, result_filename + '.txt'), 'w')

    f.write("MODEL RESULTS for run at: " + result_filename + "\n\n")

    f.write("Model Parameters:-----------------------------------\n")
    f.write("Evaluation: " + evl + "\n")
    f.write("Model: " + model + "\n")
    f.write("Feature selection: " + f_select + "\n")
    f.write("Train X is: " + train_data + "\n")
    f.write("Train y is: " + train_label + "\n")
    f.write("Test X is: " + train_data + "\n")
    f.write("Test y is: " + train_label + "\n")
    f.write(str(runs) + " runs of 10-fold CV\n\n")

    f.write("Summary of Results:------------------------------------\n")
    f.write("Mean accuracy is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(accus), np.std(accus)))
    f.write("Mean balanced accuracy is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(bal_accus),
                                                                                          np.std(bal_accus)))
    f.write("Mean AUC is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(aucs), np.std(aucs)))
    f.write("Mean sensitivity is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(senss), np.std(senss)))
    f.write("Mean specificity is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(specs), np.std(specs)))
    f.write("Mean precision is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(precs), np.std(precs)))
    f.write("Mean f1 is: {:.4f}, with Standard Deviation: {:.4f}\n".format(np.mean(f1s), np.std(f1s)))
    f.write("Mean positive predictive value is: {:.4f}, with Standard Deviation: {:.4f}\n".format(np.mean(ppvs), np.std(ppvs)))
    f.write("Mean negative predictive value is: {:.4f}, with Standard Deviation: {:.4f}\n".format(np.mean(npvs), np.std(npvs)))

    f.write("Mean true positive is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(tps), np.std(tps)))
    f.write("Mean false positive is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(fps), np.std(fps)))
    f.write("Mean true negative is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(tns), np.std(tns)))
    f.write("Mean false negative is: {:.4f}, with Standard Deviation: {:.6f}\n".format(np.mean(fns), np.std(fns)))

    f.write(
        "Mean number of features used is: {:.4f} of {:d}, with Standard Deviation: {:.4f}\n\n".format(np.mean(feats),
                                                                                                      np.size(
                                                                                                          avg_impts),
                                                                                                      np.std(feats)))

    f.write("Feature Importance And Use:---------------------------\n")
    f.write("Top 31 Features by importance, in descending order (1st most important):\n")
    f.write("By position in data matrix, 1 added to skip index=0 \n")

    if np.sum(avg_impts) != 0:
        f.write(str(top_31_features + 1) + "\n")
        for i in range(len(top_31_features)):
            f.write(feature_names[top_31_features[i] + 1] + "\n")
        f.write("\n")
    else:
        f.write("Code does not support feature for this model at this time\n")

    f.write("Statistical Significance:----------------------------\n")
    if (data_name == "full_trd" or data_name == "ovlap_trd") and model == "rf_cv" and f_select == "all":
        _, acc_pvalue = ttest_1samp(accus, 0.70)
        f.write("P-value from one sided t-test vs Nie et al's 0.70 Accuracy: {:.6f}\n".format(acc_pvalue))
        _, bal_pvalue = ttest_1samp(bal_accus, 0.70)
        f.write("P-value from one sided t-test vs Nie et al's 0.70 Balanced Accuracy: {:.6f}\n".format(bal_pvalue))
        _, auc_pvalue = ttest_1samp(aucs, 0.78)
        f.write("P-value from one sided t-test vs Nie et al's 0.78 AUC: {:.6f}\n".format(auc_pvalue))
        _, senss_pvalue = ttest_1samp(senss, 0.69)
        f.write("P-value from one sided t-test vs Nie et al's 0.69 Sensitivity: {:.6f}\n".format(auc_pvalue))
        _, specs_pvalue = ttest_1samp(specs, 0.71)
        f.write("P-value from one sided t-test vs Nie et al's 0.71 Specificity: {:.6f}\n\n".format(auc_pvalue))

    f.write("Raw results:----------------------------------------\n")
    f.write("Accuracies\n")
    f.write(re.sub(r"\s+", r",", str(accus)) + "\n")
    f.write("Balanced Accuracies\n")
    f.write(re.sub(r"\s+", r",", str(bal_accus)) + "\n")
    f.write("AUCs\n")
    f.write(re.sub(r"\s+", r",", str(aucs)) + "\n")
    f.write("Sensitivites\n")
    f.write(re.sub(r"\s+", r",", str(senss)) + "\n")
    f.write("Specificities\n")
    f.write(re.sub(r"\s+", r",", str(specs)) + "\n")
    f.write("Precisions\n")
    f.write(re.sub(r"\s+", r",", str(precs)) + "\n")
    f.write("F1s\n")
    f.write(re.sub(r"\s+", r",", str(f1s)) + "\n")
    f.write("Number of features used\n")
    f.write(re.sub(r"\s+", r",", str(feats)) + "\n")
    f.write("Mean Feature importances Across Runs\n")
    f.write(re.sub(r" +", r",", np.array_str(avg_impts, precision=4, max_line_width=100)) + "\n")
    f.write("Mean Feature importances std. deviation Across Runs\n")
    f.write(re.sub(r" +", r",", np.array_str(std_impts, precision=4, max_line_width=100)) + "\n")

    f.close()

    print("Completed after seconds: \n")
    print(datetime.datetime.now() - startTime)
