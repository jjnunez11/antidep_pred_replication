# -*- coding: utf-8 -*-
"""
Script to write some of the t-test supplementary tables for the paper

Created on Fri Sep 18 10:14:57 2020

@author: jjnun
"""
from scipy.stats import ttest_ind
from result_dicts import TABLE3_BACCS, TABLE3_AUCS, TABLE4_BACCS, TABLE4_AUCS, TABLE5_BACCS, TABLE5_AUCS

def write_t_test_grid(output_dir, baccs, aucs):
    output_path = output_dir + 't_test_grid.csv'
    f = open(output_path, 'w+')
    f.write('T_test Grid\n Balanced Accuracy two-tailed two-sided t-tests\n,')
    for bacc in baccs:
        f.write(bacc + ',')

    for bacc in baccs:
        f.write('\n' + bacc + ',')    
        for bacc2 in baccs:
            p_value = ttest_ind(baccs[bacc],baccs[bacc2]).pvalue
            f.write(f'{p_value},')
    
    f.write('\n\n AUC two-tailed two-sided t-tests\n,')
    for auc in aucs:
        f.write(auc + ',')

    for auc in aucs:
        f.write('\n' + auc + ',')    
        for auc2 in aucs:
            p_value = ttest_ind(aucs[auc],aucs[auc2]).pvalue
            f.write(f'{p_value},')

    f.close()


table3_path = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\post_publish\to_submit_correction\Supplementary\Data_For_Figures\Table3_Replication/'
table4_path = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\post_publish\to_submit_correction\Supplementary\Data_For_Figures\Table4_ExternalValidation/'
table5_path = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\post_publish\to_submit_correction\Supplementary\Data_For_Figures\Table5_Comparing/'


write_t_test_grid(table3_path, TABLE3_BACCS, TABLE3_AUCS)
write_t_test_grid(table4_path, TABLE4_BACCS, TABLE4_AUCS)
write_t_test_grid(table5_path, TABLE5_BACCS, TABLE5_AUCS)

