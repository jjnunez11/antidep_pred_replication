# -*- coding: utf-8 -*-
"""
Scripts to generate the z-score tables which compare previous results within a table
and vs the reported performance of Nie et al

Created on Fri Sep 18 10:14:57 2020

@author: jjnun
"""

from scipy.stats import zmap
from result_dicts import TABLE3_AUCS, TABLE3_BACCS

rf_full_bacc = TABLE3_BACCS["rf_full_bacc"]
rf_full_auc = TABLE3_AUCS["rf_full_auc"]
rf_full_nie_bacc = 0.70
rf_full_nie_auc  = 0.78
print(f'rf Full Features BACC Zscore {zmap(rf_full_nie_bacc, rf_full_bacc)}')
print(f'rf Full Features AUC Zscore {zmap(rf_full_nie_auc, rf_full_auc)}')


rf_chi2_bacc = TABLE3_BACCS["rf_chi2_bacc"]
rf_chi2_auc = TABLE3_AUCS["rf_chi2_auc"]
rf_chi2_nie_bacc = 0.68
rf_chi2_nie_auc  = 0.77
print(f'rf Chi2 BACC Zscore {zmap(rf_chi2_nie_bacc, rf_chi2_bacc)}')
print(f'rf Chi2 AUC Zscore {zmap(rf_chi2_nie_auc, rf_chi2_auc)}')


rf_elnet_bacc = TABLE3_BACCS["rf_elnet_bacc"]
rf_elnet_auc = TABLE3_AUCS["rf_elnet_auc"]
rf_elnet_nie_bacc = 0.69
rf_elnet_nie_auc  = 0.76
print(f'rf elnet Features BACC Zscore {zmap(rf_elnet_nie_bacc, rf_elnet_bacc)}')
print(f'rf elnet Features AUC Zscore {zmap(rf_elnet_nie_auc, rf_elnet_auc)}')


########################
gbdt_full_bacc = TABLE3_BACCS["gbdt_full_bacc"]
gbdt_full_auc = TABLE3_AUCS["gbdt_full_auc"]
gbdt_full_nie_bacc = 0.70
gbdt_full_nie_auc  = 0.78
print(f'GBDT Full Features BACC Zscore {zmap(gbdt_full_nie_bacc, gbdt_full_bacc)}')
print(f'GBDT Full Features AUC Zscore {zmap(gbdt_full_nie_auc, gbdt_full_auc)}')

gbdt_chi2_bacc = TABLE3_BACCS["gbdt_chi2_bacc"]
gbdt_chi2_auc = TABLE3_AUCS["gbdt_chi2_auc"]
gbdt_chi2_nie_bacc = 0.70
gbdt_chi2_nie_auc  = 0.77
print(f'gbdt Chi2 BACC Zscore {zmap(gbdt_chi2_nie_bacc, gbdt_chi2_bacc)}')
print(f'gbdt Chi2 AUC Zscore {zmap(gbdt_chi2_nie_auc, gbdt_chi2_auc)}')


gbdt_elnet_bacc = TABLE3_BACCS["gbdt_elnet_bacc"]
gbdt_elnet_auc = TABLE3_AUCS["gbdt_elnet_auc"]
gbdt_elnet_nie_bacc = 0.70
gbdt_elnet_nie_auc  = 0.76
print(f'gbdt elnet BACC Zscore {zmap(gbdt_elnet_nie_bacc, gbdt_elnet_bacc)}')
print(f'gbdt elnet AUC Zscore {zmap(gbdt_elnet_nie_auc, gbdt_elnet_auc)}')
#####################################

xgb_full_bacc = TABLE3_BACCS["xgb_full_bacc"]
xgb_full_auc = TABLE3_AUCS["xgb_full_auc"]
xgb_full_nie_bacc = 0.68 
xgb_full_nie_auc  = 0.76
print(f'xgb Full Features BACC Zscore {zmap(xgb_full_nie_bacc, xgb_full_bacc)}')
print(f'xgb Full Features AUC Zscore {zmap(xgb_full_nie_auc, xgb_full_auc)}')


xgb_chi2_bacc = TABLE3_BACCS["xgb_chi2_bacc"]
xgb_chi2_auc = TABLE3_AUCS["xgb_chi2_auc"]
xgb_chi2_nie_bacc = 0.67
xgb_chi2_nie_auc  = 0.73
print(f'xgb Chi2 BACC Zscore {zmap(xgb_chi2_nie_bacc, xgb_chi2_bacc)}')
print(f'xgb Chi2 AUC Zscore {zmap(xgb_chi2_nie_auc, xgb_chi2_auc)}')


xgb_elnet_bacc = TABLE3_BACCS["xgb_elnet_bacc"]
xgb_elnet_auc = TABLE3_AUCS["xgb_elnet_auc"]
xgb_elnet_nie_bacc = 0.68 
xgb_elnet_nie_auc  = 0.76
print(f'xgb elnet Features BACC Zscore {zmap(xgb_elnet_nie_bacc, xgb_elnet_bacc)}')
print(f'xgb elnet Features AUC Zscore {zmap(xgb_elnet_nie_auc, xgb_elnet_auc)}')

##########################################
l2reg_full_bacc = TABLE3_BACCS["l2reg_full_bacc"]
l2reg_full_auc = TABLE3_AUCS["l2reg_full_auc"]
l2reg_full_nie_bacc = 0.64
l2reg_full_nie_auc  = 0.69
print(f'l2reg Full Features BACC Zscore {zmap(l2reg_full_nie_bacc, l2reg_full_bacc)}')
print(f'l2reg Full Features AUC Zscore {zmap(l2reg_full_nie_auc, l2reg_full_auc)}')


l2reg_chi2_bacc = TABLE3_BACCS["l2reg_chi2_bacc"]
l2reg_chi2_auc = TABLE3_AUCS["l2reg_chi2_auc"]
l2reg_chi2_nie_bacc = 0.71
l2reg_chi2_nie_auc  = 0.73
print(f'l2reg Chi2 BACC Zscore {zmap(l2reg_chi2_nie_bacc, l2reg_chi2_bacc)}')
print(f'l2reg Chi2 AUC Zscore {zmap(l2reg_chi2_nie_auc, l2reg_chi2_auc)}')


l2reg_elnet_bacc = TABLE3_BACCS["l2reg_elnet_bacc"]
l2reg_elnet_auc = TABLE3_AUCS["l2reg_elnet_auc"]
l2reg_elnet_nie_bacc = 0.71 
l2reg_elnet_nie_auc  = 0.77
print(f'l2reg elnet Features BACC Zscore {zmap(l2reg_elnet_nie_bacc, l2reg_elnet_bacc)}')
print(f'l2reg elnet Features AUC Zscore {zmap(l2reg_elnet_nie_auc, l2reg_elnet_auc)}')

##########################################

elastic_net_bacc = TABLE3_BACCS["elastic_net_bacc"]
elastic_net_auc = TABLE3_AUCS["elastic_net_auc"]
elastic_net_nie_bacc = 0.68 
elastic_net_nie_auc  = 0.76
print(f'Elastic_net BACC Zscore {zmap(elastic_net_nie_bacc, elastic_net_bacc)}')
print(f'Elastic_net AUC Zscore {zmap(elastic_net_nie_auc, elastic_net_auc)}')
