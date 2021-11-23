filename = "{}_{}_{}_{}_{}_{}_{}.txt".format(evl, model, runs,data, label, f_select, datetime.datetime.now().strftime("%Y%m%d-%H%M"))

evl = cross validation (cv) or external validation (extval)
model = ML model; rf (random forest) elnet (elastic net), gbdt, l2logreg, xgbt
runs = number of runs for each CV
data = the file for X. "X_full_trd" is TRD for replicating Nie et al. "X_full_resp" is full features with response subject selection, etc. 
label = y label. "y_all_trd" is full STAR*D TRD crtiera,"y_overlap_trd" is very similiar, used for TRD with only overlapping subjects, etc. 
f_select = feature selection
