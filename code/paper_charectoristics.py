import os
import sys
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

# Analyses the data matrices to produce charectoristics of the study cohorts
# Run supplying directory with the X and y matrices for the ext validation portion of the study
# 

def write_overlapping_characteristics(data_dir, results_dir):
    """ Writes characteristics of the overlapping datasets toa file, takes in directory of the X and y
    matrices, will place csv file output there too"""
        
    # Read Matrices as pandas dataframes
    X_cb_extval = pd.read_csv(data_dir + "/2_ExternalValidation/" + "X_test_cb_extval.csv")
    X_sd_extval = pd.read_csv(data_dir + "/2_ExternalValidation/" + "X_train_stard_extval.csv")
    y_cb_extval = pd.read_csv(data_dir + "/2_ExternalValidation/" + "y_test_cb_extval.csv")
    y_sd_extval = pd.read_csv(data_dir + "/2_ExternalValidation/" + "y_train_stard_extval.csv")
    
    # Read the pre-overlapping matrices for some data needed for certain stuff. Manually point to these older pre-generate overlapping files
    cb_pre = pd.read_csv("./final_datasets/" + "canbind_imputed.csv")
    sd_pre = pd.read_csv("./final_datasets/" + "X_wk8_response_qids01__final.csv")
    
    X_cb_extval['Employed'] = cb_pre['EMPLOY_STATUS_1.0']
    X_sd_extval['Employed'] = sd_pre[['dm01_enroll__empl||3.0', 'dm01_enroll__empl||4.0']].max(axis=1)

    ##print(X_sd_extval['Employed'].iloc[0:8])


    # Ensure number of subjects same between X and y matrices
    assert len(X_cb_extval) == len(y_cb_extval) and len(X_sd_extval) == len(y_sd_extval), "X and y must have same length"
    sd_n = len(X_sd_extval)
    cb_n = len(X_cb_extval)
      
    # Open file to write to. Write manually to allow strings
    f = open(results_dir + "/" + "paper_overlapping_characteristics.csv", "w")
    f.write("Characteristic, STAR*D,, CAN-BIND,\n")
    
    
    # Wrtie line for n and %
    f.write(',n,%,n,%\n')
    
    
    # Female:Male. Write manually as aytypical
    defin = "Female:Male (%)"
    
    female_sd = X_sd_extval['SEX_female:::gender||F'].value_counts().to_dict()[1]
    male_sd = X_sd_extval['SEX_male:::gender||M'].value_counts().to_dict()[1]
    female_cb = X_cb_extval['SEX_female:::gender||F'].value_counts().to_dict()[1]
    male_cb = X_cb_extval['SEX_male:::gender||M'].value_counts().to_dict()[1]
    
    female_sd_perc = "{:.1f}".format(100*female_sd/sd_n)
    male_sd_perc = "{:.1f}".format(100*male_sd/sd_n)
    female_cb_perc = "{:.1f}".format(100*female_cb/cb_n)
    male_cb_perc = "{:.1f}".format(100*male_cb/cb_n)
    
    f.write(defin + "," + str(female_sd) + ":" + str(male_sd) + " ," + female_sd_perc + ":" + male_sd_perc + ",")
    f.write(str(female_cb) + ":" + str(male_cb) + " ," + female_cb_perc + ":" + male_cb_perc + "\n")
    
    # Cannot do ethnicity, missing from STAR*D
    
    # Marital status 
    married_sd = X_sd_extval['MRTL_STATUS_Married:::dm01_enroll__marital||3.0'].value_counts().to_dict()[1]
    married_cb = X_cb_extval['MRTL_STATUS_Married:::dm01_enroll__marital||3.0'].value_counts().to_dict()[1]
    write_characteristic_perc("Married (%)", married_sd, sd_n, married_cb, cb_n, f)
    
    dompart_sd = X_sd_extval['MRTL_STATUS_Domestic Partnership:::dm01_enroll__marital||2.0'].value_counts().to_dict()[1]
    dompart_cb = X_cb_extval['MRTL_STATUS_Domestic Partnership:::dm01_enroll__marital||2.0'].value_counts().to_dict()[1]
    write_characteristic_perc("Domestic Partnership (%)", dompart_sd, sd_n, dompart_cb, cb_n, f)
    
    widow_sd = X_sd_extval['MRTL_STATUS_Widowed:::dm01_enroll__marital||6.0'].value_counts().to_dict()[1]
    widow_cb = X_cb_extval['MRTL_STATUS_Widowed:::dm01_enroll__marital||6.0'].value_counts().to_dict()[1]
    write_characteristic_perc("Widowed (%)", widow_sd, sd_n, widow_cb, cb_n, f)
    
    divorc_sd = X_sd_extval['MRTL_STATUS_Divorced:::dm01_enroll__marital||5.0'].value_counts().to_dict()[1]
    divorc_cb = X_cb_extval['MRTL_STATUS_Divorced:::dm01_enroll__marital||5.0'].value_counts().to_dict()[1]
    write_characteristic_perc("Divorced (%)", divorc_sd, sd_n, divorc_cb, cb_n, f)
    
    separ_sd = X_sd_extval['MRTL_STATUS_Separated:::dm01_enroll__marital||4.0'].value_counts().to_dict()[1]
    separ_cb = X_cb_extval['MRTL_STATUS_Separated:::dm01_enroll__marital||4.0'].value_counts().to_dict()[1]
    write_characteristic_perc("Separated (%)", separ_sd, sd_n, separ_cb, cb_n, f)
    
    never_sd = X_sd_extval['MRTL_STATUS_Never Married:::dm01_enroll__marital||1.0'].value_counts().to_dict()[1]
    never_cb = X_cb_extval['MRTL_STATUS_Never Married:::dm01_enroll__marital||1.0'].value_counts().to_dict()[1]
    write_characteristic_perc("Never Married (%)", never_sd, sd_n, never_cb, cb_n, f)
    
    # Any substance-use
    # First make two new columns for any substance use, by taking the max of the two substance columns (alch, non-alch)
    X_cb_extval['substance'] = X_cb_extval[['MINI_ALCHL_ABUSE_TIME:::phx01__alcoh||1.0', 'MINI_SBSTNC_ABUSE_NONALCHL_TIME:::phx01__amphet||1.0']].max(axis=1)
    X_sd_extval['substance'] = X_sd_extval[['MINI_ALCHL_ABUSE_TIME:::phx01__alcoh||1.0', 'MINI_SBSTNC_ABUSE_NONALCHL_TIME:::phx01__amphet||1.0']].max(axis=1)
    
    col = 'substance'
    defin = "Substance use Disorder (%)"
    sd = X_sd_extval[col].value_counts().to_dict()[1]
    cb = X_cb_extval[col].value_counts().to_dict()[1]
    write_characteristic_perc(defin, sd, sd_n, cb, cb_n, f)

    # Any anxiety disorder
    col = ':::imput_anyanxiety' 
    defin = "Any Anxiety Disorder (%)"
    sd = X_sd_extval[col].value_counts().to_dict()[1]
    cb = X_cb_extval[col].value_counts().to_dict()[1]
    write_characteristic_perc(defin, sd, sd_n, cb, cb_n, f)
    

    
    
    # Employment Status
    sd = X_sd_extval['EMPLOY_STATUS_1.0:::dm01_enroll__empl||3.0'].value_counts().to_dict()[1]
    cb = X_cb_extval['EMPLOY_STATUS_1.0:::dm01_enroll__empl||3.0'].value_counts().to_dict()[1]
    write_characteristic_perc("Working/Student (%)", sd, sd_n, cb, cb_n, f)
    
    sd = X_sd_extval['EMPLOY_STATUS_2.0:::dm01_enroll__empl||1.0'].value_counts().to_dict()[1]
    cb = X_cb_extval['EMPLOY_STATUS_2.0:::dm01_enroll__empl||1.0'].value_counts().to_dict()[1]
    write_characteristic_perc("Disabled(%)", sd, sd_n, cb, cb_n, f)
    
    sd = X_sd_extval['EMPLOY_STATUS_5.0:::dm01_enroll__empl||2.0'].value_counts().to_dict()[1]
    cb = X_cb_extval['EMPLOY_STATUS_5.0:::dm01_enroll__empl||2.0'].value_counts().to_dict()[1]
    write_characteristic_perc("Unemployed (%)", sd, sd_n, cb, cb_n, f)
    
    sd = X_sd_extval['EMPLOY_STATUS_7.0:::dm01_enroll__empl||6.0'].value_counts().to_dict()[1]
    cb = X_cb_extval['EMPLOY_STATUS_7.0:::dm01_enroll__empl||6.0'].value_counts().to_dict()[1]
    write_characteristic_perc("Retired (%)", sd, sd_n, cb, cb_n, f)
    
    # line for mean and %
    f.write(', Mean, SD, Mean, SD\n')
    
    
    # Employment Status: Hours worked if employed    
    hrs_wrk_mn_sd = X_sd_extval[X_sd_extval['Employed'] > 0]['LAM_2_baseline:::wpai01__wpai_totalhrs'].mean()
    hrs_wrk_mn_cb = X_cb_extval[X_cb_extval['Employed'] > 0]['LAM_2_baseline:::wpai01__wpai_totalhrs'].mean()
                                
    hrs_wrk_std_sd = X_sd_extval[X_sd_extval['Employed'] > 0]['LAM_2_baseline:::wpai01__wpai_totalhrs'].std()
    hrs_wrk_std_cb = X_cb_extval[X_cb_extval['Employed'] > 0]['LAM_2_baseline:::wpai01__wpai_totalhrs'].std()
    defin = "Hours worked over last two weeks if employed (SD)"
    f.write(defin + "," + "{:.1f}".format(hrs_wrk_mn_sd) +" ," + "{:.1f}".format(hrs_wrk_std_sd) + ",")
    f.write("{:.1f}".format(hrs_wrk_mn_cb) +" ," + "{:.1f}".format(hrs_wrk_std_cb) + ",\n")
    
    # Employment Status: Hours missed if employed
    hrs_msd_mn_sd = X_sd_extval[X_sd_extval['Employed'] > 0]['LAM_3_baseline:::wpai01__wpai02'].mean()
    hrs_msd_mn_cb = X_cb_extval[X_cb_extval['Employed'] > 0]['LAM_3_baseline:::wpai01__wpai02'].mean()
    
    hrs_msd_std_sd = X_sd_extval[X_sd_extval['Employed'] > 0]['LAM_3_baseline:::wpai01__wpai02'].std()
    hrs_msd_std_cb = X_cb_extval[X_cb_extval['Employed'] > 0]['LAM_3_baseline:::wpai01__wpai02'].std()
    defin = "Hours missed from illness over last two weeks if employed"
    f.write(defin + "," + "{:.1f}".format(hrs_msd_mn_sd) +"," + "{:.1f}".format(hrs_msd_std_sd) + ",")
    f.write("{:.1f}".format(hrs_msd_mn_cb) +" ," + "{:.1f}".format(hrs_msd_std_cb) + "\n")
    
    # Education
    defin = "Years of education (SD)"
    edu_years_mn_sd = X_sd_extval['EDUC:::dm01_enroll__educat'].mean()
    edu_years_mn_cb = X_cb_extval['EDUC:::dm01_enroll__educat'].mean()
        
    edu_years_std_sd = X_sd_extval['EDUC:::dm01_enroll__educat'].std()
    edu_years_std_cb = X_cb_extval['EDUC:::dm01_enroll__educat'].std()
    write_characteristic_cust(defin, edu_years_mn_sd, edu_years_std_sd, edu_years_mn_cb, edu_years_std_cb, f)

    # Age at onset of first depression PSYHIS_MDD_AGE:::phx01__dage
    defin = "Age in years at onset of depression (SD)"
    col = 'PSYHIS_MDD_AGE:::phx01__dage'
    sd = X_sd_extval[col].mean()
    cb = X_cb_extval[col].mean()
        
    sd_br = X_sd_extval[col].std()
    cb_br = X_cb_extval[col].std()
    write_characteristic_cust(defin, sd, sd_br, cb, cb_br, f)
    
    # Prior depressive episode
    col = 'PSYHIS_MDD_PREV:::'
    defin = "Prior depressive episode (%)"
    sd = X_sd_extval[col].value_counts().to_dict()[1]
    cb = X_cb_extval[col].value_counts().to_dict()[1]
    write_characteristic_perc(defin, sd, sd_n, cb, cb_n, f)
    
    # No. prior episodes. Subtract 1 as the col counts current episode
    defin = "No. of prior depressive episodes (SD)"
    col = 'PSYHIS_MDE_NUM:::phx01__epino'
    sd = X_sd_extval[col].sub(1).mean()
    cb = X_cb_extval[col].sub(1).mean()
        
    sd_br = X_sd_extval[col].sub(1).std()
    cb_br = X_cb_extval[col].sub(1).std()
    write_characteristic_cust(defin, sd, sd_br, cb, cb_br, f)
    
    # Current episode duration. Might want to cat this like in the paper
    defin = "Current episode duration in months (SD)"
    col = 'PSYHIS_MDE_EP_DUR_MO:::phx01__episode_date'
    sd = X_sd_extval[col].mean()
    cb = X_cb_extval[col].mean()
        
    sd_br = X_sd_extval[col].std()
    cb_br = X_cb_extval[col].std()
    write_characteristic_cust(defin, sd, sd_br, cb, cb_br, f)

    
    # QIDS-SR Total Baseline
    col = 'QIDS_OVERL_SEVTY_baseline:::qids01_w0sr__qstot'
    defin = "Baseline QIDS-SR Total Score (SD)"
    sd = X_sd_extval[col].mean()
    cb = X_cb_extval[col].mean()
        
    sd_br = X_sd_extval[col].std()
    cb_br = X_cb_extval[col].std()
    write_characteristic_cust(defin, sd, sd_br, cb, cb_br, f)
    
    # QIDS-SR Total Week 2
    col = 'QIDS_OVERL_SEVTY_week2:::qids01_w2sr__qstot'
    defin = "Week 2 QIDS-SR Total Score (SD)"
    sd = X_sd_extval[col].mean()
    cb = X_cb_extval[col].mean()
        
    sd_br = X_sd_extval[col].std()
    cb_br = X_cb_extval[col].std()
    write_characteristic_cust(defin, sd, sd_br, cb, cb_br, f)
    
    # Consider some more, such as co-morbidites, or use of antidepressant in current
    
    # Probably also want some baseline numbers like baseline QIDS_C, QIDS_SR
    
    f.close()

def write_characteristic_perc(defin, sd, sd_n, cb, cb_n, f):
        """ Helper function that writes a line of the characteristics file
        defin -- string definition of the characteristic
        sd --  integer number of a certain characteristic
        sdtot -- int of total number of subjects in the STAR*D cohort
        cb -- integer number of a certain characteristic
        cbtot -- int of total number of subjects in the CANBIND cohort
        f -- file to write to"""
        
        # Calculate percentages    
        sdperc = "{:.1f}".format(100*sd/sd_n)
        cbperc = "{:.1f}".format(100*cb/cb_n)
        
        # Write to file
        f.write(defin + "," + str(sd) + " ," + sdperc + ", " + str(cb) + " ," + cbperc + "\n" )

def write_characteristic_cust(defin, sd, sd_br, cb, cb_br, f):
        """ Helper function that writes a line of the characteristics file, allowing custom values to be entered into the parens
        defin -- string definition of the characteristic
        sd -- STAR*D value to be outside parens when written
        sd_br -- STAR*D value written inside the parens 
        cb -- CANBIND value outside the parens
        cb_br -- CANBIND value inside the parens 
        f -- file to write to"""
                
        # Write to file
        f.write(defin + "," + "{:.1f}".format(sd) + " ," + "{:.1f}".format(sd_br) + ", " + "{:.1f}".format(cb) + " ," + "{:.1f}".format(cb_br) + "\n" )

def write_top_features(data_dir, results_dir):
    top_chi_10 = [81,74,275,76,80,457,452,413,455,459]
    top_chi_30 = [81,74,275,76,80,457,452,413,455,459,29,466,396,379,477,426,456,78,2,362,99,463,458,465,462,345,454,448,471,470]
    
    X_cb_extval = pd.read_csv(data_dir + "/1_Replication/" + "X_lvl2_rem_qids01__final.csv")
    cols = list(X_cb_extval.columns)
    
    f = open(results_dir + "/" + "top_features.csv", "w")
    f.write("Top 10 features using the chi-squared feature selection \n")
    
    for i in top_chi_30:
        f.write(cols[i + 1] + ", " + str(i) + "\n") # Plus one as the first column was removed for running the ML
    f.close()

def z_tests(results_dir):
    stat, pval = proportions_ztest([368, 344], [504, 491], 0.05, 'two-sided')
    print('{0:0.3f}'.format(pval))
    
    #us 367.92 out of 504, accuracy 0.73
    #them348.1 out of 490.6, best accuracy 0.70
    

if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        data_dir = sys.argv[1]
        
    if len(sys.argv) == 1:
        data_dir = "./final_datasets/to_run_20201016/"
        results_dir = "./final_datasets/previous results/"
        write_overlapping_characteristics(data_dir, results_dir)
        write_top_features(data_dir, results_dir)
        z_tests(results_dir)
    else:
        print("Enter valid argument\n"
              "\t path: the path to a real directory containing the final datasets\n"
               "\t path: the path to a real directory to put the output files ")
    