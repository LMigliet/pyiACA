
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm

import python_libraries.fitting_func as fitfunc



def process_qPCR_data(exp_dir, exp_id):
    # this is from LC96 output
    qpcr_data_dir = exp_dir / str(exp_id+"_qPCR_data")
    exp_file = list(Path(exp_dir).glob('*.xlsx'))[0]

    qpcr_data_ac = qpcr_data_dir / str(exp_id+"_qPCR_AC.csv")
    df_ac = pd.read_csv(qpcr_data_ac) # amplification curves
    df_ac["Target"] = df_ac["Target"].str.lower()
    df_ac.insert(loc=5, column="Exp_id", value=exp_id)
    df_ac = df_ac.iloc[:,1:] # removing channel PCR plate column

    qpcr_data_mc = qpcr_data_dir / str(exp_id+"_qPCR_MC.csv")
    df_mc= pd.read_csv(qpcr_data_mc) # melting curves
    df_mc["Target"] = df_mc["Target"].str.lower()
    df_mc.insert(loc=5, column="Exp_id", value=exp_id)
    df_mc = df_mc.iloc[:,1:] # removing channel PCR plate column

    return df_ac, df_mc


def process_qPCR_TAQ_data(exp_dir, exp_id):
    # this is from TaqMan probes - QIAquanta96 output
    qpcr_data_dir = exp_dir / str(exp_id+"_qPCR_data")
    exp_file = list(Path(exp_dir).glob('*.xlsx'))[0]

    df_temp = pd.read_excel(exp_file, sheet_name="qPCR_plate_design", header=52)
    df_meta = df_temp.iloc[:,1:7]
    
    qpcr_data_file = qpcr_data_dir / str(exp_id+"_qPCR_AC.csv")
    df_ampl_raw_data = pd.read_csv(qpcr_data_file, header=19) # amplification curves
    
    # setting up your colour:
    df_fam_raw = df_ampl_raw_data.iloc[0:96,:]
    df_fam = df_meta.merge(df_fam_raw.iloc[:,1:], on=[df_meta.iloc[:,0]])
    df_fam = df_fam.iloc[:,1:] # removing the well index (duplicate)
    
    df_fam["Target"] = df_fam["Target"].str.lower()
    df_fam.insert(loc=5, column="Exp_id", value=exp_id)

    return df_fam


def process_dPCR_data(exp_dir, exp_id):

    # from one experiment forlder get the directory and the ID (which is the data plus number of plate, i.e. 20201101_01)
    # Search for the dPCR "Extracted_data folder!
    dpcr_data_dir = exp_dir / str(exp_id+"_dPCR_data") / "Extracted_data"
    
    # Search for the excel file in the folder of the exp.
    exp_file = list(Path(exp_dir).glob('*.xlsx'))[0]
    
    # get the metadata from the excel file in the plate sheet
    df_meta = pd.read_excel(exp_file, sheet_name="dPCR_plate_design", header = 32)
    dpcr_metadata = df_meta.iloc[:48,1:6]

    # add exp_id column
    dpcr_metadata.loc[:,"Exp_id"] = exp_id

    # sorting the data collected in the Extracted_data folder
    AC_files = sorted([file for file in dpcr_data_dir.glob('*AC*')])
    MC_files = [Path(str(file).replace('_AC', '_MC')) for file in AC_files]

    ###### AC DFs ######
    
    df_acs = []
    
    for AC_file in tqdm(AC_files, desc="Reading AC Files", leave=False):
        panel_id = AC_file.name[0:7] # panel id name (i.e. panel01)
        df_meta_panel = dpcr_metadata.loc[dpcr_metadata.Channel == panel_id] # based on the channel get the metdata

        df_ac_single = fitfunc.extract_curves(AC_file).T # extract curves with the function
        df_ac_single = df_ac_single.astype(float)
        df_ac_single["Channel"] = panel_id
        df_ac_single = df_ac_single.reset_index(drop=True)

        # join metadata and AC data in a DF
        df_ac = df_meta_panel.merge(df_ac_single, how='left', left_on = "Channel", right_on = "Channel")
        df_acs.append(df_ac)
        
    ###### MC DFs ######
    
    df_mcs = []
    
    for MC_file in tqdm(MC_files, desc="Reading MC Files", leave=False):
        panel_id = MC_file.name[0:7]
        df_meta_panel = dpcr_metadata.loc[dpcr_metadata.Channel == panel_id]
        
        df_mc_single = fitfunc.extract_curves(MC_file).T 
        df_mc_single.columns = np.linspace(65, 97, num=(97-65)*2+1)
        df_mc_single = df_mc_single.iloc[:, :-1] # drop last column
        df_mc_single = df_mc_single.astype(float)
        df_mc_single["Channel"] = panel_id
        df_mc_single = df_mc_single.reset_index(drop=True)

        df_mc = df_meta_panel.merge(df_mc_single, how='left', left_on = "Channel", right_on = "Channel")
        df_mcs.append(df_mc)
    
    print("Congrats, data correctly processed for:", exp_id)
    return df_acs, df_mcs
