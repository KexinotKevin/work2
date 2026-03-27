
atlases = {
    "ATLAS_NAME":[
        "aal116",
        "aal120",
        "bna246",
        "glasser360_S1",
        "glasser360_S4",
        "schaefer1000_S4",
        "schaefer200_S1",
        "schaefer500_S4"
    ],
    "ATLAS_DIR": "/media/shulab/WD_10T/datasets/utils/mergedAtlas/Lin6/"
}

datasets = {
    "HCD":{
        "conn_dir": "/media/shulab/WD_10T/datasets/HCD/network",
        "scores_path": "/media/shulab/WD_10T/datasets/HCD/textfiles/cogcomp.csv",
        "demographics_path":"/media/shulab/WD_10T/datasets/HCD/textfiles/HCD_LS_2.0_subject_completeness.csv",
        "tgt_label_list": ['src_subject_id', 'sex', 'interview_age', 
            'nih_fluidcogcomp_unadjusted', 'nih_fluidcogcomp_ageadjusted', 
            'nih_crycogcomp_unadjusted', 'nih_crycogcomp_ageadjusted', 
            'nih_eccogcomp_unadjusted', 'nih_eccogcomp_ageadjusted', 
            'nih_totalcogcomp_unadjusted', 'nih_totalcogcomp_ageadjusted', 
            ],
        "tgt_demo_list": ['src_subject_id', 'sex', 'interview_age', 
            'nih_fluidcogcomp_unadjusted', 'nih_fluidcogcomp_ageadjusted', 
            # 'nih_fluidcogcomp_np', 'nih_fluidcogcomp_fctsc', 
            'nih_crycogcomp_unadjusted', 'nih_crycogcomp_ageadjusted', 
            # 'nih_crystalcogcomp_np', 'nih_crystalcogcomp_fctsc', 
            'nih_cogfuncogcomp_unadj', 'nih_cogfuncogcomp_ageadj', 
            # 'nih_cogfuncogcomp_np', 'nih_cogfuncogcomp_fctsc', 
            'nih_eccogcomp_unadjusted', 'nih_eccogcomp_ageadjusted', 
            # 'nih_earlychildcogcomp_np', 'nih_earlychildcogcomp_fctsc', 
            'nih_totalcogcomp_unadjusted', 'nih_totalcogcomp_ageadjusted', 
            # 'nih_totalcogcomp_np', 'nih_totalcogcomp_fctsc'
            ],
    },
    
    "ABCD":{
        "conn_dir": "/media/shulab/WD_10T/datasets/ABCD/network/site16",
        # "scores_path": "/media/shulab/WD_10T/datasets/ABCD/textfiles/neurocognition/nc_y_nihtb.csv",
        "scores_path":"/media/shulab/WD_10T/datasets/ABCD/textfiles/abcd_baseline_rearranged_scores.csv",
        "demographics_path": "/media/shulab/WD_10T/datasets/ABCD/textfiles/abcd-general/abcd_p_demo.csv",
        "tgt_label_list": ['src_subject_id', 'demo_sex_v2', 'demo_brthdat_v2',
            'nihtbx_fluidcomp_uncorrected', 'nihtbx_fluidcomp_agecorrected',
            'nihtbx_cryst_uncorrected', 'nihtbx_cryst_agecorrected',
            'nihtbx_totalcomp_uncorrected', 'nihtbx_totalcomp_agecorrected'],
        "tgt_demo_list": ['src_subject_id', 'demo_sex_v2', 'demo_brthdat_v2',
            # 'nihtbx_picvocab_uncorrected', 'nihtbx_picvocab_agecorrected',
            # 'nihtbx_flanker_uncorrected', 'nihtbx_flanker_agecorrected',
            # 'nihtbx_list_uncorrected', 'nihtbx_list_agecorrected',
            # 'nihtbx_cardsort_uncorrected', 'nihtbx_cardsort_agecorrected',
            # 'nihtbx_pattern_uncorrected', 'nihtbx_pattern_agecorrected',
            # 'nihtbx_picture_uncorrected', 'nihtbx_picture_agecorrected',
            # 'nihtbx_reading_uncorrected', 'nihtbx_reading_agecorrected',
            'nihtbx_fluidcomp_uncorrected', 'nihtbx_fluidcomp_agecorrected',
            'nihtbx_cryst_uncorrected', 'nihtbx_cryst_agecorrected',
            'nihtbx_totalcomp_uncorrected', 'nihtbx_totalcomp_agecorrected'],
    },
    
    "S1200":{
        "conn_dir": "/media/shulab/WD_10T/datasets/S1200/network",
        "scores_path": "/media/shulab/WD_10T/datasets/S1200/textfiles/S1200_889_CogScores.csv",
        "demographics_path":"/media/shulab/WD_10T/datasets/S1200/textfiles/S1200_free_demographics_full-MRI_889.csv",
        "tgt_label_list": ['Subject', 'Gender', 'Age', 
                           'CogFluidComp_Unadj', 'CogFluidComp_AgeAdj',
                           'CogEarlyComp_Unadj', 'CogEarlyComp_AgeAdj', 'CogTotalComp_Unadj',
                           'CogTotalComp_AgeAdj', 'CogCrystalComp_Unadj', 'CogCrystalComp_AgeAdj'],
        "tgt_demo_list": []
    },
    
    "UKB":{
        "conn_dir": "",
        "scores_path": "",
        "demographics_path":"",
        "tgt_label_list": [],
        "tgt_demo_list": []
    }
}

def get_dataset_cfg(dataset_key):
    dataset_key_upper = str(dataset_key).upper()
    for valid_name in datasets:
        if valid_name.upper() == dataset_key_upper:
            return datasets[valid_name]
    raise KeyError(
        f"Unknown dataset '{dataset_key}'. Valid options: {sorted(datasets)}"
    )
