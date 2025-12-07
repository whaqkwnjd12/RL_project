import pandas as pd
from vital_code import *

def clip_to_physiological_range(df):
    vital_ranges = {
        ART_SBP: (50, 250),   # mmHg
        ART_MBP: (30, 150),   # mmHg
        HR: (30, 200),         # bpm
        SPO2: (70, 100),       # %
    }
    medication_ranges = {
        PPF20_RATE: (0, 500),     # mL/hr
        RFTN20_RATE: (0, 400),    # mL/hr
        PPF20_VOL: (0, 10000),    # mL
        RFTN20_VOL: (0, 1000),    # mL
    }
    """생리학적으로 타당한 범위로 클리핑"""
    df_clipped = df.copy()
    
    for col, (min_val, max_val) in {**vital_ranges, **medication_ranges}.items():
        if col in df_clipped.columns:
            df_clipped[col] = df_clipped[col].clip(min_val, max_val)
            
    return df_clipped