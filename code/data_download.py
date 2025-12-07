import os
from tqdm import tqdm
import vitaldb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import torch
import random
from vital_code import VITAL_COLS, MEDICATION_COLS
from preprocessing import clip_to_physiological_range
from state_action_reward import StateActionRewardDesigner

track_list = VITAL_COLS + MEDICATION_COLS

def save_data(data_direc, case_list, history_length, random_state=67):
    designer = StateActionRewardDesigner()
    all_states = []
    all_actions = []
    train_direc = os.path.join(data_direc, 'train')
    valid_direc = os.path.join(data_direc, 'valid')
    os.makedirs(train_direc, exist_ok=True)
    os.makedirs(valid_direc, exist_ok=True)
    for (case_id, df) in tqdm(case_list):
        
        
        if df is None or len(df) == 0:
            print(f"  Skipping case {case_id}: No data")
            continue
        
        # 최소 길이 체크
        if len(df) < history_length + 10:
            print(f"  Skipping case {case_id}: Too short")
            continue
        
        df_processed = clip_to_physiological_range(df)
        
        # State와 Action 추출
        for idx in range(history_length, len(df_processed), history_length//2):
            state = designer.extract_state(df_processed, idx, history_length)
            if np.isnan(state).any():
                continue
            action = designer.extract_action(df_processed, idx, history_length)
            all_states.append(state)
            all_actions.append(action)

    train_state, valid_state, train_action, valid_action = train_test_split(all_states, all_actions, random_state=random_state, test_size=0.3)
    train_state = np.stack(train_state, axis=0)
    valid_state = np.stack(valid_state, axis=0)
    train_action = np.stack(train_action, axis=0)
    valid_action = np.stack(valid_action, axis=0)



    


    np.save(os.path.join(train_direc, f'state_{history_length}.npy'), train_state)
    np.save(os.path.join(train_direc, f'action_{history_length}.npy'), train_action)
    np.save(os.path.join(valid_direc, f'state_{history_length}.npy'), valid_state)
    np.save(os.path.join(valid_direc, f'action_{history_length}.npy'), valid_action)

def get_params():
    
    parser = argparse.ArgumentParser(description="Data Path parameter")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to your data directory",
        default='./dataset'
    )
    parser.add_argument(
        "--history_length",
        type=int,
        help="Size of time series data",
        default=30
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Size of time series data",
        default=67
    )
    return parser

if __name__ == '__main__':

    args = get_params().parse_args()
    data_dir = os.path.abspath(args.data_dir)
    clinical_data_path = os.path.join(data_dir, 'clinical_data.csv')
    assert os.path.exists(clinical_data_path), f"Clinical CSV file not found: {clinical_data_path}"
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_save_dir = os.path.join(data_dir, 'signal_data')
    clinical_data = pd.read_csv(clinical_data_path)
    history_length = args.history_length
    os.makedirs(data_save_dir, exist_ok=True)


    data_pd_list = list()
    for subjid in tqdm(clinical_data['caseid']):
        vf = vitaldb.VitalFile(subjid, track_list, interval=2)
        if set(vf.get_track_names()) != set(track_list):
            continue
        data_pd_list.append((subjid, vf.to_pandas(track_list, 2, return_timestamp=True)))
    
    save_data(data_save_dir, data_pd_list, history_length, random_state=seed)