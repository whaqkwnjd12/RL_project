import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import argparse
import random
import json
from datetime import datetime
warnings.filterwarnings('ignore')

from trainer import BehaviorCloningTrainer
from models import TransformerPolicy, MixtureOfGaussiansPolicy, TransformerPolicy, LSTMPolicy
from dataset import AnesthesiaDataset
from utils import get_device


def get_params():
    
    parser = argparse.ArgumentParser(description="Data Path parameter")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to your signal_data directory",
        default='./dataset'
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Path to your save directory",
        default='./results'
    )
    parser.add_argument(
        "--history_length",
        type=int,
        help="Size of time series data",
        default=10
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        help="Type of Policy Model",
        default='transformer'
    )
    return parser

def main(seed, hyper_params, save_direc, args):
    """메인 실행 함수"""
    
    # ========== 1. 데이터 준비 ==========
    print("=" * 60)
    print("STEP 1: Data Preparation")
    print("=" * 60)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    info_dict = dict()
    info_dict = {
        'seed': seed,
        'params': hyper_params
    }
    
    
    time_steps = args.history_length
    data_direc = args.data_dir
    X_train = np.load(os.path.join(data_direc, 'signal_data', 'train', f'state_{time_steps}.npy'))
    X_val = np.load(os.path.join(data_direc, 'signal_data', 'valid', f'state_{time_steps}.npy'))
    y_train = np.load(os.path.join(data_direc, 'signal_data', 'train', f'action_{time_steps}.npy'))
    y_val = np.load(os.path.join(data_direc, 'signal_data', 'valid', f'action_{time_steps}.npy'))
    
    scaler = MinMaxScaler()
    _, time_steps, feature_num = X_train.shape
    processed_X_train = scaler.fit_transform(X_train.reshape(-1, feature_num)).reshape(-1, time_steps, feature_num)
    processed_X_val = scaler.transform(X_val.reshape(-1, feature_num)).reshape(-1, time_steps, feature_num)

    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Val set: {len(X_val)} samples")
    
    # PyTorch Dataset 생성
    train_dataset = AnesthesiaDataset(processed_X_train, y_train, X_train)
    val_dataset = AnesthesiaDataset(processed_X_val, y_val, X_val)
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=1024, shuffle=True, num_workers=1
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1024, shuffle=False, num_workers=1
    )
    
    # ========== 2. 모델 선택 및 학습 ==========
    print("\n" + "=" * 60)
    print("STEP 2: Model Training")
    print("=" * 60)
    
    state_dim = X_train.shape[-1]
    action_dim = y_train.shape[-1]
    # 여러 모델 중 선택 가능
    policy_type = args.policy_type  # 'lstm', 'mog', 'transformer'
    info_dict['policy_type'] = policy_type
    
    if policy_type == 'lstm':
        policy = LSTMPolicy(state_dim, action_dim,
                            **hyper_params,
         ) #  hidden_dim=128, num_layers=2
        
    elif policy_type == 'mog':
        policy = MixtureOfGaussiansPolicy(
            state_dim, action_dim, 
            **hyper_params, # num_gaussians=5, hidden_dim=256
        )
    else:  # transformer
        policy = TransformerPolicy(
            state_dim, action_dim,
            **hyper_params, # hidden_dim=128, num_heads=4, num_layers=3
        )

    info_dict['params']['state_dim'] = state_dim
    info_dict['params']['action_dim'] = action_dim
    
    print(f"\nModel: {policy_type.upper()}")
    print(f"Total parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Trainer 생성
    trainer = BehaviorCloningTrainer(
        policy=policy,
        learning_rate=1e-4,
        save_direc=save_direc,
        device = get_device(),
    )
    
    # 학습
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,
        policy_type=policy_type,
        # early_stopping_patience=20
    )
    
    # 학습 곡선 시각화
    trainer.plot_learning_curves()
    info_dict['performance'] = {
        'train_loss': trainer.train_losses[-1],
        'valid_loss': trainer.val_losses[-1],
        'train_reward': trainer.train_rewards[-1],
        'valid_reward': trainer.val_rewards[-1],
    }
    with open(os.path.join(save_direc, 'training_info.json'), 'w') as f:
        json.dump(info_dict, f, indent=4)

if __name__ == "__main__":
    args = get_params().parse_args()
    policy_type = args.policy_type

    # main(seed_num, param_set, save_direc, args=args)
    cnt = 0
    for seed_num in range(5):
        for layer_num in [1, 2, 3]:
            print(f"[{cnt}-th TRAINING]")
            cnt += 1
            serial = datetime.now().strftime("%m%d%H%M%S")
            save_direc = os.path.join(args.save_dir, policy_type, serial)
            os.makedirs(save_direc, exist_ok=True)
            param_set = dict(hidden_dim=128, num_heads=4, num_layers=layer_num)
            main(seed_num*20, param_set, save_direc, args=args)
