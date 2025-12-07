import numpy as np
import pandas as pd
from vital_code import *
import torch

def compute_slope(series):
    """
    주어진 window의 시계열에서 linear regression slope 계산.
    series: numpy array of shape [window_length]
    """
    if len(series) <= 1:
        return 0.0  # slope 계산 불가할 때 0 처리

    y = np.array(series, dtype=float)
    x = np.arange(len(series), dtype=float)

    # variance(x) = 0이면 slope 계산이 안 되므로 방어 코드
    if np.var(x) < 1e-6:
        return 0.0

    slope = np.cov(x, y)[0, 1] / np.var(x)
    slope = slope * len(series)
    return slope

class StateActionRewardDesigner:
    """
    State, Action, Reward를 설계하는 클래스
    
    State: 환자의 현재 생체신호 + 약물 투여 이력
    Action: 약물 투여량 변화 (PPF20_RATE, RFTN20_RATE)
    Reward: 혈압 안정성 + 약물 효율성
    """
    
    def __init__(self):
        pass
        # 목표 혈압 범위 (저혈압 방지가 핵심)
        self.target_map_min = 65  # mmHg
        self.target_map_max = 100  # mmHg
        self.target_hr_min = 50   # bpm
        self.target_hr_max = 100  # bpm
        self.target_spo2_min = 95 # %
        
    def extract_state(self, df, idx, history_length = 10):
        """
        State 벡터 구성:
        1. 현재 vital signs (MBP, SBP, HR, SpO2)
        2. 최근 vital signs 통계 (평균, 표준편차, 최대, 최소)
        3. 현재 약물 투여율 (PPF20_RATE, RFTN20_RATE)
        4. 누적 약물량 (PPF20_VOL, RFTN20_VOL)
        5. 시간 특징 (수술 경과 시간)
        """
        start_idx = max(0, idx - history_length)
        window = df.iloc[start_idx:idx+1]

        # 필요한 feature만 선택 (순서 매우 중요)
        feature_cols = VITAL_COLS + MEDICATION_COLS

        # raw 시계열을 그대로 반환
        state = window[feature_cols].values.astype(np.float32)
        return state
        
    def extract_action(self, df, idx, history_length = 10):
        """
        Action 추출: 전문가(의사)가 선택한 약물 투여율
        
        연속 행동 공간:
        - PPF20_RATE 변화량
        - RFTN20_RATE 변화량
        """
        
        if idx == 0:
            # 첫 시점: 변화량 없음
            action = np.array([
                df.loc[idx,PPF20_RATE],
                df.loc[idx, RFTN20_RATE],
            ], dtype=np.float32)
        else:
            # 변화량 계산
            start = max(0, idx - history_length)
            window_ppf = df.loc[start:idx, PPF20_RATE]
            window_rftn = df.loc[start:idx, RFTN20_RATE]
            action = np.array([
                compute_slope(window_ppf),# compute_slope(window_ppf), df.loc[idx, PPF20_RATE] - df.loc[start, PPF20_RATE],
                compute_slope(window_rftn),#compute_slope(window_rftn), df.loc[idx, RFTN20_RATE] - df.loc[start, RFTN20_RATE],
            ], dtype=np.float32)
        
        return action

    def compute_reward(self, df_batch, predicted_actions):
        """
        df_batch : (B, T, F) torch.Tensor
            df_batch의 feature 순서: ART_SBP, ART_MBP, HR, SPO2, PPF20_VOL, PPF20_RATE, RFTN20_VOL, RFTN20_RATE
        predicted_actions : (B, 2) torch.Tensor  # [ΔPPF, ΔRFTN]
            predicted_action의 순서: PPF20_RATE, RFTN20_RATE
        return:
            reward : (B,) torch.Tensor
        
        """

        # 마지막 timestep의 vital signs를 추출 (B, F)
        last_state = df_batch[:, -1, :]

        ART_MBP_idx = 1
        HR_idx = 2
        SPO2_idx = 3

        map_val = last_state[:, ART_MBP_idx]
        hr_val  = last_state[:, HR_idx]
        spo2_val = last_state[:, SPO2_idx]

        # -----------------------------
        # 1. Physiology reward (batch-wise)
        # -----------------------------
        reward = torch.zeros(df_batch.shape[0], device=df_batch.device)

        # MAP reward
        in_range = (map_val >= self.target_map_min) & (map_val <= self.target_map_max)
        reward = reward + in_range.float() * 1.0

        low_mask = map_val < self.target_map_min
        reward = reward - low_mask.float() * (2.0 * (self.target_map_min - map_val) / self.target_map_min)

        high_mask = map_val > self.target_map_max
        reward = reward - high_mask.float() * (0.5 * (map_val - self.target_map_max) / self.target_map_max)

        # HR reward
        hr_range = (hr_val >= self.target_hr_min) & (hr_val <= self.target_hr_max)
        reward = reward + hr_range.float() * 0.5
        reward = reward - (~hr_range).float() * 0.3

        # SpO2 reward
        spo2_ok = spo2_val >= self.target_spo2_min
        reward = reward + spo2_ok.float() * 0.5
        reward = reward - (~spo2_ok).float() * (3.0 * (self.target_spo2_min - spo2_val) / self.target_spo2_min)

        # -----------------------------
        # 2. Action suitability reward
        # -----------------------------
        ppf_delta = predicted_actions[:, 0]
        rftn_delta = predicted_actions[:, 1]

        # 저혈압인데 Propofol 증가 → penalty
        mask_low_map = map_val < self.target_map_min
        reward = reward - (mask_low_map & (ppf_delta > 0)).float() * 2.0
        reward = reward + (mask_low_map & (ppf_delta < 0)).float() * 1.0

        # 고혈압인데 Propofol 증가 → bonus
        mask_high_map = map_val > self.target_map_max
        reward = reward + (mask_high_map & (ppf_delta > 0)).float() * 0.5
        reward = reward - (mask_high_map & (ppf_delta < 0)).float() * 0.5

        # HR low case
        mask_low_hr = hr_val < self.target_hr_min
        reward = reward - (mask_low_hr & (rftn_delta > 0)).float() * 1.0

        # HR high case
        mask_high_hr = hr_val > self.target_hr_max
        reward = reward + (mask_high_hr & (rftn_delta > 0)).float() * 0.5

        # -----------------------------
        # 3. Drug efficiency penalty
        # -----------------------------
        drug_penalty = 0.001 * (ppf_delta.abs() + rftn_delta.abs())
        reward = reward - drug_penalty

        return reward.sum()
