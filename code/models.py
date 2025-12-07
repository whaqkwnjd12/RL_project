import numpy as np
import torch
import torch.nn as nn


class LSTMPolicy(nn.Module):
    """
    LSTM 기반 정책 네트워크
    
    시계열 의존성을 고려하여 과거 이력을 학습
    - 슬라이드에서 언급된 "History-aware models"
    - 같은 observation이라도 과거 맥락에 따라 다른 action
    """
    
    def __init__(self, state_dim, action_dim,  hidden_dim = 128, num_layers = 2):
        super(LSTMPolicy, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, x,  hidden):
        """
        Args:
            x: [batch, seq_len, state_dim] 또는 [batch, state_dim]
            hidden: LSTM hidden state
            
        Returns:
            action: [batch, action_dim]
            hidden: Updated hidden state
        """
        
        # Single timestep이면 sequence dimension 추가
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch, 1, state_dim]
        
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 마지막 timestep의 output 사용
        last_output = lstm_out[:, -1, :]  # [batch, hidden_dim]
        
        # Action 예측
        action = self.fc_layers(last_output)
        
        return action, hidden


class MixtureOfGaussiansPolicy(nn.Module):
    """
    Mixture of Gaussians (MoG) Policy
    
    다중 모드 행동 모델링:
    - 같은 상황에서 여러 가지 valid한 action 가능
    - 예: 혈압이 낮을 때 -> Propofol 줄이기 OR Remifentanil 줄이기
    - 슬라이드의 "Expressive action models" 구현
    """
    
    def __init__(self, state_dim, action_dim, num_gaussians = 5, hidden_dim = 256):
        super(MixtureOfGaussiansPolicy, self).__init__()
        
        self.num_gaussians = num_gaussians
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Mixture weights (π_i)
        self.mixture_weights = nn.Linear(hidden_dim, num_gaussians)
        
        # Means for each Gaussian (μ_i)
        self.means = nn.Linear(hidden_dim, num_gaussians * action_dim)
        
        # Log standard deviations (log σ_i)
        self.log_stds = nn.Linear(hidden_dim, num_gaussians * action_dim)
        
    def forward(self, state):
        """
        Returns:
            weights: [batch, num_gaussians]
            means: [batch, num_gaussians, action_dim]
            stds: [batch, num_gaussians, action_dim]
        """
        batch_size = state.shape[0]
        
        # Extract features
        features = self.feature_extractor(state)
        
        # Mixture weights (softmax for probability)
        weights = torch.softmax(self.mixture_weights(features), dim=-1)
        
        # Means
        means = self.means(features).view(batch_size, self.num_gaussians, self.action_dim)
        
        # Standard deviations (exp to ensure positive)
        log_stds = self.log_stds(features).view(batch_size, self.num_gaussians, self.action_dim)
        stds = torch.exp(log_stds.clamp(-10, 2))  # Stability
        
        return weights, means, stds
    
    def sample_action(self, state):
        """샘플링을 통한 action 생성"""
        weights, means, stds = self.forward(state)
        
        batch_size = state.shape[0]
        
        # Categorical distribution으로 어느 Gaussian을 쓸지 선택
        mixture_indices = torch.multinomial(weights, 1).squeeze(-1)
        
        # 선택된 Gaussian에서 샘플링
        selected_means = means[torch.arange(batch_size), mixture_indices]
        selected_stds = stds[torch.arange(batch_size), mixture_indices]
        
        # Gaussian sampling
        action = selected_means + selected_stds * torch.randn_like(selected_means)
        
        return action
    
    def get_log_prob(self, state, action):
        """Action의 log probability 계산 (학습용)"""
        weights, means, stds = self.forward(state)
        
        action = action.unsqueeze(1)  # [batch, 1, action_dim]
        
        # 각 Gaussian의 log probability 계산
        log_probs = []
        for i in range(self.num_gaussians):
            mean = means[:, i:i+1, :]
            std = stds[:, i:i+1, :]
            
            # Gaussian log probability
            log_prob = -0.5 * (
                ((action - mean) / std) ** 2 + 
                2 * torch.log(std) + 
                np.log(2 * np.pi)
            ).sum(dim=-1)
            
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs, dim=-1)  # [batch, num_gaussians]
        
        # Mixture의 log probability
        log_mixture_probs = torch.log(weights + 1e-8) + log_probs
        
        # Log-sum-exp trick for numerical stability
        total_log_prob = torch.logsumexp(log_mixture_probs, dim=-1)
        
        return total_log_prob


class TransformerPolicy(nn.Module):
    """
    Transformer 기반 정책 네트워크
    
    Self-attention으로 긴 시계열 의존성 학습:
    - LSTM보다 장기 의존성 포착에 유리
    - 병렬 처리 가능
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim = 128, num_heads = 4, num_layers = 3, seq_len = 50):
        super(TransformerPolicy, self).__init__()
        
        self.state_dim = state_dim
        self.seq_len = seq_len
        
        # Input embedding
        self.input_embedding = nn.Linear(state_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, seq_len, hidden_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, state_dim]
            
        Returns:
            action: [batch, action_dim]
        """
        # Embedding
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.shape[1], :]
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use last timestep for action prediction
        last_output = x[:, -1, :]
        
        # Predict action
        action = self.output_head(last_output)
        
        return action
    
