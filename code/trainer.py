import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from state_action_reward import StateActionRewardDesigner

class BehaviorCloningTrainer:
    """
    Behavioral Cloning 학습 클래스
    
    Expert demonstrations로부터 supervised learning
    - 슬라이드의 behavior cloning 개념 구현
    """
    
    def __init__(self, policy, learning_rate, save_direc, device: str = 'cpu'):
        self.policy = policy.to(device)
        self.device = device
        self.save_direc = save_direc
        
        # Optimizer
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Loss history
        self.train_losses = []
        self.val_losses = []
        self.train_rewards = []
        self.val_rewards = []
        self.reward_func = StateActionRewardDesigner().compute_reward
        
    def train_epoch(self, dataloader) -> float:
        """한 epoch 학습"""
        self.policy.train()
        total_loss = 0.0
        all_rewards = 0.0
        
        for states, actions, orig_state in dataloader:
            states = states.to(self.device, torch.float32)
            actions = actions.to(self.device, torch.float32)
            orig_state = orig_state.to(self.device, torch.float32)
            
            # Forward pass
            # Standard MSE loss
            predicted_actions = self.policy(states)
            if isinstance(predicted_actions, tuple):
                predicted_actions = predicted_actions[0]
            all_rewards += self.reward_func(orig_state, predicted_actions).item()
            loss = nn.MSELoss()(predicted_actions, actions)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_reward = all_rewards / len(dataloader)
        return avg_loss, avg_reward
    
    def validate(self, dataloader):
        """검증"""
        self.policy.eval()
        total_loss = 0.0
        all_rewards = 0.0
        
        with torch.no_grad():
            for states, actions, orig_state in dataloader:
                states = states.to(self.device, torch.float32)
                actions = actions.to(self.device, torch.float32)
                orig_state = orig_state.to(self.device, torch.float32)
                
                predicted_actions = self.policy(states)
                if isinstance(predicted_actions, tuple):
                    predicted_actions = predicted_actions[0]
                all_rewards += self.reward_func(orig_state, predicted_actions).item()
                loss = nn.MSELoss()(predicted_actions, actions)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_reward = all_rewards / len(dataloader)
        return avg_loss, avg_reward
    
    def train(self, train_loader, val_loader, num_epochs = 100, policy_type = 'standard'):
             #early_stopping_patience: int = 20):
        """전체 학습 루프"""
        
        best_val_loss = float('inf')
        # patience_counter = 0
        
        print(f"Training Behavioral Cloning model...")
        print(f"Policy type: {policy_type}")
        print(f"Device: {self.device}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_reward = self.train_epoch(train_loader)

            self.train_losses.append(train_loss)
            self.train_rewards.append(train_reward)
            
            # Validate
            val_loss, val_reward = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_rewards.append(val_reward)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print progress
            if True or (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train Return: {train_reward:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Return: {val_reward:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # patience_counter = 0
                # Save best model
                torch.save(self.policy.state_dict(), os.path.join(self.save_direc, 'best_policy.pth'))
            # else:
            #     patience_counter += 1
                
            # if patience_counter >= early_stopping_patience:
            #     print(f"\nEarly stopping at epoch {epoch+1}")
            #     print(f"Best validation loss: {best_val_loss:.4f}")
            #     break
        
        # Load best model
        self.policy.load_state_dict(torch.load(os.path.join(self.save_direc, 'best_policy.pth')))
        print("\nTraining completed!")
        
    def plot_learning_curves(self):
        save_direc = self.save_direc
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', alpha=0.7)
        plt.plot(self.val_losses, label='Valid Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Behavioral Cloning Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_direc, 'learning_curves.png'), dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_rewards, label='Train Reward', alpha=0.7)
        plt.plot(self.val_rewards, label='Valid Reward', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.title('Behavioral Cloning Learning Reward Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_direc, 'reward_curves.png'), dpi=300)
        plt.close()