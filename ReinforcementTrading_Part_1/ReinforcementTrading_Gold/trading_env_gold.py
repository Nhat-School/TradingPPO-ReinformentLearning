import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

class GoldScalpingEnv(gym.Env):
    """
    5-Minute Timeframe Gold Scalping Environment.
    Fixed Risk:Reward = 1:2 (SL = $30, TP = $60).
    Uses actual PNL for equity tracking, RL rewards for learning.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, window_size: int = 60,
                 sl_usd: float = 30.0, tp_usd: float = 60.0,
                 spread_usd: float = 1.0,
                 feature_columns: list = None,
                 feature_mean: np.ndarray = None, feature_std: np.ndarray = None,
                 hold_penalty: float = 0.05,
                 entry_cost: float = 0.3):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.sl_usd = sl_usd
        self.tp_usd = tp_usd
        self.spread_usd = spread_usd
        self.hold_penalty = hold_penalty
        self.entry_cost = entry_cost  # RL penalty for opening a trade (discourages spam)

        self.feature_columns = feature_columns if feature_columns else df.columns.tolist()
        self.feature_data = self.df[self.feature_columns].values.astype(np.float32)

        if feature_mean is not None and feature_std is not None:
            self.obs_mean = feature_mean.astype(np.float32)
            self.obs_std = feature_std.astype(np.float32)
        else:
            self.obs_mean = np.mean(self.feature_data, axis=0).astype(np.float32)
            self.obs_std = np.std(self.feature_data, axis=0).astype(np.float32)
        
        # Prevent division by zero
        self.obs_std[self.obs_std == 0] = 1e-8

        # Actions: 0 = LONG, 1 = SHORT
        self.action_space = spaces.Discrete(2)
        
        # Obs: [Window data (normalized), pos_dir, unrealized_normalized, bars_in_trade]
        obs_shape = (window_size * len(self.feature_columns) + 3,)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=obs_shape, dtype=np.float32)

        self.current_step = self.window_size
        self.end_step = len(self.df) - 1

        self.pos_dir = 0
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.bars_in_trade = 0
        
        # History
        self.history_rewards = []
        self.history_equity = []
        self.history_pnl = []  # Track actual PNL per step
        self.equity = 10000.0
        self.total_trades = 0
        self.winning_trades = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        
        self.pos_dir = 0
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.bars_in_trade = 0
        
        self.equity = 10000.0
        self.history_rewards = []
        self.history_equity = [self.equity]
        self.history_pnl = []
        self.total_trades = 0
        self.winning_trades = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        window = self.feature_data[self.current_step - self.window_size : self.current_step]
        norm_window = (window - self.obs_mean) / self.obs_std
        # Clip to prevent extreme values corrupting the neural network
        norm_window = np.clip(norm_window, -5.0, 5.0)
        flat_window = norm_window.flatten()
        
        # In Oracle Fast-Forward mode, the bot is always "Flat" when making a decision.
        # We dummy out the extra 3 dimensions to maintain compatibility with the CNN architecture shape.
        extra = np.zeros(3, dtype=np.float32)
        return np.concatenate([flat_window, extra])

    def step(self, action):
        reward = 0.0
        actual_pnl = 0.0
        done = False
            
        # Oracle Fast-Forward logic for LONG(0) and SHORT(1)
        current_price = self.df.loc[self.current_step, "Close"]
        
        if action == 0: # LONG
            sl_price = current_price - self.sl_usd
            tp_price = current_price + self.tp_usd
        else: # SHORT (action == 1)
            sl_price = current_price + self.sl_usd
            tp_price = current_price - self.tp_usd
            
        future_idx = self.current_step
        outcome = 0 # 0=Timeout (Ranging), 1=Win, -1=Loss
        
        # Fast-Forward: Look ahead up to 100 bars (25 hours for 15m)
        max_lookahead = min(self.current_step + 100, self.end_step)
        
        for i in range(self.current_step + 1, max_lookahead):
            high = self.df.loc[i, "High"]
            low = self.df.loc[i, "Low"]
            
            if action == 0: # LONG
                # Check SL first for conservative outcome if huge volatile candle
                if low <= sl_price:
                    outcome = -1
                    future_idx = i
                    break
                elif high >= tp_price:
                    outcome = 1
                    future_idx = i
                    break
            else: # SHORT
                if high >= sl_price:
                    outcome = -1
                    future_idx = i
                    break
                elif low <= tp_price:
                    outcome = 1
                    future_idx = i
                    break
                    
        if outcome == 0:
            # Time out - ranging market, no resolution. Slight penalty to discourage spam.
            reward = -0.1
            actual_pnl = 0.0
            self.current_step += 1 # just move one step to try again
        else:
            reward = float(outcome)
            actual_pnl = self.tp_usd if outcome == 1 else -self.sl_usd
            
            self.total_trades += 1
            if outcome == 1:
                self.winning_trades += 1
                
            # ORACLE TIME-JUMP: Move clock directly to resolution timestamp
            # This completely solves Credit Assignment!
            self.current_step = future_idx
            
        self.equity += actual_pnl
        self.history_rewards.append(reward)
        self.history_pnl.append(actual_pnl)
        self.history_equity.append(self.equity)
        
        if self.current_step >= self.end_step:
            done = True
            
        return self._get_obs(), reward, done, False, {
            "actual_pnl": actual_pnl,
            "equity": self.equity,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades
        }
