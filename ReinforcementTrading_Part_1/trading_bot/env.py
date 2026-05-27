from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MultiAssetTradingEnv(gym.Env):
    """Position-persistent trading environment shared by all Binance spot symbols."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        feature_columns: list[str],
        window_size: int,
        sl_options: tuple[int, ...],
        tp_options: tuple[int, ...],
        price_distance_mode: str = "bps",
        pip_value: float = 1.0,
        lot_size: float = 1.0,
        spread_pips: float = 2.0,
        commission_pips: float = 0.0,
        max_slippage_pips: float = 0.0,
        initial_equity_usd: float = 10_000.0,
        risk_fraction_per_trade: float = 0.01,
        max_notional_fraction: float = 1.0,
        min_equity_fraction: float = 0.2,
        reward_scale: float = 100.0,
        feature_mean=None,
        feature_std=None,
        reward_mode: str = "pnl_drawdown",
        random_start: bool = True,
        min_episode_steps: int = 1000,
        episode_max_steps: int | None = 2000,
        open_penalty_pips: float = 2.0,
        hold_reward_weight: float = 0.03,
        time_penalty_pips: float = 0.0,
        drawdown_penalty_weight: float = 25.0,
        sharpe_window: int = 64,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_columns = list(feature_columns)
        self.window_size = int(window_size)
        self.sl_options = tuple(sl_options)
        self.tp_options = tuple(tp_options)
        self.price_distance_mode = price_distance_mode
        self.pip_value = float(pip_value)
        self.lot_size = float(lot_size)
        self.spread_pips = float(spread_pips)
        self.commission_pips = float(commission_pips)
        self.max_slippage_pips = float(max_slippage_pips)
        self.initial_equity_usd = float(initial_equity_usd)
        self.risk_fraction_per_trade = float(risk_fraction_per_trade)
        self.max_notional_fraction = float(max_notional_fraction)
        self.min_equity_fraction = float(min_equity_fraction)
        self.reward_scale = float(reward_scale)
        self.feature_mean = None if feature_mean is None else np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = None if feature_std is None else np.asarray(feature_std, dtype=np.float32)
        self.reward_mode = reward_mode
        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps
        self.open_penalty_pips = float(open_penalty_pips)
        self.hold_reward_weight = float(hold_reward_weight)
        self.time_penalty_pips = float(time_penalty_pips)
        self.drawdown_penalty_weight = float(drawdown_penalty_weight)
        self.sharpe_window = int(sharpe_window)

        if len(self.df) <= self.window_size + 2:
            raise ValueError("Dataframe is too short for the requested window.")

        self.action_map = [("HOLD", None, None, None), ("CLOSE", None, None, None)]
        for direction in [0, 1]:
            for sl in self.sl_options:
                for tp in self.tp_options:
                    self.action_map.append(("OPEN", direction, float(sl), float(tp)))
        self.action_space = spaces.Discrete(len(self.action_map))

        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 3
        self.num_features = self.base_num_features + self.state_num_features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32,
        )
        self._reset_state()

    def _reset_state(self):
        self.current_step = self.window_size
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False
        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.position_size_units = 0.0
        self.time_in_trade = 0
        self.prev_unrealized_pnl_usd = 0.0
        self.equity_usd = self.initial_equity_usd
        self.equity_peak_usd = self.initial_equity_usd
        self.equity_curve = []
        self.reward_history = deque(maxlen=max(8, self.sharpe_window))
        self.last_trade_info = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        if self.random_start:
            max_start = len(self.df) - max(self.min_episode_steps, self.window_size) - 2
            self.current_step = self.window_size if max_start <= self.window_size else int(self.np_random.integers(self.window_size, max_start))
        return self._get_observation(), {}

    def _state_features(self):
        pos = float(self.position)
        time_scaled = float(self.time_in_trade) / 1000.0
        unreal = self._unrealized_return_pct() if self.position else 0.0
        return np.asarray([pos, time_scaled, unreal], dtype=np.float32)

    def _get_observation(self):
        start = max(0, self.current_step - self.window_size)
        base = self.df.iloc[start:self.current_step][self.feature_columns].values.astype(np.float32)
        if len(base) < self.window_size:
            pad = np.tile(base[0], (self.window_size - len(base), 1))
            base = np.vstack([pad, base])
        state = np.tile(self._state_features(), (self.window_size, 1))
        obs = np.hstack([base, state]).astype(np.float32)
        if self.feature_mean is not None and self.feature_std is not None:
            std = np.where(self.feature_std == 0, 1.0, self.feature_std)
            obs = ((obs - self.feature_mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)
        return obs

    def _price_distance(self, value: float, reference_price: float):
        if self.price_distance_mode == "bps":
            return max(reference_price * float(value) / 10_000.0, 1e-9)
        return max(float(value) * self.pip_value, 1e-9)

    def _slippage_distance(self, reference_price: float):
        if self.max_slippage_pips <= 0:
            return 0.0
        sampled = float(self.np_random.uniform(0.0, self.max_slippage_pips))
        return self._price_distance(sampled, reference_price)

    def _unrealized_pips(self):
        if self.position == 0 or self.entry_price is None:
            return 0.0
        close = float(self.df.loc[self.current_step, "Close"])
        return (close - self.entry_price) / self.pip_value if self.position == 1 else (self.entry_price - close) / self.pip_value

    def _unrealized_pnl_usd(self):
        if self.position == 0 or self.entry_price is None:
            return 0.0
        return self._unrealized_pips() * self.pip_value * self.position_size_units

    def _unrealized_return_pct(self):
        if self.position == 0 or self.entry_price is None:
            return 0.0
        close = float(self.df.loc[self.current_step, "Close"])
        raw_return = (close - self.entry_price) / max(self.entry_price, 1e-9)
        return float(raw_return * self.position * 100.0)

    def _open(self, direction, sl_pips, tp_pips):
        close = float(self.df.loc[self.current_step, "Close"])
        slip = self._slippage_distance(close)
        if direction == 1:
            entry = close + slip
            sl_distance = self._price_distance(sl_pips, entry)
            tp_distance = self._price_distance(tp_pips, entry)
            self.sl_price = entry - sl_distance
            self.tp_price = entry + tp_distance
            self.position = 1
        else:
            entry = close - slip
            sl_distance = self._price_distance(sl_pips, entry)
            tp_distance = self._price_distance(tp_pips, entry)
            self.sl_price = entry + sl_distance
            self.tp_price = entry - tp_distance
            self.position = -1
        risk_usd = max(self.equity_usd * self.risk_fraction_per_trade, 0.0)
        risk_units = risk_usd / max(sl_distance, 1e-9)
        notional_units = (self.equity_usd * self.max_notional_fraction) / max(entry, 1e-9)
        self.position_size_units = max(0.0, min(risk_units, notional_units)) * self.lot_size
        self.entry_price = entry
        self.time_in_trade = 0
        self.prev_unrealized_pnl_usd = 0.0
        open_cost_usd = self._price_distance(self.open_penalty_pips, entry) * self.position_size_units
        self.last_trade_info = {
            "event": "OPEN",
            "step": self.current_step,
            "position": self.position,
            "position_size_units": float(self.position_size_units),
            "risk_fraction_per_trade": float(self.risk_fraction_per_trade),
        }
        return open_cost_usd

    def _close(self, reason, exit_price):
        raw = (exit_price - self.entry_price) / self.pip_value if self.position == 1 else (self.entry_price - exit_price) / self.pip_value
        net_price_distance = raw * self.pip_value
        net_price_distance -= self._price_distance(self.spread_pips, self.entry_price)
        net_price_distance -= self._price_distance(self.commission_pips, self.entry_price)
        net = net_price_distance / self.pip_value
        pnl_usd = net_price_distance * self.position_size_units
        self.equity_usd = max(0.0, self.equity_usd + pnl_usd)
        trade = {
            "event": "CLOSE",
            "reason": reason,
            "step": self.current_step,
            "position": self.position,
            "position_size_units": float(self.position_size_units),
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "realized_pips": float(raw),
            "net_pips": float(net),
            "pnl_usd": float(pnl_usd),
            "equity_usd": float(self.equity_usd),
            "time_in_trade": int(self.time_in_trade),
        }
        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.position_size_units = 0.0
        self.time_in_trade = 0
        self.prev_unrealized_pnl_usd = 0.0
        self.last_trade_info = trade
        return pnl_usd

    def _check_intrabar_exit(self):
        if self.position == 0:
            return None
        if self.current_step >= len(self.df) - 2:
            return self._close("END_OF_DATA", float(self.df.loc[self.current_step, "Close"]))
        high = float(self.df.loc[self.current_step + 1, "High"])
        low = float(self.df.loc[self.current_step + 1, "Low"])
        if self.position == 1:
            sl_hit = low <= self.sl_price
            tp_hit = high >= self.tp_price
        else:
            sl_hit = high >= self.sl_price
            tp_hit = low <= self.tp_price
        if sl_hit and tp_hit:
            return self._close("SL_AND_TP_SAME_BAR_SL_FIRST", self.sl_price)
        if sl_hit:
            return self._close("SL_HIT", self.sl_price)
        if tp_hit:
            return self._close("TP_HIT", self.tp_price)
        return None

    def _shape_reward(self, realized_pnl_usd):
        reward = (realized_pnl_usd / max(self.initial_equity_usd, 1e-9)) * self.reward_scale
        if self.reward_mode == "pnl_drawdown":
            self.equity_peak_usd = max(self.equity_peak_usd, self.equity_usd)
            drawdown = max(0.0, (self.equity_peak_usd - self.equity_usd) / max(self.equity_peak_usd, 1e-9))
            reward -= self.drawdown_penalty_weight * drawdown
        elif self.reward_mode == "sharpe_proxy":
            recent_std = np.std(self.reward_history) if len(self.reward_history) > 3 else 1.0
            reward = reward / max(float(recent_std), 1.0)
        self.reward_history.append(float(reward))
        return reward

    def step(self, action):
        if self.terminated or self.truncated:
            return self._get_observation(), 0.0, True, False, {}

        self.steps_in_episode += 1
        act_type, direction, sl_pips, tp_pips = self.action_map[int(action)]
        reward = 0.0

        if act_type == "CLOSE" and self.position != 0:
            close = float(self.df.loc[self.current_step, "Close"])
            slip = self._slippage_distance(close)
            exit_price = close - slip if self.position == 1 else close + slip
            reward += self._shape_reward(self._close("MANUAL_CLOSE", exit_price))
        elif act_type == "OPEN" and self.position == 0:
            open_cost_usd = self._open(direction, sl_pips, tp_pips)
            reward -= (open_cost_usd / max(self.initial_equity_usd, 1e-9)) * self.reward_scale

        realized = self._check_intrabar_exit()
        if realized is not None:
            reward += self._shape_reward(realized)

        if self.position != 0:
            self.time_in_trade += 1
            unreal = self._unrealized_pnl_usd()
            reward += self.hold_reward_weight * (
                (unreal - self.prev_unrealized_pnl_usd) / max(self.initial_equity_usd, 1e-9)
            ) * self.reward_scale
            reward -= self.time_penalty_pips
            self.prev_unrealized_pnl_usd = unreal

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.terminated = True
        if self.equity_usd <= self.initial_equity_usd * self.min_equity_fraction:
            self.terminated = True
        if self.episode_max_steps is not None and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        self.equity_curve.append(float(self.equity_usd))
        info = {
            "equity_usd": float(self.equity_usd),
            "position": int(self.position),
            "reward": float(reward),
            "last_trade_info": self.last_trade_info,
        }
        return self._get_observation(), float(reward), self.terminated, self.truncated, info
