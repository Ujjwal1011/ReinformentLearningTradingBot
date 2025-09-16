# evaluation/environment_factory.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Assuming these can be imported
from config import regression_features # Your list of features for the agent
from environments.environment import TradingEnvironment
from environments.wrappers.tf.tfenv import TFTradingEnvironment # Your existing TF wrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment

# Import your metric classes (needed for TradingEnvironment)
from metrics.trading.pnl import CumulativeLogReturn
from metrics.trading.risk import InvestmentRisk
from metrics.trading.sharpe import SharpeRatio
from metrics.trading.sortino import SortinoRatio
from metrics.trading.drawdown import MaximumDrawdown

# The reward function CLASS will be passed in, so no specific reward function import needed here.

def create_fold_environments(
    fold_raw_data_df: pd.DataFrame,       # Raw DataFrame slice for this specific data segment (train or eval)
    reward_fn_class,                      # The class of the reward function (e.g., MarketLimitOrderRF)
    env_construction_params: dict,        # Params like timeframe_size, target_horizon_len, fees, episode_steps
    scaler_for_transform: MinMaxScaler = None, # Optional: if provided, use this scaler (for eval data)
                                               # If None, a new scaler will be fit (for train data)
    env_id_suffix: str = "env"             # Suffix for logging if TFTradingEnvironment used env_id
):
    """
    Creates a TFPyEnvironment for a given data slice for use in a walk-forward fold.

    Args:
        fold_raw_data_df: Pandas DataFrame containing the raw data for this segment.
                          Must include 'high', 'low', 'close' and all `regression_features`.
        reward_fn_class: The class of the reward function to instantiate.
        env_construction_params: Dictionary containing:
            'timeframe_size': int
            'target_horizon_len': int
            'fees': float
            'episode_steps': int (max steps for episodes in this environment)
        scaler_for_transform: Optional, an already fitted MinMaxScaler.
                              If None, a new scaler will be fitted to fold_raw_data_df.
                              Typically, for training data of a fold, this is None.
                              For OOS evaluation data of a fold, this is the scaler from training.
        env_id_suffix: Suffix to potentially identify the environment instance.

    Returns:
        A TFPyEnvironment instance, or None if data is insufficient.
        A fitted scaler if one was created, otherwise the scaler_for_transform passed in.
    """
    timeframe_s = env_construction_params['timeframe_size']
    target_h = env_construction_params['target_horizon_len']

    min_raw_data_for_reward_calc = timeframe_s + target_h
    if len(fold_raw_data_df) < min_raw_data_for_reward_calc:
        print(f"ENV_FACTORY WARN [{env_id_suffix}]: Raw data slice too short ({len(fold_raw_data_df)}) "
              f"for reward calculation (min raw needed: {min_raw_data_for_reward_calc}). Cannot create env.")
        return None, scaler_for_transform # Return None for env

    # Prepare features for the agent
    features_raw = fold_raw_data_df[regression_features].to_numpy(dtype=np.float32)
    
    current_scaler = scaler_for_transform
    if current_scaler is None: # Fit a new scaler (typically for training data part of a fold)
        current_scaler = MinMaxScaler(feature_range=(0, 1.0))
        features_scaled = current_scaler.fit_transform(features_raw)
    else: # Use the provided scaler (typically for OOS evaluation data)
        features_scaled = current_scaler.transform(features_raw)

    num_possible_sequences = len(features_scaled) - timeframe_s - target_h + 1
    if num_possible_sequences <= 0:
        print(f"ENV_FACTORY WARN [{env_id_suffix}]: Not enough scaled data ({len(features_scaled)}) "
              f"to form any sequences. Cannot create env.")
        return None, current_scaler
    
    x_data_for_env = np.float32([
        features_scaled[i : i + timeframe_s] for i in range(num_possible_sequences)
    ])
    if len(x_data_for_env) == 0: return None, current_scaler # Should be caught by num_possible_sequences

    # Instantiate Reward Function
    try:
        reward_fn_obj = reward_fn_class(
            timeframe_size=timeframe_s, target_horizon_len=target_h,
            highs=fold_raw_data_df['high'].to_numpy(dtype=np.float32),
            lows=fold_raw_data_df['low'].to_numpy(dtype=np.float32),
            closes=fold_raw_data_df['close'].to_numpy(dtype=np.float32),
            fees_percentage=env_construction_params['fees'],
            verbose=False # Or pass from env_construction_params
        )
        # Align reward function length with x_data length
        if hasattr(reward_fn_obj, 'reward_fn') and len(reward_fn_obj.reward_fn) != len(x_data_for_env):
            # print(f"ENV_FACTORY INFO [{env_id_suffix}]: Aligning reward_fn ({len(reward_fn_obj.reward_fn)}) and x_data ({len(x_data_for_env)})")
            min_len = min(len(x_data_for_env), len(reward_fn_obj.reward_fn))
            if min_len == 0:
                print(f"ENV_FACTORY WARN [{env_id_suffix}]: Zero length after aligning reward_fn and x_data. Cannot create env.")
                return None, current_scaler
            x_data_for_env = x_data_for_env[:min_len]
            reward_fn_obj.reward_fn = reward_fn_obj.reward_fn[:min_len]
        elif not hasattr(reward_fn_obj, 'reward_fn'): # Should not happen if RewardFunction initializes _rewards_fn
             print(f"ENV_FACTORY ERROR [{env_id_suffix}]: reward_fn_obj does not have 'reward_fn' attribute after init. Cannot create env.")
             return None, current_scaler


    except ValueError as e:
        print(f"ENV_FACTORY ERROR [{env_id_suffix}]: ValueError during reward_fn instantiation: {e}. Cannot create env.")
        return None, current_scaler
    except AttributeError as e: # Catch if _rewards_fn was not set
        print(f"ENV_FACTORY ERROR [{env_id_suffix}]: AttributeError in reward_fn (likely _rewards_fn not set): {e}. Cannot create env.")
        return None, current_scaler


    # Ensure episode_steps is valid for the amount of data
    max_possible_steps = len(x_data_for_env) -1 if len(x_data_for_env) > 1 else 1
    if max_possible_steps < 1: # If x_data_for_env has only 0 or 1 sequence
        print(f"ENV_FACTORY WARN [{env_id_suffix}]: Not enough sequences in x_data ({len(x_data_for_env)}) for meaningful episode. Cannot create env.")
        return None, current_scaler
        
    actual_episode_steps = min(env_construction_params.get('episode_steps', 100), max_possible_steps)
    if actual_episode_steps < 1: actual_episode_steps = 1


    py_env = TradingEnvironment(env_config={
        'states': x_data_for_env,
        'reward_fn': reward_fn_obj,
        'episode_steps': actual_episode_steps,
        'metrics': [CumulativeLogReturn(), InvestmentRisk(), SharpeRatio(), SortinoRatio(), MaximumDrawdown()]
    })
    
    # Your TFTradingEnvironment __init__ does not take env_id as per your constraint
    tf_env_wrapper = TFTradingEnvironment(env=py_env)
    
    print(f"ENV_FACTORY INFO [{env_id_suffix}]: Successfully created TFPyEnvironment with {len(x_data_for_env)} states, episode_steps={actual_episode_steps}.")
    return TFPyEnvironment(environment=tf_env_wrapper), current_scaler