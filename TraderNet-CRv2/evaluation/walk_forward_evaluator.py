# walk_forward_evaluator.py

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import datetime
import json
from sklearn.preprocessing import MinMaxScaler

# Assuming your project structure allows these imports
from config import regression_features
from environments.environment import TradingEnvironment
from environments.wrappers.tf.tfenv import TFTradingEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from metrics.trading.pnl import CumulativeLogReturn
from metrics.trading.risk import InvestmentRisk
from metrics.trading.sharpe import SharpeRatio
from metrics.trading.sortino import SortinoRatio
from metrics.trading.drawdown import MaximumDrawdown
# from agents.tfagents.dqn import DQNAgent # Example - uncomment if used
# from agents.tfagents.ppo import PPOAgent # Example - uncomment if used

BASE_WALK_FORWARD_RESULTS_DIR = "walk_forward_results"
os.makedirs(BASE_WALK_FORWARD_RESULTS_DIR, exist_ok=True)

def log_wf_event(log_file_path, message):
    timestamp_str = datetime.datetime.now().isoformat()
    try:
        with open(log_file_path, "a") as f:
            f.write(f"{timestamp_str}: {message}\n")
    except Exception as e:
        print(f"ERROR writing to walk-forward log ({log_file_path}): {e}")

class WalkForwardEvaluator:
    def __init__(self,
                 full_dataset_df: pd.DataFrame,
                 train_window_points: int,
                 eval_window_points: int,
                 step_points: int,
                 env_params: dict,
                 agent_constructor_dict: dict,
                 reward_fn_class,
                 training_iterations_per_fold: int,
                 eval_episodes_oos: int = 1,
                 experiment_label: str = "WalkForwardRun",
                 results_subdir: str = "default_experiment_type"):

        self.full_dataset_df = full_dataset_df.copy()
        self.train_window_points = train_window_points
        self.eval_window_points = eval_window_points
        self.step_points = step_points
        self.env_params = env_params
        self.agent_constructor_dict = agent_constructor_dict
        self.reward_fn_class = reward_fn_class
        self.training_iterations_per_fold = training_iterations_per_fold
        self.eval_episodes_oos = eval_episodes_oos
        self.experiment_label = experiment_label
        self.results_subdir = results_subdir

        timestamp_folder = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_results_dir = os.path.join(BASE_WALK_FORWARD_RESULTS_DIR, self.results_subdir, f"{self.experiment_label}_{timestamp_folder}")
        os.makedirs(self.run_results_dir, exist_ok=True)

        self.log_file_path = os.path.join(self.run_results_dir, "walk_forward_log.txt")
        self.all_folds_oos_results_path = os.path.join(self.run_results_dir, "all_folds_out_of_sample_results.csv")
        
        self.overall_oos_metrics_history = []

        log_wf_event(self.log_file_path, f"Initialized WalkForwardEvaluator for: {self.experiment_label} ({self.results_subdir})")
        log_wf_event(self.log_file_path, f"Results will be saved in: {self.run_results_dir}")
        log_wf_event(self.log_file_path, f"  Train window: {train_window_points} data points")
        log_wf_event(self.log_file_path, f"  Eval (OOS) window: {eval_window_points} data points")
        log_wf_event(self.log_file_path, f"  Step size: {step_points} data points")
        log_wf_event(self.log_file_path, f"  Training iterations per fold: {self.training_iterations_per_fold}")


    def _prepare_fold_data_and_env(self, fold_num: int, train_start_idx: int):
        train_end_idx = train_start_idx + self.train_window_points
        oos_eval_start_idx = train_end_idx
        oos_eval_end_idx = oos_eval_start_idx + self.eval_window_points

        log_wf_event(self.log_file_path, f"Fold {fold_num}: Initial raw data slice indices: Train {train_start_idx}-{train_end_idx-1}, OOS Eval {oos_eval_start_idx}-{oos_eval_end_idx-1}")

        if oos_eval_end_idx > len(self.full_dataset_df):
            log_wf_event(self.log_file_path, f"Fold {fold_num}: Not enough data for full OOS evaluation window (needs up to {oos_eval_end_idx-1}, data len {len(self.full_dataset_df)}). Ending.")
            return None, None, None, None, None

        train_df_fold_raw = self.full_dataset_df.iloc[train_start_idx:train_end_idx].copy()
        oos_eval_df_fold_raw = self.full_dataset_df.iloc[oos_eval_start_idx:oos_eval_end_idx].copy()

        timeframe_s = self.env_params['timeframe_size']
        target_h = self.env_params['target_horizon_len']
        min_raw_data_for_reward_calc = timeframe_s + target_h
        
        if len(train_df_fold_raw) < min_raw_data_for_reward_calc:
            log_wf_event(self.log_file_path, f"Fold {fold_num}: Training data slice too short ({len(train_df_fold_raw)}) for reward calculation (min raw needed: {min_raw_data_for_reward_calc}). Skipping fold.")
            return None, None, None, None, None
        if len(oos_eval_df_fold_raw) < min_raw_data_for_reward_calc:
            log_wf_event(self.log_file_path, f"Fold {fold_num}: OOS Eval data slice too short ({len(oos_eval_df_fold_raw)}) for reward calculation (min raw needed: {min_raw_data_for_reward_calc}). Skipping fold.")
            return None, None, None, None, None

        # --- Training Data ---
        train_features_fold_raw = train_df_fold_raw[regression_features].to_numpy(dtype=np.float32)
        scaler = MinMaxScaler(feature_range=(0, 1.0))
        train_features_fold_scaled = scaler.fit_transform(train_features_fold_raw)
        
        num_possible_train_sequences = len(train_features_fold_scaled) - timeframe_s - target_h + 1
        if num_possible_train_sequences <= 0:
            log_wf_event(self.log_file_path, f"Fold {fold_num}: Not enough scaled training data ({len(train_features_fold_scaled)}) to form any sequences with timeframe {timeframe_s} and target horizon {target_h}. Skipping.")
            return None, None, None, None, None
        
        x_train_fold = np.float32([
            train_features_fold_scaled[i : i + timeframe_s]
            for i in range(num_possible_train_sequences)
        ])
        log_wf_event(self.log_file_path, f"Fold {fold_num}: Generated {len(x_train_fold)} training sequences (x_train_fold).")
        
        if len(x_train_fold) == 0: # Redundant due to num_possible_train_sequences check, but safe
            log_wf_event(self.log_file_path, f"Fold {fold_num}: x_train_fold is empty. Skipping.")
            return None, None, None, None, None

        train_high_raw = train_df_fold_raw['high'].to_numpy(dtype=np.float32)
        train_low_raw = train_df_fold_raw['low'].to_numpy(dtype=np.float32)
        train_close_raw = train_df_fold_raw['close'].to_numpy(dtype=np.float32)

        try:
            train_reward_fn_fold = self.reward_fn_class(
                timeframe_size=timeframe_s, target_horizon_len=target_h,
                highs=train_high_raw, lows=train_low_raw,
                closes=train_close_raw, fees_percentage=self.env_params['fees']
            )
        except ValueError as e:
            log_wf_event(self.log_file_path, f"ERROR Fold {fold_num}: ValueError during training reward_fn: {e}. Skipping.")
            return None, None, None, None, None

        if len(train_reward_fn_fold.reward_fn) != len(x_train_fold):
            log_wf_event(self.log_file_path, f"WARN Fold {fold_num}: Mismatch x_train_fold ({len(x_train_fold)}) vs train_reward_fn ({len(train_reward_fn_fold.reward_fn)}). Aligning.")
            min_len = min(len(x_train_fold), len(train_reward_fn_fold.reward_fn))
            if min_len == 0:
                log_wf_event(self.log_file_path, f"Fold {fold_num}: Zero length after alignment. Skipping.")
                return None, None, None, None, None
            x_train_fold = x_train_fold[:min_len]
            train_reward_fn_fold.reward_fn = train_reward_fn_fold.reward_fn[:min_len]

        # --- Internal Split for agent.train()'s eval_env ---
        num_internal_eval_samples = int(len(x_train_fold) * 0.2)
        min_samples_for_internal_eval = self.env_params.get('train_episode_steps_per_fold', 100)
        
        if len(x_train_fold) < 2 or num_internal_eval_samples < 1: # Need at least 1 sample for eval, and more for train
             log_wf_event(self.log_file_path, f"Fold {fold_num}: x_train_fold too small ({len(x_train_fold)}) for internal train/eval split. Using all for both.")
             x_train_for_agent_learning = x_train_fold
             reward_fn_for_agent_learning = train_reward_fn_fold
             x_eval_for_agent_training_phase = x_train_fold
             reward_fn_for_agent_training_eval = train_reward_fn_fold
        elif num_internal_eval_samples < min_samples_for_internal_eval and len(x_train_fold) > num_internal_eval_samples : # If eval split is too small but training part is ok
            log_wf_event(self.log_file_path, f"Fold {fold_num}: Internal eval split too small ({num_internal_eval_samples}). Using a larger portion or all for internal eval.")
            # Make eval at least min_samples_for_internal_eval or half if total is small
            num_internal_eval_samples = max(min_samples_for_internal_eval, len(x_train_fold) // 2, 1) 
            if len(x_train_fold) - num_internal_eval_samples < 1: # Ensure training part is not empty
                 x_train_for_agent_learning = x_train_fold
                 reward_fn_for_agent_learning = train_reward_fn_fold
                 x_eval_for_agent_training_phase = x_train_fold
                 reward_fn_for_agent_training_eval = train_reward_fn_fold
            else:
                x_eval_for_agent_training_phase = x_train_fold[-num_internal_eval_samples:]
                x_train_for_agent_learning = x_train_fold[:-num_internal_eval_samples]
                reward_fn_for_agent_learning_data = train_reward_fn_fold.reward_fn[:-num_internal_eval_samples]
                reward_fn_for_agent_training_eval_data = train_reward_fn_fold.reward_fn[-num_internal_eval_samples:]
                reward_fn_for_agent_learning = self.reward_fn_class(timeframe_size=0,target_horizon_len=0,highs=np.array([]),lows=np.array([]),closes=np.array([]),fees_percentage=0); reward_fn_for_agent_learning.reward_fn = reward_fn_for_agent_learning_data
                reward_fn_for_agent_training_eval = self.reward_fn_class(timeframe_size=0,target_horizon_len=0,highs=np.array([]),lows=np.array([]),closes=np.array([]),fees_percentage=0); reward_fn_for_agent_training_eval.reward_fn = reward_fn_for_agent_training_eval_data
        else: # Normal split
            x_eval_for_agent_training_phase = x_train_fold[-num_internal_eval_samples:]
            x_train_for_agent_learning = x_train_fold[:-num_internal_eval_samples]
            reward_fn_for_agent_learning_data = train_reward_fn_fold.reward_fn[:-num_internal_eval_samples]
            reward_fn_for_agent_training_eval_data = train_reward_fn_fold.reward_fn[-num_internal_eval_samples:]
            reward_fn_for_agent_learning = self.reward_fn_class(timeframe_size=0,target_horizon_len=0,highs=np.array([]),lows=np.array([]),closes=np.array([]),fees_percentage=0); reward_fn_for_agent_learning.reward_fn = reward_fn_for_agent_learning_data
            reward_fn_for_agent_training_eval = self.reward_fn_class(timeframe_size=0,target_horizon_len=0,highs=np.array([]),lows=np.array([]),closes=np.array([]),fees_percentage=0); reward_fn_for_agent_training_eval.reward_fn = reward_fn_for_agent_training_eval_data

        if len(x_train_for_agent_learning) == 0 or len(x_eval_for_agent_training_phase) == 0:
             log_wf_event(self.log_file_path, f"Fold {fold_num}: Training or internal eval data became empty after split. Skipping.")
             return None,None,None,None,None

        train_py_env_for_agent = TradingEnvironment(env_config={
            'states': x_train_for_agent_learning, 'reward_fn': reward_fn_for_agent_learning,
            'episode_steps': min(self.env_params.get('train_episode_steps_per_fold', 100), len(x_train_for_agent_learning)-1 if len(x_train_for_agent_learning)>1 else 1),
            'metrics': [CumulativeLogReturn(), InvestmentRisk(), SharpeRatio(), SortinoRatio(), MaximumDrawdown()]})
        tf_train_env_for_agent = TFPyEnvironment(TFTradingEnvironment(env=train_py_env_for_agent))

        eval_py_env_for_training_phase = TradingEnvironment(env_config={
            'states': x_eval_for_agent_training_phase, 'reward_fn': reward_fn_for_agent_training_eval,
            'episode_steps': len(x_eval_for_agent_training_phase) -1 if len(x_eval_for_agent_training_phase) >1 else 1,
            'metrics': [CumulativeLogReturn(), InvestmentRisk(), SharpeRatio(), SortinoRatio(), MaximumDrawdown()]})
        tf_eval_env_for_training_phase = TFPyEnvironment(TFTradingEnvironment(env=eval_py_env_for_training_phase))

        # --- OOS Evaluation Data & Environment ---
        oos_eval_features_raw = oos_eval_df_fold_raw[regression_features].to_numpy(dtype=np.float32)
        oos_eval_features_scaled = scaler.transform(oos_eval_features_raw)
        num_possible_oos_sequences = len(oos_eval_features_scaled) - timeframe_s - target_h + 1
        if num_possible_oos_sequences <= 0:
            log_wf_event(self.log_file_path, f"Fold {fold_num}: Not enough scaled OOS data ({len(oos_eval_features_scaled)}) for sequences. Skipping.")
            return None, None, None, None, None
        
        x_oos_eval_fold = np.float32([
            oos_eval_features_scaled[i : i + timeframe_s]
            for i in range(num_possible_oos_sequences)
        ])
        log_wf_event(self.log_file_path, f"Fold {fold_num}: Generated {len(x_oos_eval_fold)} OOS evaluation sequences.")
        if len(x_oos_eval_fold) == 0: # Redundant, but safe
            log_wf_event(self.log_file_path, f"Fold {fold_num}: x_oos_eval_fold is empty. Skipping.")
            return None, None, None, None, None

        oos_high_raw = oos_eval_df_fold_raw['high'].to_numpy(dtype=np.float32)
        oos_low_raw = oos_eval_df_fold_raw['low'].to_numpy(dtype=np.float32)
        oos_close_raw = oos_eval_df_fold_raw['close'].to_numpy(dtype=np.float32)
        try:
            oos_eval_reward_fn_fold = self.reward_fn_class(
                timeframe_size=timeframe_s, target_horizon_len=target_h,
                highs=oos_high_raw, lows=oos_low_raw,
                closes=oos_close_raw, fees_percentage=self.env_params['fees']
            )
        except ValueError as e:
            log_wf_event(self.log_file_path, f"ERROR Fold {fold_num}: ValueError during OOS reward_fn: {e}. Skipping.")
            return None, None, None, None, None

        if len(oos_eval_reward_fn_fold.reward_fn) != len(x_oos_eval_fold):
            log_wf_event(self.log_file_path, f"WARN Fold {fold_num}: Mismatch x_oos_eval ({len(x_oos_eval_fold)}) vs oos_reward_fn ({len(oos_eval_reward_fn_fold.reward_fn)}). Aligning.")
            min_len_oos = min(len(x_oos_eval_fold), len(oos_eval_reward_fn_fold.reward_fn))
            if min_len_oos == 0:
                log_wf_event(self.log_file_path, f"Fold {fold_num}: Zero length for OOS after alignment. Skipping.")
                return None, None, None, None, None
            x_oos_eval_fold = x_oos_eval_fold[:min_len_oos]
            oos_eval_reward_fn_fold.reward_fn = oos_eval_reward_fn_fold.reward_fn[:min_len_oos]

        if len(x_oos_eval_fold) == 0: # After alignment
            log_wf_event(self.log_file_path, f"Fold {fold_num}: x_oos_eval_fold became empty post-alignment. Skipping.")
            return None, None, None, None, None

        oos_py_env_fold = TradingEnvironment(env_config={
            'states': x_oos_eval_fold, 'reward_fn': oos_eval_reward_fn_fold,
            'episode_steps': len(x_oos_eval_fold) -1 if len(x_oos_eval_fold) > 1 else 1,
            'metrics': [CumulativeLogReturn(), InvestmentRisk(), SharpeRatio(), SortinoRatio(), MaximumDrawdown()]
        })
        tf_oos_eval_env_fold = TFPyEnvironment(TFTradingEnvironment(env=oos_py_env_fold))
        
        train_dates = train_df_fold_raw['date'] if 'date' in train_df_fold_raw.columns else pd.Series(dtype='datetime64[ns]')
        oos_eval_dates = oos_eval_df_fold_raw['date'] if 'date' in oos_eval_df_fold_raw.columns else pd.Series(dtype='datetime64[ns]')

        return tf_train_env_for_agent, tf_eval_env_for_training_phase, tf_oos_eval_env_fold, train_dates, oos_eval_dates


    def _build_agent_for_fold(self, fold_num: int, fold_checkpoint_path: str):
        agent_class = self.agent_constructor_dict['agent_class']
        constructor_params = {k: v for k, v in self.agent_constructor_dict.items() if k != 'agent_class'}
        constructor_params['checkpoint_filepath'] = fold_checkpoint_path
        log_wf_event(self.log_file_path, f"Fold {fold_num}: Prepared constructor params for {agent_class.__name__} (specs to be added). Checkpoint: {fold_checkpoint_path}")
        return agent_class, constructor_params


    def _run_one_fold(self, fold_num: int, train_start_idx: int):
        fold_checkpoint_dir = os.path.join(self.run_results_dir, f"fold_{fold_num}_checkpoints")
        os.makedirs(fold_checkpoint_dir, exist_ok=True)

        prepared_data = self._prepare_fold_data_and_env(fold_num, train_start_idx)
        if prepared_data is None or any(item is None for item in prepared_data): # More robust check
            log_wf_event(self.log_file_path, f"Fold {fold_num}: Data/Env preparation returned None. Skipping fold.")
            return None
        
        tf_train_env, tf_eval_env_for_training, tf_oos_eval_env, train_dates, oos_eval_dates = prepared_data

        agent_class, agent_constructor_kwargs_base = self._build_agent_for_fold(fold_num, fold_checkpoint_dir)

        agent_constructor_kwargs_base['input_tensor_spec'] = tf_train_env.observation_spec()
        agent_constructor_kwargs_base['action_spec'] = tf_train_env.action_spec()
        agent_constructor_kwargs_base['time_step_spec'] = tf_train_env.time_step_spec()
        agent_constructor_kwargs_base['env_batch_size'] = tf_train_env.batch_size

        try:
            agent = agent_class(**agent_constructor_kwargs_base)
            agent.initialize()
        except Exception as e:
            log_wf_event(self.log_file_path, f"ERROR Fold {fold_num}: Failed to instantiate or initialize agent: {e}")
            import traceback
            traceback.print_exc(file=open(self.log_file_path, "a")) # Log traceback to file
            return None


        log_wf_event(self.log_file_path, f"Fold {fold_num}: Starting agent training for {self.training_iterations_per_fold} iterations...")
        
        iterations_per_eval_fold = self.agent_constructor_dict.get('steps_per_eval_fold', max(1, self.training_iterations_per_fold // 20))
        iterations_per_log_fold = self.agent_constructor_dict.get('steps_per_log_fold', iterations_per_eval_fold)
        iterations_per_checkpoint_fold = self.agent_constructor_dict.get('steps_per_checkpoint_fold', iterations_per_eval_fold * 2)
        save_best_only_fold_training = self.agent_constructor_dict.get('save_best_only_fold', True)

        try:
            training_avg_returns, training_detailed_metrics_history = agent.train(
                train_env=tf_train_env, eval_env=tf_eval_env_for_training,
                train_iterations=self.training_iterations_per_fold,
                eval_episodes=self.env_params.get('eval_episodes_during_training', 1),
                iterations_per_eval=iterations_per_eval_fold,
                iterations_per_log=iterations_per_log_fold,
                iterations_per_checkpoint=iterations_per_checkpoint_fold,
                save_best_only=save_best_only_fold_training
            )
        except Exception as e:
            log_wf_event(self.log_file_path, f"ERROR Fold {fold_num}: Exception during agent.train(): {e}")
            import traceback
            traceback.print_exc(file=open(self.log_file_path, "a"))
            return None

        if training_detailed_metrics_history:
            fold_train_log_df = pd.DataFrame(training_detailed_metrics_history)
            fold_train_log_path = os.path.join(self.run_results_dir, f"fold_{fold_num}_training_eval_log.csv")
            fold_train_log_df.to_csv(fold_train_log_path, index=False)
            log_wf_event(self.log_file_path, f"Fold {fold_num}: Saved training phase evaluation log to {fold_train_log_path}")
        log_wf_event(self.log_file_path, f"Fold {fold_num}: Training complete.")

        # --- Out-of-Sample Evaluation ---
        log_wf_event(self.log_file_path, f"Fold {fold_num}: Starting OOS evaluation ({self.eval_episodes_oos} episodes)...")
        
        oos_py_env_wrapper = None # Logic to get oos_py_env_wrapper as before
        if hasattr(tf_oos_eval_env, 'pyenv'): oos_py_env_wrapper = tf_oos_eval_env.pyenv
        elif hasattr(tf_oos_eval_env, '_env') and hasattr(tf_oos_eval_env._env, '_pyenv'): oos_py_env_wrapper = tf_oos_eval_env._env._pyenv
        
        if hasattr(oos_py_env_wrapper, 'reset_underlying_env_metrics'):
             if isinstance(oos_py_env_wrapper, list):
                 if oos_py_env_wrapper: oos_py_env_wrapper[0].reset_underlying_env_metrics()
             else:
                 oos_py_env_wrapper.reset_underlying_env_metrics()

        oos_tf_metric = AverageReturnMetric(batch_size=tf_oos_eval_env.batch_size, buffer_size=self.eval_episodes_oos)
        try:
            oos_eval_driver = agent._get_episode_driver(
                env=tf_oos_eval_env, policy=agent.policy, observers=[oos_tf_metric], num_episodes=self.eval_episodes_oos
            )
            if hasattr(oos_eval_driver, 'run') and callable(oos_eval_driver.run) and not isinstance(oos_eval_driver.run, tf.types.experimental.Function):
                oos_eval_driver.run = function(oos_eval_driver.run)
            oos_eval_driver.run()
        except Exception as e:
            log_wf_event(self.log_file_path, f"ERROR Fold {fold_num}: Exception during OOS eval_driver.run(): {e}")
            import traceback
            traceback.print_exc(file=open(self.log_file_path, "a"))
            return None # Skip this fold's results if OOS eval fails
        
        oos_avg_return = oos_tf_metric.result().numpy()
        oos_custom_metrics = {}
        if hasattr(oos_py_env_wrapper, 'get_underlying_env_metrics_results'):
            if isinstance(oos_py_env_wrapper, list):
                 if oos_py_env_wrapper: oos_custom_metrics = oos_py_env_wrapper[0].get_underlying_env_metrics_results()
            else:
                oos_custom_metrics = oos_py_env_wrapper.get_underlying_env_metrics_results()

        log_wf_event(self.log_file_path, f"Fold {fold_num} OOS Eval - TF Avg Return: {oos_avg_return:.4f}")
        if oos_custom_metrics: log_wf_event(self.log_file_path, f"  Fold {fold_num} OOS Custom Metrics: {oos_custom_metrics}")

        fold_results = {
            'fold_num': fold_num,
            'train_start_date': train_dates.iloc[0] if not train_dates.empty else None,
            'train_end_date': train_dates.iloc[-1] if not train_dates.empty else None,
            'oos_eval_start_date': oos_eval_dates.iloc[0] if not oos_eval_dates.empty else None,
            'oos_eval_end_date': oos_eval_dates.iloc[-1] if not oos_eval_dates.empty else None,
            'oos_tf_avg_return': float(oos_avg_return),
            **oos_custom_metrics
        }
        self.overall_oos_metrics_history.append(fold_results)
        return fold_results

    def run_evaluation(self):
        log_wf_event(self.log_file_path, "Starting walk-forward evaluation...")
        start_idx = 0
        fold_num = 1
        while True:
            if start_idx + self.train_window_points + self.eval_window_points > len(self.full_dataset_df):
                log_wf_event(self.log_file_path, "Not enough data for a new training and OOS evaluation window. Ending.")
                break
            if self.step_points <= 0:
                log_wf_event(self.log_file_path, "ERROR: step_points is not positive. Aborting to prevent infinite loop.")
                break

            log_wf_event(self.log_file_path, f"\n===== Processing Fold {fold_num} =====")
            fold_result = self._run_one_fold(fold_num, start_idx)
            
            if fold_result is None:
                log_wf_event(self.log_file_path, f"Fold {fold_num} was skipped. Advancing to next potential fold.")
            
            start_idx += self.step_points
            fold_num += 1

            if self.overall_oos_metrics_history: # Save incrementally
                try:
                    pd.DataFrame(self.overall_oos_metrics_history).to_csv(self.all_folds_oos_results_path, index=False)
                except Exception as e:
                    log_wf_event(self.log_file_path, f"ERROR saving intermediate all_folds_results.csv: {e}")


        log_wf_event(self.log_file_path, "Walk-forward evaluation finished.")
        if self.overall_oos_metrics_history:
            final_results_df = pd.DataFrame(self.overall_oos_metrics_history)
            final_results_df.to_csv(self.all_folds_oos_results_path, index=False)
            log_wf_event(self.log_file_path, f"Overall out-of-sample results saved to: {self.all_folds_oos_results_path}")
            print(f"\nOverall Out-of-Sample Results saved to {self.all_folds_oos_results_path}")
            # print(final_results_df) # Optionally print the full df
        else:
            log_wf_event(self.log_file_path, "No folds were successfully evaluated and recorded.")
            print("\nNo folds were successfully evaluated and recorded.")
        
        return self.overall_oos_metrics_history