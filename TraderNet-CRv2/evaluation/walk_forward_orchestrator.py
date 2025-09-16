# evaluation/walk_forward_orchestrator.py

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import datetime
import shutil
# No need for MinMaxScaler here anymore, it's handled by the factory
# from sklearn.preprocessing import MinMaxScaler

# Import from your new factory
from .environment_factory import create_fold_environments # Assuming it's in the same 'evaluation' package
# Or if it's in a different path: from path.to.environment_factory import create_fold_environments

# Keep other necessary imports
from config import regression_features # Still needed here for _prepare_data_for_fold (data slicing)
# Agent classes will be passed in
# Reward function CLASS will be passed in
# Metric classes are used by the factory when creating TradingEnvironment

BASE_WALK_FORWARD_RESULTS_DIR = "walk_forward_orchestration_results"
os.makedirs(BASE_WALK_FORWARD_RESULTS_DIR, exist_ok=True)

def log_wfo_event(log_file_path, message):
    # ... (same as before)
    timestamp_str = datetime.datetime.now().isoformat()
    full_message = f"{timestamp_str}: {message}"
    print(full_message)
    try:
        with open(log_file_path, "a") as f:
            f.write(full_message + "\n")
    except Exception as e:
        print(f"ERROR writing to WFO log ({log_file_path}): {e}")


class WalkForwardOrchestrator:
    def __init__(self,
                 full_dataset_df: pd.DataFrame,
                 train_window_points: int,
                 eval_window_points: int,
                 step_points: int,
                 env_construction_params: dict,
                 agent_constructor_params: dict,
                 reward_fn_class,
                 agent_train_call_params: dict,
                 eval_episodes_oos: int = 1,
                 experiment_label: str = "WFO_Run",
                 results_subdir: str = "default_wfo_type",
                 retrain_from_scratch_each_fold: bool = True):

        self.full_dataset_df = full_dataset_df.copy()
        self.train_window_points = train_window_points
        self.eval_window_points = eval_window_points
        self.step_points = step_points
        self.env_construction_params = env_construction_params
        self.agent_constructor_params = agent_constructor_params
        self.reward_fn_class = reward_fn_class
        self.agent_train_call_params = agent_train_call_params
        self.eval_episodes_oos = eval_episodes_oos
        self.experiment_label = experiment_label
        self.results_subdir = results_subdir
        self.retrain_from_scratch_each_fold = retrain_from_scratch_each_fold

        timestamp_folder = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_results_dir = os.path.join(BASE_WALK_FORWARD_RESULTS_DIR, self.results_subdir, f"{self.experiment_label}_{timestamp_folder}")
        os.makedirs(self.run_results_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.run_results_dir, "wfo_log.txt")
        self.all_folds_oos_results_path = os.path.join(self.run_results_dir, "wfo_all_folds_out_of_sample_results.csv")
        self.overall_oos_metrics_history = []
        log_wfo_event(self.log_file_path, f"Initialized WalkForwardOrchestrator for: {self.experiment_label}...")
        # ... (rest of init logging)


    def _prepare_data_slices_for_fold(self, fold_num: int, train_start_idx: int):
        """ Just prepares raw data slices and dates for a fold. """
        train_end_idx = train_start_idx + self.train_window_points
        oos_eval_start_idx = train_end_idx
        oos_eval_end_idx = oos_eval_start_idx + self.eval_window_points

        if oos_eval_end_idx > len(self.full_dataset_df):
            log_wfo_event(self.log_file_path, f"Fold {fold_num}: Not enough data for full OOS evaluation window. Ending.")
            return None, None, None, None

        train_df_fold_raw = self.full_dataset_df.iloc[train_start_idx:train_end_idx].copy()
        oos_eval_df_fold_raw = self.full_dataset_df.iloc[oos_eval_start_idx:oos_eval_end_idx].copy()
        
        train_dates = train_df_fold_raw['date'] if 'date' in train_df_fold_raw.columns else pd.Series(dtype='datetime64[ns]')
        oos_eval_dates = oos_eval_df_fold_raw['date'] if 'date' in oos_eval_df_fold_raw.columns else pd.Series(dtype='datetime64[ns]')

        return train_df_fold_raw, oos_eval_df_fold_raw, train_dates, oos_eval_dates

    # _build_agent_for_fold remains the same as your last correct version

    def _build_agent_for_fold(self, fold_num: int, fold_checkpoint_path: str):
        agent_class = self.agent_constructor_params['agent_class']
        constructor_params = {k: v for k, v in self.agent_constructor_params.items() if k != 'agent_class'}
        constructor_params['checkpoint_filepath'] = fold_checkpoint_path
        log_wfo_event(self.log_file_path, f"Fold {fold_num}: Prepared constructor params for {agent_class.__name__} (specs to be added). Checkpoint: {fold_checkpoint_path}")
        return agent_class, constructor_params


    def _run_one_fold(self, fold_num: int, train_start_idx: int, agent_instance_to_continue_from=None):
        fold_label = f"Fold_{fold_num}"
        fold_checkpoint_dir = os.path.join(self.run_results_dir, f"{fold_label}_checkpoints")
        if self.retrain_from_scratch_each_fold and os.path.exists(fold_checkpoint_dir):
            log_wfo_event(self.log_file_path, f"{fold_label}: Retraining, removing old checkpoints: {fold_checkpoint_dir}")
            shutil.rmtree(fold_checkpoint_dir)
        os.makedirs(fold_checkpoint_dir, exist_ok=True)

        data_slices = self._prepare_data_slices_for_fold(fold_num, train_start_idx)
        if data_slices is None or any(item is None for item in data_slices):
            return None, None
        train_df_fold, oos_eval_df_fold, train_dates, oos_eval_dates = data_slices

        # --- Create Training Environment ---
        # For training, a new scaler is fit internally by create_fold_environments
        tf_train_env, scaler_for_fold = create_fold_environments(
            fold_raw_data_df=train_df_fold,
            reward_fn_class=self.reward_fn_class,
            env_construction_params=self.env_construction_params, # Pass full dict
            scaler_for_transform=None, # Fit new scaler
            env_id_suffix=f"F{fold_num}_Train"
        )
        if not tf_train_env: return None, None

        # --- Create Internal Evaluation Environment (for agent.train's periodic eval) ---
        # Use a subset of the training data, scaled with the *same* scaler_for_fold
        internal_eval_split_idx = int(len(train_df_fold) * 0.8) # e.g., last 20%
        if internal_eval_split_idx < self.env_construction_params.get('timeframe_size',12) + self.env_construction_params.get('target_horizon_len',1): # Ensure enough for at least one sequence
             internal_eval_df_for_train_phase = train_df_fold # Use all if split is too small
             log_wfo_event(self.log_file_path, f"{fold_label}: Internal eval split too small, using full training data for internal eval.")
        else:
            internal_eval_df_for_train_phase = train_df_fold.iloc[internal_eval_split_idx:]

        tf_eval_env_for_training, _ = create_fold_environments(
            fold_raw_data_df=internal_eval_df_for_train_phase,
            reward_fn_class=self.reward_fn_class,
            env_construction_params=self.env_construction_params,
            scaler_for_transform=scaler_for_fold, # Use scaler from this fold's training
            env_id_suffix=f"F{fold_num}_TrainInternalEval"
        )
        if not tf_eval_env_for_training:
            log_wfo_event(self.log_file_path, f"{fold_label}: Could not create internal eval env for training. Using training env for both.")
            tf_eval_env_for_training = tf_train_env


        # --- Create/Load Agent ---
        agent_class, agent_constructor_kwargs_base = self._build_agent_for_fold(fold_num, fold_checkpoint_dir)
        agent_constructor_kwargs_base.update({
            'input_tensor_spec': tf_train_env.observation_spec(),
            'action_spec': tf_train_env.action_spec(),
            'time_step_spec': tf_train_env.time_step_spec(),
            'env_batch_size': tf_train_env.batch_size
        })
        
        agent_to_train = None
        if agent_instance_to_continue_from and not self.retrain_from_scratch_each_fold:
            log_wfo_event(self.log_file_path, f"{fold_label}: Continuing training for agent.")
            agent_to_train = agent_instance_to_continue_from
            agent_to_train._checkpointer = Checkpointer(ckpt_dir=fold_checkpoint_dir, max_to_keep=1, agent=agent_to_train, policy=agent_to_train.policy)
            agent_to_train.load()
        else:
            log_wfo_event(self.log_file_path, f"{fold_label}: Creating new agent.")
            try:
                agent_to_train = agent_class(**agent_constructor_kwargs_base)
                agent_to_train.initialize()
            except Exception as e:
                log_wfo_event(self.log_file_path, f"ERROR Fold {fold_num}: Agent instantiation/init failed: {e}\n{traceback.format_exc()}")
                return None, None

        # --- Train Agent for this Fold ---
        log_wfo_event(self.log_file_path, f"{fold_label}: Starting agent training ({self.agent_train_call_params['train_iterations']} iterations)...")
        try:
            _, fold_training_detailed_metrics = agent_to_train.train(
                train_env=tf_train_env,
                eval_env=tf_eval_env_for_training,
                **self.agent_train_call_params # Unpack all training call parameters
            )
            if fold_training_detailed_metrics: # Save the log from agent.train()
                pd.DataFrame(fold_training_detailed_metrics).to_csv(os.path.join(self.run_results_dir, f"fold_{fold_num}_training_phase_eval_log.csv"), index=False)
            log_wfo_event(self.log_file_path, f"{fold_label}: Training complete.")
        except Exception as e:
            log_wfo_event(self.log_file_path, f"ERROR Fold {fold_num}: Exception during agent.train(): {e}\n{traceback.format_exc()}")
            return agent_to_train if not self.retrain_from_scratch_each_fold else None, None


        # --- Out-of-Sample Evaluation ---
        tf_oos_eval_env, _ = create_fold_environments(
            fold_raw_data_df=oos_eval_df_fold,
            reward_fn_class=self.reward_fn_class,
            env_construction_params=self.env_construction_params,
            scaler_for_transform=scaler_for_fold, # CRITICAL: use scaler from this fold's training
            env_id_suffix=f"F{fold_num}_OOSEval"
        )
        if not tf_oos_eval_env:
            log_wfo_event(self.log_file_path, f"Fold {fold_num}: Failed to create OOS eval environment.")
            return agent_to_train if not self.retrain_from_scratch_each_fold else None, None

        log_wfo_event(self.log_file_path, f"{fold_label}: Starting OOS evaluation ({self.eval_episodes_oos} episodes)...")
        
        oos_py_env_wrapper = None
        if hasattr(tf_oos_eval_env, 'pyenv'): oos_py_env_wrapper = tf_oos_eval_env.pyenv
        # ... (add other checks for _env._pyenv if needed, as in TFAgentBase)
        
        if hasattr(oos_py_env_wrapper, 'reset_underlying_env_metrics'):
             if isinstance(oos_py_env_wrapper, list):
                 if oos_py_env_wrapper: oos_py_env_wrapper[0].reset_underlying_env_metrics()
             else:
                 oos_py_env_wrapper.reset_underlying_env_metrics()
        
        oos_avg_return = agent_to_train.eval(eval_env=tf_oos_eval_env, num_episodes=self.eval_episodes_oos)
        
        oos_custom_metrics = {}
        if hasattr(oos_py_env_wrapper, 'get_underlying_env_metrics_results'):
            if isinstance(oos_py_env_wrapper, list):
                 if oos_py_env_wrapper: oos_custom_metrics = oos_py_env_wrapper[0].get_underlying_env_metrics_results()
            else:
                oos_custom_metrics = oos_py_env_wrapper.get_underlying_env_metrics_results()

        log_wfo_event(self.log_file_path, f"{fold_label} OOS Eval - TF Avg Return: {oos_avg_return:.4f}")
        if oos_custom_metrics: log_wfo_event(self.log_file_path, f"  {fold_label} OOS Custom Metrics: {oos_custom_metrics}")

        fold_results_dict = {
            'fold_num': fold_num,
            'train_start_date': train_dates.iloc[0] if not train_dates.empty else None,
            'train_end_date': train_dates.iloc[-1] if not train_dates.empty else None,
            'oos_eval_start_date': oos_eval_dates.iloc[0] if not oos_eval_dates.empty else None,
            'oos_eval_end_date': oos_eval_dates.iloc[-1] if not oos_eval_dates.empty else None,
            'oos_tf_avg_return': float(oos_avg_return),
            **oos_custom_metrics
        }
        self.overall_oos_metrics_history.append(fold_results_dict)
        
        return agent_to_train if not self.retrain_from_scratch_each_fold else None, fold_results_dict

    # run_evaluation method (mostly same, ensures agent is passed if needed)
    def run_evaluation(self):
        log_wfo_event(self.log_file_path, "Starting walk-forward evaluation run...")
        current_train_start_idx = 0
        fold_counter = 1
        agent_from_previous_fold = None # For continuous training

        while True:
            min_len_train_data = self.env_construction_params['timeframe_size'] + self.env_construction_params['target_horizon_len']
            min_len_eval_data = self.env_construction_params['timeframe_size'] + self.env_construction_params['target_horizon_len']

            # Check if enough data left for a full training window AND a full evaluation window
            if current_train_start_idx + self.train_window_points + self.eval_window_points > len(self.full_dataset_df):
                log_wfo_event(self.log_file_path, f"Not enough data remaining for a full train ({self.train_window_points}) + OOS eval ({self.eval_window_points}) window starting at {current_train_start_idx}. Total data: {len(self.full_dataset_df)}. Ending walk-forward.")
                break
            # Also ensure individual windows are long enough for sequence creation
            if self.train_window_points < min_len_train_data or self.eval_window_points < min_len_eval_data:
                log_wfo_event(self.log_file_path, f"Train window ({self.train_window_points}) or Eval window ({self.eval_window_points}) is too short for sequence/reward generation. Ending.")
                break
            
            if self.step_points <= 0:
                log_wfo_event(self.log_file_path, "ERROR: step_points is not positive. Aborting.")
                break

            log_wfo_event(self.log_file_path, f"\n===== Orchestrating Fold {fold_counter} =====")
            
            agent_to_use_this_fold = agent_from_previous_fold if not self.retrain_from_scratch_each_fold else None
            
            returned_agent, fold_oos_metrics = self._run_one_fold(fold_counter, current_train_start_idx, agent_to_use_this_fold)

            if not self.retrain_from_scratch_each_fold:
                agent_from_previous_fold = returned_agent # This could be None if fold failed
            else:
                agent_from_previous_fold = None 
            
            if fold_oos_metrics is None:
                log_wfo_event(self.log_file_path, f"Fold {fold_counter} was skipped or OOS eval failed. Advancing.")
            
            current_train_start_idx += self.step_points
            fold_counter += 1

            if self.overall_oos_metrics_history: # Save incrementally
                try:
                    pd.DataFrame(self.overall_oos_metrics_history).to_csv(self.all_folds_oos_results_path, index=False)
                except Exception as e:
                    log_wfo_event(self.log_file_path, f"ERROR saving intermediate OOS results: {e}")
        
        log_wfo_event(self.log_file_path, "Walk-forward evaluation finished.")
        if self.overall_oos_metrics_history:
            final_results_df = pd.DataFrame(self.overall_oos_metrics_history)
            final_results_df.to_csv(self.all_folds_oos_results_path, index=False)
            log_wfo_event(self.log_file_path, f"Overall OOS results saved to: {self.all_folds_oos_results_path}")
            print(f"\nOverall Out-of-Sample Results saved to {self.all_folds_oos_results_path}")
        else:
            log_wfo_event(self.log_file_path, "No folds were successfully evaluated and recorded.")
            print("\nNo folds were successfully evaluated and recorded.")
        
        return self.overall_oos_metrics_history