import os
import time
import numpy as np
import pandas as pd
import optuna
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import QRDQN

# Importiere Environment und DummyFMU aus der main.py
# pyrefly: ignore [missing-import]
from main import HeatPumpEnv, MockFMU

# Simulierte Rechenzeit auf 0 für schnelles Hyperparameter-Tuning
DUMMY_FMU_DELAY_MS = 0.0

def objective(trial):
    # 1. Hyperparameter von Optuna vorschlagen lassen
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    gamma = trial.suggest_float("gamma", 0.8, 0.9999)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
    
    # 2. Daten laden
    try:
        df = pd.read_csv("RL_Trainingsdaten_Frankfurt_2010_bis_2020_ohne_April_bis_September_100s.csv")
    except FileNotFoundError:
        print("[Optuna] CSV nicht gefunden! Bitte lege die Datei in denselben Ordner wie dieses Skript.")
        raise
        
    data_dict = {
        "Tair_degC": df["Tair_degC"].values,
        "phiAir": df["phiAir"].values,
        "T_Ruecklauf": df["T_Ruecklauf"].values,
        "Theta": df["Theta"].values,
    }

    def make_env():
        def _init():
            return HeatPumpEnv(data_dict, "dein_modell_pfad.fmu", simulated_delay_ms=DUMMY_FMU_DELAY_MS)
        return _init

    # 3. Environments erstellen (Weniger CPUs für Optuna um Overhead zu reduzieren)
    num_cpu = 2 
    env = SubprocVecEnv([make_env() for _ in range(num_cpu)])
    env = VecFrameStack(env, n_stack=36)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 4. Evaluierungs-Environment (Wichtig: norm_reward=False, training=False)
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecFrameStack(eval_env, n_stack=36)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    # 5. Modell initialisieren
    model = QRDQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        verbose=0  # Leise, da Optuna viele Trials durchführt
    )

    # 6. Modell trainieren (Für schnelle Durchläufe z.B. 10.000 Timesteps)
    model.learn(total_timesteps=10000, progress_bar=True)

    # 7. Evaluieren
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
    
    # 8. Aufräumen um Memory Leaks zu vermeiden
    env.close()
    eval_env.close()

    # Optuna maximiert diesen Reward
    return mean_reward

if __name__ == "__main__":
    print("Starte Optuna Hyperparameter-Suche...")
    
    # Optuna Studie erstellen. Wir maximieren den Reward.
    study = optuna.create_study(direction="maximize")
    
    # Führe 20 verschiedene Durchläufe aus, um die besten Hyperparameter zu finden
    study.optimize(objective, n_trials=20)
    
    print("\n==============================")
    print("Optimierung abgeschlossen!")
    print("Bester gefundener Reward:", study.best_value)
    print("Beste Parameter:", study.best_params)
    print("==============================\n")
