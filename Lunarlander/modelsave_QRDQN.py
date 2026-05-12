import gymnasium as gym
from sb3_contrib import QRDQN
import os

# Verzeichnisse anlegen
models_dir = "models/QRDQN"
logdir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Environment erstellen
env = gym.make('LunarLander-v3')

# Initialisierung des QR-DQN Modells mit optimierten Replay-Buffer-Werten
model = QRDQN(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=logdir,
    # --- REPLAY BUFFER EINSTELLUNGEN ---
    buffer_size=100000,          # Speichert die letzten 100.000 Erlebnisse.
    learning_starts=10000,       # Das Modell lernt erst, wenn der Buffer 10.000 Steps gesammelt hat.
    batch_size=128,              # Zieht 128 zufällige Erlebnisse pro Lernschritt aus dem Buffer.
    # --- WEITERE WICHTIGE PARAMETER FÜR LUNARLANDER ---
    learning_rate=0.00063,       # Bewährte Lernrate für LunarLander.
    exploration_fraction=0.12,   # Das Modell wählt für die ersten 12% des Trainings oft zufällige Aktionen.
    target_update_interval=250,  # Aktualisiert das Target-Network alle 250 Schritte.
    policy_kwargs=dict(n_quantiles=200) # Spezifisch für QR-DQN: Anzahl der Quantile.
)

TIMESTEPS = 10000
iters = 0

print("Starte QR-DQN Training...")

# Training in Schleife, genau wie in deinem alten Skript
for i in range(50): # 50 * 10k = 500.000 Schritte (LunarLander braucht meist etwas mehr als 300k)
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="QRDQN")
    model.save(f"{models_dir}/{TIMESTEPS * iters}")

env.close()
print("Training abgeschlossen!")