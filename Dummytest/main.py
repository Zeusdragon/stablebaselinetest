import os
import time
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


# ==========================================
# 1. DIE DUMMY FMU (DEIN PLATZHALTER)
# ==========================================
class MockFMU:
    def __init__(self, fmu_path, simulated_delay_ms: float = 1000.0):
        # fmu_path wird im Dummy ignoriert
        print(f"[MockFMU] Initialisiere Dummy-Modell anstelle von: {fmu_path}")
        self.t_luft = 273.15 + 2.0
        self.t_verdampfer = 273.15 + 0.0
        self.t_ruecklauf = 273.15 + 30.0
        self.cop = 3.5
        self.time = 0.0
        self.set_simulated_delay_ms(simulated_delay_ms)

    def set_simulated_delay_ms(self, simulated_delay_ms: float):
        # Simulierte FMU-Rechenzeit pro do_step-Aufruf.
        self.simulated_delay_s = max(float(simulated_delay_ms), 0.0) / 1000.0

    def reset(self):
        self.time = 0.0
        self.t_verdampfer = self.t_luft - 2.0
        self.cop = 3.5

    def set(self, variable_name, value):
        if variable_name == "TairInlet":
            self.t_luft = value
        elif variable_name == "TliqInlet":
            self.t_ruecklauf = value
        elif variable_name == "reverseCycle":  # Aktion des Agenten!
            if value == True:
                # Simuliere Abtauen: Verdampfer wird warm, COP regeneriert sich
                self.t_verdampfer = self.t_luft
                self.cop = 3.5

    def do_step(self, current_time, step_size):
        if self.simulated_delay_s > 0:
            time.sleep(self.simulated_delay_s)

        self.time += step_size
        # Simuliere langsames Eiswachstum: Verdampfer wird immer kälter als die Luft
        self.t_verdampfer -= 0.02 * (step_size / 60)

        # Simuliere Leistungseinbruch bei dickem Eis (T_verdampfer fällt stark ab)
        if (self.t_luft - self.t_verdampfer) > 5.0:
            self.cop = max(1.0, self.cop - 0.1)

    def get(self, variable_name):
        if variable_name == "COP":
            return self.cop
        if variable_name == "T_Verdampfer":
            return self.t_verdampfer
        return 0.0


# ==========================================
# 2. DAS GYM ENVIRONMENT
# ==========================================
class HeatPumpEnv(gym.Env):
    def __init__(self, data_dict, fmu_path, simulated_delay_ms: float = 1000.0):
        super().__init__()
        self.step_seconds = 100
        # Daten-Referenzen (Blitzschneller Zugriff auf Numpy Arrays)
        self.t_air = data_dict["Tair_degC"]
        self.phi_air = data_dict["phiAir"]
        self.t_ruecklauf = data_dict["T_Ruecklauf"]
        self.theta = data_dict["Theta"]

        self.total_rows = len(self.t_air)
        self.episode_length = int((24 * 60 * 60) / self.step_seconds)  # 24h in 100s Schritten

        # RL Spaces
        self.action_space = spaces.Discrete(2)  # 0 = Heizen, 1 = Abtauen
        # Beobachtungen: [T_Luft, T_Verdampfer, prev_action]
        # prev_action = Aktion des vorherigen Zeitschritts (0 oder 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        # Historische Aktion initialisieren
        self.prev_action = 0

        # LATER: Hier tauschst du MockFMU gegen fmpy aus!
        # self.fmu = setup_fmu(fmu_path)
        self.fmu = MockFMU(fmu_path, simulated_delay_ms=simulated_delay_ms)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 1. Gültigen Startpunkt in den dynamischen Wetterdaten suchen
        while True:
            start_idx = np.random.randint(0, self.total_rows - self.episode_length)
            episoden_theta = self.theta[start_idx : start_idx + self.episode_length]

            # Filter: Nur heizrelevante Wintertage zulassen
            if np.mean(episoden_theta) > 0.10:
                self.start_idx = start_idx
                break

        self.current_step = 0
        self.fmu.reset()

        # Startbedingungen setzen
        idx = self.start_idx
        self.fmu.set("TairInlet", self.t_air[idx] + 273.15)
        self.fmu.set("TliqInlet", self.t_ruecklauf[idx] + 273.15)
        # Startaktion (keine vorherige Aktion vorhanden)
        self.prev_action = 0

        return self._get_obs(), {}

    def step(self, action):
        idx = self.start_idx + self.current_step

        # 1. Wetterdaten updaten
        self.fmu.set("TairInlet", self.t_air[idx] + 273.15)
        self.fmu.set("TliqInlet", self.t_ruecklauf[idx] + 273.15)

        # 2. Aktion ausführen (True = 1, False = 0)
        self.fmu.set("reverseCycle", bool(action == 1))

        # 3. Physik simulieren (100 Sekunden pro RL-Aktion)
        self.fmu.do_step(
            current_time=self.current_step * self.step_seconds,
            step_size=self.step_seconds,
        )

        # 4. State & Reward berechnen
        # Wichtig: Beobachtung soll die Aktion des vorherigen Schritts enthalten.
        # `self.prev_action` hält diese historische Aktion; erst nach Berechnung
        # der Beobachtung aktualisieren wir sie mit der aktuellen Aktion.
        obs = self._get_obs()
        cop_aktuell = float(self.fmu.get("COP"))

        # Carnot-COP berechnen: T_hot / (T_hot - T_cold)
        # Verwende Vorlauf-/Rücklauftemperatur als heißes Reservoir
        T_hot = float(self.t_ruecklauf[idx] + 273.15)
        T_cold = float(self.t_air[idx] + 273.15)
        # Schutz gegen sehr kleine Nenner
        denom = max(T_hot - T_cold, 0.1)
        carnot_cop = T_hot / denom

        # Reward als Verhältnis von realem COP zur theoretischen Carnot-COP
        reward = cop_aktuell / carnot_cop
        # Bestrafung für das Abtauen selbst (verbraucht Energie)
        if action == 1:
            reward -= 2.0

        # Logge jede Aktion==1 in eine per-process CSV unter ./tb_logs/
        if int(action) == 1:
            try:
                os.makedirs("./tb_logs/", exist_ok=True)
                log_path = os.path.join("./tb_logs", f"actions_{os.getpid()}.csv")
                with open(log_path, "a", encoding="utf-8") as f:
                    # Format: global_idx,episode_start_idx,step_in_episode,action,reward,pid,wall_time
                    f.write(
                        f"{idx},{self.start_idx},{self.current_step},{int(action)},{reward},{os.getpid()},{pd.Timestamp.now()}\n"
                    )
            except Exception:
                pass

        # jetzt die vorherige Aktion für den nächsten Schritt aktualisieren
        self.prev_action = int(action)
        self.current_step += 1
        done = self.current_step >= self.episode_length

        # Liefere die Aktion im Info-Dict mit (kann von Monitor/Logger genutzt werden)
        info = {"action_taken": int(action)}
        return obs, reward, done, False, info

    def _get_obs(self):
        idx = self.start_idx + self.current_step
        # Rückgabe in KELVIN (wird von VecNormalize automatisch um 0 zentriert)
        # Rücklauftemperatur wird NICHT mehr Teil der Observation.
        return np.array(
            [
                self.t_air[idx] + 273.15,
                self.fmu.get("T_Verdampfer"),
                float(self.prev_action),
            ],
            dtype=np.float32,
        )


# ==========================================
# 3. BUG-FIX CALLBACK (VEC-NORMALIZE RETTER)
# ==========================================
class SaveVecNormalizeCallback(BaseCallback):
    """
    Dieser Callback speichert die VecNormalize Statistiken regelmäßig ab.
    Verhindert "Bug 1", bei dem der Agent beim Laden erblindet.
    """

    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"qrdqn_model_{self.n_calls}_steps")
            self.model.save(model_path)

            # HIER PASSIERT DIE MAGIE: Speichern der Statistiken
            stats_path = os.path.join(
                self.save_path, f"vec_normalize_{self.n_calls}_steps.pkl"
            )
            self.training_env.save(stats_path)
            if self.verbose > 0:
                print(
                    f"[Callback] Checkpoint und VecNormalize gespeichert bei Step {self.n_calls}"
                )
        return True


# ==========================================
# 4. DAS HAUPTSKRIPT (MULTIPROCESSING)
# ==========================================
# WICHTIG: Unter Windows MUSS multiprocessing in __main__ stehen!
if __name__ == "__main__":
    script_start = time.perf_counter()
    print("Starte RL-Trainings-Pipeline...")
    try:
        # Pro FMU-Schritt simulierte Rechenzeit in Millisekunden (Default: 100 ms).
        # Kann ohne Code-Änderung via Umgebungsvariable überschrieben werden:
        #   set DUMMY_FMU_DELAY_MS=250
        dummy_fmu_delay_ms = float(os.getenv("DUMMY_FMU_DELAY_MS", "100"))
        print(f"Simulierte FMU-Rechenzeit pro Schritt: {dummy_fmu_delay_ms:.1f} ms")

        # ---------------------------------------------------------
        # HIER SPÄTER DEIN ECHTES CSV LADEN:
        df = pd.read_csv(
            "RL_Trainingsdaten_Frankfurt_2010_bis_2020_ohne_April_bis_September_100s.csv"
        )
        # ---------------------------------------------------------
        # DUMMY DATENGENERATOR FÜR DEN DRY-RUN:
        # print("Erzeuge Dummy-Wetterdaten...")
        # dummy_len = 50000
        # df = pd.DataFrame({
        #    'Tair_degC': np.random.normal(loc=2, scale=5, size=dummy_len),
        #    'phiAir': np.random.normal(loc=85, scale=5, size=dummy_len),
        #    'T_Ruecklauf': np.random.normal(loc=35, scale=2, size=dummy_len),
        #    'Theta': np.random.uniform(0.1, 1.0, size=dummy_len) # Immer Winter im Dummy
        # })

        # Daten in Numpy konvertieren für maximale Performance
        data_dict = {
            "Tair_degC": df["Tair_degC"].values,
            "phiAir": df["phiAir"].values,
            "T_Ruecklauf": df["T_Ruecklauf"].values,
            "Theta": df["Theta"].values,
        }

        # Environment Fabrik-Funktion
        def make_env():
            def _init():
                env = HeatPumpEnv(
                    data_dict,
                    "dein_modell_pfad.fmu",
                    simulated_delay_ms=dummy_fmu_delay_ms,
                )
                return Monitor(env)

            return _init

        # 8 Parallele Umgebungen starten
        num_cpu = 8
        print(f"Starte {num_cpu} parallele Umgebungen...")
        # Nutze DummyVecEnv falls SubprocVecEnv beim Testen abstürzt (besser fürs Debugging!)
        env = SubprocVecEnv([make_env() for i in range(num_cpu)])

        # WICHTIG: VecNormalize anwenden! (Normalisiert Observationen auf Mean 0, Std 1)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # Checkpoint Callback einrichten
        checkpoint_dir = "./rl_checkpoints/"
        save_callback = SaveVecNormalizeCallback(
            save_freq=10000 // num_cpu, save_path=checkpoint_dir
        )

        # Stelle sicher, dass TensorBoard-Ordner existiert
        os.makedirs("./tb_logs/", exist_ok=True)

        # RL-Modell initialisieren
        print("Initialisiere QRDQN Agenten...")
        model = QRDQN(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./tb_logs/",
            learning_rate=3e-4,
            batch_size=256,
        )

        # Training starten (10x längere Laufzeit)
        print("Starte Training (Abbruch mit STRG+C)...")
        model.learn(
            total_timesteps=50000,
            callback=save_callback,
            tb_log_name="qrdqn_heatpump_run",
            progress_bar=True,
        )

        # Finales Speichern
        model.save("qrdqn_heatpump_final")
        env.save("vec_normalize_final.pkl")
        print("Training erfolgreich beendet!")
    finally:
        total_runtime_s = time.perf_counter() - script_start
        print(
            f"Gesamtlaufzeit main.py: {total_runtime_s:.2f} s ({total_runtime_s / 60:.2f} min)"
        )
