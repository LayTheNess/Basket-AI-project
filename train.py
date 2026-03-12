from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from BasketBallShooterV2 import BasketballShooterEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

env = BasketballShooterEnv()
check_env(env)
SAVE_FREQ = 10000
def make_env():
    env = BasketballShooterEnv()
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=32,
    gamma=0.99,
    tensorboard_log="./tensorboard/"
)
checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path="./checkpoints", name_prefix="ppo_shooter_checkpoint")

model.learn(total_timesteps=90000, callback=checkpoint_callback, tb_log_name="basket_ppo_v2")
model.save("basket_ppo")

print("Training complete.")
