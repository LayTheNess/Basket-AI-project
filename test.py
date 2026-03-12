from stable_baselines3 import PPO
from BasketBallShooterV2 import BasketballShooterEnv

model = PPO.load("basket_ppo")

env = BasketballShooterEnv(render_mode="human")

episodes = 200
score = 0

for _ in range(episodes):
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    print(reward)
    if env.scored:
        score += 1

accuracy = score / episodes
print(f"Accuracy: {accuracy*100:.2f}%")


env.close()
