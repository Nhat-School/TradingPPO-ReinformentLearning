from stable_baselines3 import PPO

# Nạp model
model = PPO.load("ReinforcementTrading_Part_1/checkpoints/ppo_eurusd_50000_steps.zip")

print("--- Model Information ---")
print(f"Policy: {model.policy}")
print(f"Learning rate: {model.learning_rate}")
print(f"Model loaded successfully!")

