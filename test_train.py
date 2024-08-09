from envs.marquise_base_training import *
from models.marquise_model import *

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


logger.setLevel(logging.ERROR)
env = MarquiseMainBaseEnv()
# model = PPO(CustomPolicy, env, verbose=1)
# model.learn(total_timesteps=10_000)

# model.save("test_model")

# TESTING
loaded_model = PPO.load("test_model")
# env = loaded_model.get_env()

logger.setLevel(logging.DEBUG)
# print(evaluate_policy(loaded_model, env, n_eval_episodes=10, return_episode_rewards=True))

loaded_model