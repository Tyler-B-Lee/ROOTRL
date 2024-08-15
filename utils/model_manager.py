import random
import gymnasium as gym

from utils.files import load_model, load_all_models, get_best_model_name
from utils.agents import Agent

import config

import logging
# logging.basicConfig(
#     filename='file.log',
#     format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
#     filemode='w',
#     # level=logging.DEBUG
# )
logger = logging.getLogger(__name__)


def model_manager_wrapper(env: gym.Env):
    class ModelManagerEnv(env):
        # wrapper over the normal single player env, but loads the correct opponent models
        def __init__(self, name, opponent_type, verbose):
            super(ModelManagerEnv, self).__init__(name, verbose)
            self.opponent_type = opponent_type
            self.opponent_models = load_all_models(self)
            self.best_model_name = get_best_model_name(self.name)

        def setup_opponents(self):
            self.agents = [None] * 4
            for pid in range(4):
                if pid == self.main_player_id:
                    continue

                if self.opponent_type == 'rules':
                    self.agents[pid] = Agent(name='rules',action_space_size=config.ACTION_SPACE_SIZES[pid])
                else:
                    # incremental load of new model
                    best_model_name = get_best_model_name(self.name)
                    if self.best_model_name != best_model_name:
                        self.opponent_models.append(load_model(self, best_model_name ))
                        self.best_model_name = best_model_name

                    if self.opponent_type == 'random':
                        start = 0
                        end = len(self.opponent_models) - 1
                        i = random.randint(start, end)
                        self.agents[pid] = Agent(name='ppo_opponent', action_space_size=config.ACTION_SPACE_SIZES[pid], model=self.opponent_models[i])

                    elif self.opponent_type == 'best':
                        self.agents[pid] = Agent(name='ppo_opponent', action_space_size=config.ACTION_SPACE_SIZES[pid], model=self.opponent_models[-1])

                    elif self.opponent_type == 'mostly_best':
                        j = random.uniform(0,1)
                        if j < 0.8:
                            self.agents[pid] = Agent(name='ppo_opponent', action_space_size=config.ACTION_SPACE_SIZES[pid], model=self.opponent_models[-1])
                        else:
                            start = 0
                            end = len(self.opponent_models) - 1
                            i = random.randint(start, end)
                            self.agents[pid] = Agent(name='ppo_opponent', action_space_size=config.ACTION_SPACE_SIZES[pid], model=self.opponent_models[i])

                    elif self.opponent_type == 'base':
                        self.agents[pid] = Agent(name='base', action_space_size=config.ACTION_SPACE_SIZES[pid], model=self.opponent_models[0])
            
            logger.debug(f"Agents loaded: {self.agents}")
            try:
                #if self.players is defined on the base environment
                logger.debug(f'Agent plays as Player {self.main_player_id}')
            except:
                pass


        def reset(self, seed:int=None):
            super(ModelManagerEnv, self).reset(seed)
            logger.debug(f"Post super-setup main player: {self.main_player_id} | Current player num: {self.current_player_num}")
            self.setup_opponents()

            if self.current_player_num != self.main_player_id:   
                logger.debug(f"Pre continue_game main player: {self.main_player_id} | Current player num: {self.current_player_num}")
                self.continue_game()

            return self.observation, {}

        @property
        def current_agent(self):
            return self.agents[self.current_player_num]

        def continue_game(self):
            observation = None
            reward = None
            terminated = None
            truncated = None

            while self.current_player_num != self.main_player_id:
                self.render()
                action = self.current_agent.choose_action(self, choose_best_action = False, mask_invalid_actions = False)
                observation, reward, terminated, truncated, _ = super(ModelManagerEnv, self).step(action)
                logger.debug(f'Rewards: {reward}')
                logger.debug(f'Terminated: {terminated} | Truncated: {truncated}')
                if (terminated or truncated):
                    break

            return observation, reward, terminated, truncated, None


        def step(self, action):
            self.render()
            observation, reward, terminated, truncated, _ = super(ModelManagerEnv, self).step(action)
            logger.debug(f'Action played by agent: {action}')
            logger.debug(f'Rewards: {reward}')
            logger.debug(f'Terminated: {terminated} | Truncated: {truncated}')

            if not (terminated or truncated):
                package = self.continue_game()
                if package[0] is not None:
                    observation, reward, terminated, truncated, _ = package


            agent_reward = reward[self.main_player_id]
            logger.debug(f'\nReward To Agent: {agent_reward}')

            if (terminated or truncated):
                self.render()
                observation = self.get_main_observation()

            return observation, agent_reward, terminated, truncated, {} 

    return ModelManagerEnv