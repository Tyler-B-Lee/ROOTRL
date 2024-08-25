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


def training_model_manager_wrapper(env: gym.Env):
    class TrainingModelManagerEnv(env):
        # wrapper over the normal single player env, but loads the correct opponent models
        def __init__(self, rules_type, model_type, model_players, opponent_type, verbose):
            super(TrainingModelManagerEnv, self).__init__(rules_type, model_players, verbose)
            self.opponent_type = opponent_type
            self.model_type = model_type
            self.rules_type = rules_type

            self.opponent_models = load_all_models(self)
            self.best_model_name = get_best_model_name(self.model_type)

            def name_to_id(string):
                if "MC" in string:
                    return 0
                if "EY" in string:
                    return 1
                if "WA" in string:
                    return 2
                if "VB" in string:
                    return 3
            if 'Hates' in model_type:
                self.target_id = name_to_id(model_type)
                self.multiplier = -1
            elif 'Helps' in model_type:
                self.target_id = name_to_id(model_type)
                self.multiplier = 1
            elif 'Main' in model_type:
                self.target_id = self.main_player_id
                self.multiplier = 1
            else:
                raise Exception(f"Unknown Player type to train from environment '{model_type}'")
            logger.info(f"> Loaded agent env '{model_type}', Target: {self.target_id} / Multi: {self.multiplier}")

        def setup_opponents(self):
            self.agents = [None] * 4
            for pid in range(4):
                if pid == self.main_player_id:
                    continue

                if self.opponent_type == 'rules':
                    self.agents[pid] = Agent(name=f'{self.rules_type} rules',action_space_size=config.ACTION_SPACE_SIZES[pid])
                else:
                    # incremental load of new model
                    best_model_name = get_best_model_name(self.model_type)
                    if self.best_model_name != best_model_name:
                        self.opponent_models.append(load_model(self.model_type, best_model_name, self))
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
            super(TrainingModelManagerEnv, self).reset(seed)
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
            rewards_to_return = [0,0,0,0]
            terminated = None
            truncated = None

            while self.current_player_num != self.main_player_id:
                self.render()
                action = self.current_agent.choose_action(self, choose_best_action = False, mask_invalid_actions = False)
                observation, reward_list, terminated, truncated, _ = super(TrainingModelManagerEnv, self).step(action)
                for i in range(4):
                    rewards_to_return[i] += reward_list[i]
                logger.debug(f'Rewards: {reward_list} | Total to return: {rewards_to_return}')
                logger.debug(f'Terminated: {terminated} | Truncated: {truncated}')
                if (terminated or truncated):
                    break

            return observation, rewards_to_return, terminated, truncated, None


        def step(self, action):
            self.render()
            observation, rewards_to_return, terminated, truncated, _ = super(TrainingModelManagerEnv, self).step(action)
            logger.debug(f'Action played by agent: {action}')
            logger.debug(f'Rewards: {rewards_to_return}')
            logger.debug(f'Terminated: {terminated} | Truncated: {truncated}')

            if not (terminated or truncated):
                package = self.continue_game()
                if package[0] is not None:
                    observation, reward_list, terminated, truncated, _ = package
                    for i in range(4):
                        rewards_to_return[i] += reward_list[i]
                    logger.debug(f'Rewards from other turns: {reward_list} | New Total to return: {rewards_to_return}')

            agent_reward = rewards_to_return[self.target_id] * self.multiplier
            logger.debug(f'\nReward To Agent: {agent_reward}')

            if (terminated or truncated):
                self.render()
                observation = self.get_main_observation(self.main_player_id)
            
            logger.debug(f"Agent reward for their step: {agent_reward}")

            return observation, agent_reward, terminated, truncated, {} 

    return TrainingModelManagerEnv