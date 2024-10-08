import os
import random
import gymnasium as gym
import logging
import config
from utils.agents import Agent
from utils.files import load_model, logger
# logging.basicConfig(
#     filename='file.log',
#     format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
#     filemode='w',
#     # level=logging.DEBUG
# )
logger = logging.getLogger(__name__)


def load_all_models(env):
    """
    Load all of the opponent models that could be used for opponents in the given model manager environment.

    Returns a dictionary of the loaded models, with the faction name as the key.
    Always loads the "base" model of the main model type first. Then loads rules agents for the other factions.
    If the environment is an Arena environment, loads all models from the zoo.

    :param env: The model manager environment to load the models for.

    :return: A dictionary of the loaded Agents, with the faction name as the key pointing to a list of Agents.
    """
    zoo_dir = config.MODELDIR if not env.kaggle else ('/kaggle/input/rootrl-files/' + config.MODELDIR)

    # first ensure that the factions we want to load models of
    # for this environment atleast have a "base" model
    base_filename = os.path.join(zoo_dir, env.model_type, "base.pt")
    if not os.path.exists(base_filename):
        load_model(env.model_type, "base.pt", zoo_dir, env)

    agents = {'Marquise': [], 'Eyrie': [], 'Alliance': [], 'Vagabond': []}
    logger.info("\n> Loading rules agents...")
    # load rules agents first
    for faction_id, faction, aspace_size in zip([0,1,2,3], agents.keys(), config.ACTION_SPACE_SIZES):
        if faction_id == env.main_player_id:
            continue
        rules_agent = Agent(f'{faction} rules', faction_id, aspace_size, model=None)
        agents[faction].append(rules_agent)

    if env.rules_type == 'Arena':
        logger.info("\n> Loading Arena models...")
        # load the model agents next from each folder in the zoo
        for folder_name in os.listdir(zoo_dir):
            for faction_id, faction, aspace_size in zip([0,1,2,3], agents.keys(), config.ACTION_SPACE_SIZES):
                if faction_id == env.main_player_id:
                    continue
                if faction in folder_name:
                    for model_name in os.listdir(os.path.join(zoo_dir, folder_name)):
                        if 'best_model' in model_name:
                            continue

                        model = load_model(folder_name, model_name, zoo_dir, None)

                        model_agent = Agent(f'{folder_name} | {model_name}', faction_id, aspace_size, model)
                        agents[faction].append(model_agent)
                        logger.debug(f"Loaded model '{model_name}' for {faction} from {folder_name}")
    return agents


def training_model_manager_wrapper(env: gym.Env):
    class TrainingModelManagerEnv(env):
        # wrapper over the normal single player env, but loads the correct opponent models
        def __init__(self, rules_type, model_type, model_players, opponent_type, verbose, kaggle):
            super(TrainingModelManagerEnv, self).__init__(rules_type, model_players, verbose)
            self.opponent_type = opponent_type
            self.model_type = model_type
            self.rules_type = rules_type
            self.kaggle = kaggle

            # find which faction the main player is
            if "Marquise" in model_type:
                self.main_player_id = 0
            elif "Eyrie" in model_type:
                self.main_player_id = 1
            elif "Alliance" in model_type:
                self.main_player_id = 2
            elif "Vagabond" in model_type:
                self.main_player_id = 3
            else:
                raise Exception(f"No valid faction found in '{model_type}'")

            self.opponent_models = load_all_models(self)
            # self.best_model_name = get_best_model_name(self.model_type)

            # logger.info(f"> Loaded agent env '{model_type}', Target: {self.target_id} / Multi: {self.multiplier}")

        def setup_opponents(self):
            self.model_players = [self.main_player_id]
            self.agents = [None] * 4
            for pid,faction_name in enumerate(self.opponent_models.keys()):
                if pid == self.main_player_id:
                    continue

                # load a random agent for this opponent with a 20% chance
                if random.random() < 0.2:
                    self.agents[pid] = random.choice(self.opponent_models[faction_name])
                else:
                    # load the agent with the best model for this faction
                    self.agents[pid] = self.opponent_models[faction_name][-1]
            
            # output each agent loaded
            for agent in self.agents:
                if agent is not None:
                    logger.debug(f"Loaded agent '{agent.name}'")
                    if 'rules' not in agent.name.lower():
                        self.model_players.append(agent.faction_id)

        def reset(self, seed:int=None):
            super(TrainingModelManagerEnv, self).reset(seed)
            # logger.debug(f"Post super-setup main player: {self.main_player_id} | Current player num: {self.current_player_num}")
            self.setup_opponents()

            if self.current_player_num != self.main_player_id:   
                # logger.debug(f"Pre continue_game main player: {self.main_player_id} | Current player num: {self.current_player_num}")
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

            # agent_reward = rewards_to_return[self.target_id] * self.multiplier
            agent_reward = rewards_to_return[self.main_player_id]
            logger.debug(f'\nReward To Agent: {agent_reward}')

            if (terminated or truncated):
                self.render()
                observation = self.get_main_observation(self.main_player_id)
            
            logger.debug(f"Agent reward for their step: {agent_reward}")

            return observation, agent_reward, terminated, truncated, {} 

    return TrainingModelManagerEnv