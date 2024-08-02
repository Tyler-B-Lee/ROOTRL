import gym
import numpy as np
import os
import time

import config
import utils.agents

from stable_baselines import logger
from stable_baselines.ppo1 import PPO1

from .rootGameClasses.rootMechanics import *

# docker-compose exec app tensorboard --logdir ./logs
# notes:
# - os (optimizer stepsize) I think is the 'learning rate' parameter
#        - Should be decreased linearly over training to 0 or very small
#        - Some papers have it as small as 1e-6 at the end

# start
# docker-compose exec app mpirun -np 2 python3 train.py -e root4pbaseeyrie -ne 25 -ef 20480 -tpa 2048 -ent 0.025 -ob 128 -g 0.995 -oe 6 -t 0

MARQUISE_ID = 0
EYRIE_ID = 1
ALLIANCE_ID = 2
VAGABOND_ID = 3

MAIN_PLAYER_ID = EYRIE_ID
BEST_MODEL_CHANCE = 0.8
OPPONENT_INFO = [(MARQUISE_ID,"root4pbasemarquise"),(ALLIANCE_ID,"root4pbasealliance"),(VAGABOND_ID,"root4pbasevagabond")]

# marquise - 1538 obs / 548 actions
# eyrie - 1522 / 492
# alliance - 1571 / 530
# vagabond - 1547 / 604

class rootEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(rootEnv, self).__init__()
        self.name = 'root4pbaseeyrie'
        self.n_players = 1
        self.manual = manual

        self.action_space = gym.spaces.Discrete(492)
        self.observation_space = gym.spaces.Box(-1, 1, (
            1522
            + self.action_space.n
            , )
        )  
        self.verbose = verbose
        self.env = RootGame(CHOSEN_MAP,STANDARD_DECK_COMP)

        self.opp_model_pool = {}
        self.best_opponent_models = [None] * 4

        for opp_id,opp_env_name in OPPONENT_INFO:
            logger.info(f"\nLoading {ID_TO_PLAYER[opp_id]} Models to train against...")
            self.opp_model_pool[opp_id] = []
            opp_dir = os.path.join(config.MODELDIR, opp_env_name)

            opp_files = os.listdir(opp_dir)
            for model_file_name in opp_files:
                full_path = os.path.join(opp_dir,model_file_name)
                if os.path.exists(full_path):
                    logger.info(f"\tLoading {model_file_name} to play / train against...")
                    cont = True
                    while cont:
                        try:
                            ppo_model = PPO1.load(full_path)
                            cont = False
                        except Exception as e:
                            time.sleep(5)
                            print(e)
                else:
                    raise Exception(f'\n{full_path} not found!')

                this_model = utils.agents.Agent(ID_TO_PLAYER[opp_id] + " " + model_file_name[:-4], ppo_model)
                if model_file_name == "best_model.zip":
                    self.best_opponent_models[opp_id] = this_model
                else:
                    self.opp_model_pool[opp_id].append(this_model)

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_num]
    
    @property
    def opposing_player(self) -> Player:
        i = (self.current_player_num + 1) % 2
        return self.players[i]
        
    @property
    def observation(self):
        ret = np.append(self.env.get_eyrie_observation(),self.legal_actions)
        return ret

    @property
    def legal_actions(self):
        ret = np.zeros(self.action_space.n)
        ret.put(self.env.legal_actions(), 1)
        return ret
    
    def run_opponent_turn(self):
        opponent_id = self.env.to_play()
        opp = self.opponent_models[opponent_id]
        logger.debug(f'\n{opp.name} model choices')
        # action_chosen = random.choice(self.get_legal_action_numbers())

        if opponent_id == PIND_MARQUISE:
            la = np.zeros(548)
            la.put(self.env.legal_actions(),1)
            opponent_obs = np.append(self.env.get_marquise_observation(), la)
        elif opponent_id == PIND_ALLIANCE:
            la = np.zeros(530)
            la.put(self.env.legal_actions(),1)
            opponent_obs = np.append(self.env.get_alliance_observation(), la)
        else:
            la = np.zeros(604)
            la.put(self.env.legal_actions(),1)
            opponent_obs = np.append(self.env.get_vagabond_observation(), la)

        action_probs = opp.model.action_probability(opponent_obs)
        value = opp.model.policy_pi.value(np.array([opponent_obs]))[0]
        logger.debug(f"Value {value:.3f}")

        opp.print_top_actions(action_probs)

        action_probs = utils.agents.mask_actions(la, action_probs)
        logger.debug('Masked ->')
        opp.print_top_actions(action_probs)

        action_chosen = utils.agents.sample_action(action_probs)
        logger.debug(f'Sampled action {action_chosen} chosen')
        
        reward, done = self.env.step(action_chosen)

        return reward,done

    def step(self, action):
        reward, self.done = self.env.step(action)
        self.current_player_num = (self.env.to_play() - MAIN_PLAYER_ID)

        # play the game until it comes back to the main player's turn or the game ends
        while self.current_player_num != 0 and (not self.done):
            r, self.done = self.run_opponent_turn()

            reward = [reward[i] + r[i] for i in range(4)]
            self.current_player_num = (self.env.to_play() - MAIN_PLAYER_ID)

        return self.observation, reward[MAIN_PLAYER_ID:MAIN_PLAYER_ID + 1], self.done, {}

    def reset(self):
        self.done = False
        self.env.reset()
        self.current_player_num = (self.env.to_play() - MAIN_PLAYER_ID)
        
        self.opponent_models = [None] * 4
        for i,name in OPPONENT_INFO:
            if random.random() < BEST_MODEL_CHANCE:
                self.opponent_models[i] = self.best_opponent_models[i]
            else:
                self.opponent_models[i] = random.choice(self.opp_model_pool[i])
            logger.debug(f"> Chosen {ID_TO_PLAYER[i]} Model: {self.opponent_models[i].name}")

        # play the game until it comes back to the main player's turn or the game ends
        while self.current_player_num != 0 and (not self.done):
            r, self.done = self.run_opponent_turn()

            self.current_player_num = (self.env.to_play() - MAIN_PLAYER_ID)

        return self.observation

    def render(self, mode='human', close=False):
        if close:
            return
        
        # self.env.render()

        if self.verbose:
            logger.debug(f'\nObservation: \n{[i if o == 1 else (i,o) for i,o in enumerate(self.observation) if o != 0]}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')


    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for Geschenkt!')

# if __name__ == "__main__":
#     env = rootEnv()
#     env.reset()
#     done = False
#     total_rewards = np.zeros(1)
#     while not done:
#         legal_actions = env.legal_actions