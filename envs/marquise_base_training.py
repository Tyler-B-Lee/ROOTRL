import gymnasium as gym
import numpy as np
import config

from .rootGameClasses.rootMechanics import *

# Game Obs length: 1487 (down from 4474 i think)
# docker-compose exec app tensorboard --logdir ./logs
# notes:
# - os (optimizer stepsize) I think is the 'learning rate' parameter
#        - Should be decreased linearly over training to 0 or very small
#        - Some papers have it as small as 1e-6 at the end

# start
# docker-compose exec app mpirun -np 2 python3 train.py -e root4pbasemarquise -ne 25 -ef 20480 -tpa 2048 -ent 0.025 -ob 128 -g 0.995 -oe 6 -t 0

MARQUISE_ID = 0
EYRIE_ID = 1
ALLIANCE_ID = 2
VAGABOND_ID = 3

# marquise - 1538 obs / 548 actions
# eyrie - 1522 / 492
# alliance - 1571 / 530
# vagabond - 1547 / 604

class MarquiseMainBaseEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(MarquiseMainBaseEnv, self).__init__()
        self.name = 'MarquiseMainBase'
        self.n_players = 4
        self.manual = manual
        self.main_player_id = MARQUISE_ID

        self.action_space = gym.spaces.Discrete(config.ACTION_SPACE_SIZES[MARQUISE_ID])
        self.observation_space = gym.spaces.Box(-1, 1, (
            config.NUM_OBS_MARQUISE
            + self.action_space.n
            , )
        )  
        self.verbose = verbose
        self.env = RootGame(CHOSEN_MAP, STANDARD_DECK_COMP, 1000, 1500)

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_num]
    
    @property
    def opposing_player(self) -> Player:
        i = (self.current_player_num + 1) % 2
        return self.players[i]
    
    def get_main_observation(self):
        la = np.zeros(self.action_space.n)
        la.put(self.env.legal_actions(), 1.0)

        ret = np.append(self.env.get_marquise_observation(), la)
        return ret.astype(np.float32)
        
    @property
    def observation(self):
        # since the opponents are always rules based, don't calculate their obs/legal actions
        if (self.env.to_play() - self.main_player_id) != 0:
            return [0]
        return self.get_main_observation()

    @property
    def legal_actions(self):
        if (self.env.to_play() - self.main_player_id) != 0:
            ret = np.zeros(604)
        else:
            ret = np.zeros(self.action_space.n)
        ret.put(self.env.legal_actions(), 1.0)
        return ret
    
    def run_opponent_turn(self):
        opponent_id = self.env.to_play()
        # action_chosen = random.choice(self.get_legal_action_numbers())
        la = self.env.legal_actions()
        la_set = set(la)

        # players always ambush if they can
        ans_set = la_set.intersection(AMBUSH_ACTIONS_SET)

        # Vagabond explores if they can, otherwise moves/slips at random
        if len(ans_set) == 0 and opponent_id == PIND_VAGABOND:
            ans_set = la_set.intersection({VB_EXPLORE})
        if len(ans_set) == 0 and opponent_id == PIND_VAGABOND:
            ans_set = la_set.intersection(SLIP_ACTIONS_SET)
        
        # Alliance always spreads sympathy if they can at random
        if len(ans_set) == 0 and opponent_id == PIND_ALLIANCE:
            ans_set = la_set.intersection(SPREAD_SYM_ACTIONS_SET)

        # otherwise, players try skipping optional actions
        if len(ans_set) == 0:
            ans_set = la_set.intersection(SKIP_ACTIONS_SET)
        
        # otherwise, players play a random action possible in this position
        if len(ans_set) == 0:
            action_chosen = random.choice(la)
            logger.debug(f'Sampled action {action_chosen} chosen from all actions')
        else:
            action_chosen = random.choice( list(ans_set) )
            logger.debug(f'Sampled action {action_chosen} chosen from subset')
        
        reward, terminated, truncated = self.env.step(action_chosen)

        return reward, terminated, truncated

    def step(self, action):
        reward, self.terminated, self.truncated = self.env.step(action)
        self.current_player_num = (self.env.to_play() - self.main_player_id)

        # play the game until it comes back to the main player's turn or the game ends
        # while self.current_player_num != 0 and not (self.terminated or self.truncated):
        #     r, self.terminated, self.truncated = self.run_opponent_turn()

        #     reward = [reward[i] + r[i] for i in range(4)]
        #     self.current_player_num = (self.env.to_play() - self.main_player_id)

        # return self.observation, reward[self.main_player_id], self.terminated, self.truncated, {}
        return self.observation, reward, self.terminated, self.truncated, {}

    def reset(self, seed:int=None):
        super().reset(seed=seed)
        self.terminated = self.truncated = False
        self.env.randomize()
        self.current_player_num = (self.env.to_play() - self.main_player_id)

        # play the game until it comes back to the main player's turn or the game ends
        # while self.current_player_num != 0 and not (self.terminated or self.truncated):
        #     r, self.terminated, self.truncated = self.run_opponent_turn()
            
        #     self.current_player_num = (self.env.to_play() - self.main_player_id)

        # return self.observation, {}
        # real observation returned on the model manager wrapper env class
        return [], {}

    def render(self, mode='human', close=False):
        if close:
            return
        
        # self.env.render()

        if self.verbose:
            logger.debug(f'\nObservation: \n{[i if o == 1 else (i,o) for i,o in enumerate(self.observation) if o != 0]}')
        
        if not (self.terminated or self.truncated):
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')


    def rules_move(self, action_space_size:int):
        opponent_id = self.env.to_play()
        assert (opponent_id != self.main_player_id)

        ret = np.zeros(action_space_size)
        # action_chosen = random.choice(self.get_legal_action_numbers())
        la = self.env.legal_actions()
        la_set = set(la)

        # players always ambush if they can
        ans_set = la_set.intersection(AMBUSH_ACTIONS_SET)

        # Vagabond explores if they can, otherwise moves/slips at random
        if len(ans_set) == 0 and opponent_id == PIND_VAGABOND:
            ans_set = la_set.intersection({VB_EXPLORE})
        if len(ans_set) == 0 and opponent_id == PIND_VAGABOND:
            ans_set = la_set.intersection(SLIP_ACTIONS_SET)
        
        # Alliance always spreads sympathy if they can at random
        if len(ans_set) == 0 and opponent_id == PIND_ALLIANCE:
            ans_set = la_set.intersection(SPREAD_SYM_ACTIONS_SET)

        # otherwise, players try skipping optional actions
        if len(ans_set) == 0:
            ans_set = la_set.intersection(SKIP_ACTIONS_SET)
        
        # otherwise, players play a random action possible in this position
        if len(ans_set) == 0:
            ret.put(la, 1/len(la))
        else:
            ans_list = list(ans_set)
            ret.put(ans_list, 1/len(ans_list) )
        
        return ret
        
        

if __name__ == "__main__":
    env = MarquiseMainBaseEnv()
    env.reset()
    terminated = truncated = False
    total_rewards = np.zeros(N_PLAYERS)
    while not (terminated or truncated):
        legal_actions = env.env.legal_actions()
        logger.debug(f"> Action {env.env.num_actions_played} - Player: {ID_TO_PLAYER[env.env.current_player]}")
        logger.info(f"Legal Actions: {legal_actions}")
        # print(f"Player: {ID_TO_PLAYER[env.current_player]}")
        # print(f"> Action {action_count} - Legal Actions: {legal_actions}")

        # action = -1
        # while action not in legal_actions:
        #     action = int(input("Choose a valid action: "))
        action = random.choice(legal_actions)
        # print(f"\tAction Chosen: {action}")
        logger.info(f"\t> Action Chosen: {action}")
        obs,reward,terminated,truncated,info = env.step(action)

        logger.debug(f"- Reward for this action: {reward}")
        total_rewards += reward
        logger.debug(f"\t> New reward total: {total_rewards}")