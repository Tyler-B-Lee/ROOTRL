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

class RootEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, rules_type:str, model_players:list, verbose = False, manual = False):
        super(RootEnv, self).__init__()

        if rules_type == 'Base':
            self.rules_move = self.starter_rules_move
            self.game = RootGame(CHOSEN_MAP, STANDARD_DECK_COMP, 900, 1500)
        elif rules_type == 'Algo':
            self.rules_move = self.algo_rules_move
            self.game = RootGame(CHOSEN_MAP, STANDARD_DECK_COMP, 500, 1500)
        else:
            raise Exception(f"Unknown rules_type '{rules_type}'")

        self.model_players = model_players
        # when training, the first model ID given will be the ID of the player model being trained
        self.main_player_id = self.model_players[0]
        
        # logger.info(f"Loading Algo env for model type '{model_type}'")

        self.n_players = 4
        self.verbose = verbose
        self.manual = manual

        self.action_space = gym.spaces.Discrete(config.ACTION_SPACE_SIZES[self.main_player_id])
        self.observation_space = gym.spaces.Box(-1, 1, (
            config.OBS_SPACE_SIZES[self.main_player_id]
            + self.action_space.n
            , )
        )  

        self.obs_methods = [self.game.get_marquise_observation,self.game.get_eyrie_observation,self.game.get_alliance_observation,self.game.get_vagabond_observation]

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_num]
    
    # @property
    # def opposing_player(self) -> Player:
    #     i = (self.current_player_num + 1) % 2
    #     return self.players[i]
    
    def get_main_observation(self, pid:int):
        la = np.zeros(config.ACTION_SPACE_SIZES[pid])
        la.put(self.game.legal_actions(), 1.0)

        obs_method = self.obs_methods[pid]

        ret = np.append(obs_method(), la)
        return ret.astype(np.float32)
        
    @property
    def observation(self):
        # since some opponents are rule-based (no obs needed), don't calculate their obs/legal actions
        if self.game.to_play() not in self.model_players:
            return [0]
        return self.get_main_observation(self.game.to_play())

    @property
    def legal_actions(self):
        if self.game.to_play() not in self.model_players:
            ret = np.zeros(604)
        else:
            ret = np.zeros(config.ACTION_SPACE_SIZES[self.game.to_play()])
        ret.put(self.game.legal_actions(), 1.0)
        return ret

    def step(self, action):
        reward_list, self.terminated, self.truncated = self.game.step(action)
        self.current_player_num = self.game.to_play()

        # play the game until it comes back to the main player's turn or the game ends
        # while self.current_player_num != 0 and not (self.terminated or self.truncated):
        #     r, self.terminated, self.truncated = self.run_opponent_turn()

        #     reward = [reward[i] + r[i] for i in range(4)]
        #     self.current_player_num = (self.env.to_play() - self.main_player_id)

        # return self.observation, reward[self.main_player_id], self.terminated, self.truncated, {}
        return self.observation, reward_list, self.terminated, self.truncated, {}

    def reset(self, seed:int=None):
        super().reset(seed=seed)
        self.terminated = self.truncated = False
        self.game.randomize(min_points=5)
        self.current_player_num = self.game.to_play()

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
            logger.debug(f'Legal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')

    def starter_rules_move(self, action_space_size:int):
        """
        Returns action probabilities for the current rules-based faction.
        
        This is the base-level rules bot, making basic choices for each
        faction to barely stay in the game and keep things progressing
        very slowly. Supposed to not be challenging to defeat at all, 
        mainly serving as a warm-up for training against.
        """
        opponent_id = self.game.to_play()
        assert (opponent_id not in self.model_players)

        ret = np.zeros(action_space_size)
        # action_chosen = random.choice(self.get_legal_action_numbers())
        la = self.game.legal_actions()
        la_set = set(la)

        # players always ambush if they can
        ans_set = la_set.intersection(AMBUSH_ACTIONS_SET)

        # Marquise recruits and uses field hospitals if possible
        if len(ans_set) == 0 and opponent_id == PIND_MARQUISE:
            ans_set = la_set.intersection({MC_RECRUIT} | FIELD_HOSPITALS_ACTIONS_SET)

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
            if opponent_id == PIND_MARQUISE:
                ans_set = la_set.intersection(MARQUISE_SKIP_ACTIONS_SET)
            elif opponent_id == PIND_EYRIE:
                ans_set = la_set.intersection(EYRIE_SKIP_ACTIONS_SET)
            elif opponent_id == PIND_ALLIANCE:
                ans_set = la_set.intersection(ALLIANCE_SKIP_ACTIONS_SET)
            elif opponent_id == PIND_VAGABOND:
                ans_set = la_set.intersection(VAGABOND_SKIP_ACTIONS_SET)
        
        # otherwise, players play a random action possible in this position
        if len(ans_set) == 0:
            ret.put(la, 1/len(la))
        else:
            ans_list = list(ans_set)
            ret.put(ans_list, 1/len(ans_list) )
        
        return ret


    def algo_rules_move(self, action_space_size:int):
        """
        Returns action probabilities for the current rules-based faction.
        
        This is the standard algo-bot. It serves as a sort of greedy-strategy
        for each faction, likely performing better than random for most factions.
        Actions that usually give points or easily lead to points are always
        chosen first, then actions are randomly chosen.

        Serves as a "brainless" opponent running a straightforward strategy
        that could be difficult to beat if you are unlucky, but should provide
        a decent CPU-level challenge to overcome.
        """
        opponent_id = self.game.to_play()
        assert (opponent_id not in self.model_players)

        ret = np.zeros(action_space_size)
        # action_chosen = random.choice(self.get_legal_action_numbers())
        la = self.game.legal_actions()
        la_set = set(la)

        # players always ambush or craft if they can
        ans_set = la_set.intersection(AMBUSH_ACTIONS_SET | CRAFT_ACTIONS_SET)
        if len(ans_set) == 0 and opponent_id != PIND_VAGABOND:
            ans_set = la_set.intersection(CRAFT_RC_ACTIONS_SET)

        # Marquise
        if len(ans_set) == 0 and opponent_id == PIND_MARQUISE:
            # First, recruit and use field hospitals if possible
            ans_set = la_set.intersection({MC_RECRUIT} | FIELD_HOSPITALS_ACTIONS_SET)
            if len(ans_set) == 0: # Try to build
                ans_set = la_set.intersection(MC_BUILD_ACTIONS_SET)
            if len(ans_set) == 0 and random.random() < 0.5: # Attempt to battle some of the time
                ans_set = la_set.intersection(MC_BATTLE_ACTIONS_SET)
            if len(ans_set) == 0 and random.random() < 0.66: # Attempt to overwork some of the time
                ans_set = la_set.intersection(MC_OVERWORK_ACTIONS_SET)

        # Vagabond
        if len(ans_set) == 0 and opponent_id == PIND_VAGABOND:
            # Vagabond explores if they can
            ans_set = la_set.intersection({VB_EXPLORE})
            if len(ans_set) == 0: # Complete a quest with random reward chosen
                ans_set = la_set.intersection(QUEST_ACTIONS_SET)
            if len(ans_set) == 0: # Strike an open cardboard piece
                ans_set = la_set.intersection(FREE_STRIKE_ACTIONS_SET)
        
        # Alliance
        if len(ans_set) == 0 and opponent_id == PIND_ALLIANCE:
            # always revolts and spreads sympathy if they can at random
            ans_set = la_set.intersection(SPREAD_SYM_ACTIONS_SET | REVOLT_ACTIONS_SET)
            if len(ans_set) == 0: # Organizes if possible
                ans_set = la_set.intersection(ORGANIZE_ACTIONS_SET)
        
        # otherwise, players play a random action possible in this position
        if len(ans_set) == 0:
            ret.put(la, 1/len(la))
        else:
            ans_list = list(ans_set)
            ret.put(ans_list, 1/len(ans_list) )
        
        return ret
        
        

if __name__ == "__main__":
    env = RootEnv("MarquiseMainBase", [MARQUISE_ID])
    env.reset()
    terminated = truncated = False
    total_rewards = np.zeros(N_PLAYERS)
    while not (terminated or truncated):
        legal_actions = env.game.legal_actions()
        logger.debug(f"> Action {env.game.num_actions_played} - Player: {ID_TO_PLAYER[env.game.current_player]}")
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