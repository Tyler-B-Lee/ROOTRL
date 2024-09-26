import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import random
import string
import torch

import logging
# logging.basicConfig(
#     filename='file.log',
#     format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
#     filemode='w',
#     # level=logging.DEBUG
# )
logger = logging.getLogger(__name__)

def sample_action(action_probs):
    action = np.random.choice(len(action_probs), p = action_probs)
    return action


def mask_actions(legal_actions, action_probs):
    masked_action_probs = np.multiply(legal_actions, action_probs)
    masked_action_probs = masked_action_probs / np.sum(masked_action_probs)
    return masked_action_probs


class Agent():
    def __init__(self, name, faction_id, action_space_size, model = None):
        self.name = name
        self.id = self.name + '_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
        self.model = model
        self.points = 0
        self.action_space_size = action_space_size
        self.faction_id = faction_id

        def name_to_id(string):
            if "MC" in string:
                return 0
            if "EY" in string:
                return 1
            if "WA" in string:
                return 2
            if "VB" in string:
                return 3
            raise Exception(f"Unknown target faction name in environment '{string}'")
        
        if 'Hates' in name:
            self.target_id = name_to_id(name)
            self.multiplier = -1
        elif 'Helps' in name:
            self.target_id = name_to_id(name)
            self.multiplier = 1
        else:
            self.target_id = self.faction_id
            self.multiplier = 1

    def print_top_actions(self, action_probs):
        top5_action_idx = np.argsort(-action_probs)[:5]
        top5_actions = action_probs[top5_action_idx]
        logger.debug(f"Top 5 actions: {[str(i) + ': ' + str(round(a,2))[:5] for i,a in zip(top5_action_idx, top5_actions)]}")

    def choose_action(self, env, choose_best_action, mask_invalid_actions):
        if 'rules' in self.name.lower():
            action_probs = np.array(env.rules_move(self.action_space_size))
            value = None
        else:
            with torch.no_grad():
                obs_input = torch.tensor(np.array([env.observation]))
                action_probs = self.model.policy.action_probability(obs_input)[0].numpy()
                value = self.model.policy.predict_values(obs_input)[0].item()
                wr = (value + 1) * 50
                logger.debug(f'Value: {value:.2f} (~{wr:.2f}% Predicted Win Chance)')

        self.print_top_actions(action_probs)
        
        # if mask_invalid_actions:
        #   action_probs = mask_actions(env.legal_actions, action_probs)
        #   logger.debug('Masked ->')
        #   self.print_top_actions(action_probs)
            
        action = np.argmax(action_probs)
        logger.debug(f'Best action {action}')

        if not choose_best_action:
            action = sample_action(action_probs)
            logger.debug(f'Sampled action {action} chosen')

        return action
    