import copy
import random
import numpy as np
from typing import Tuple
from .actionConstants import *
# from actionConstants import *

Recipe = Tuple[int]

import logging

# logging.basicConfig(
#     filename='file.log',
#     format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
#     filemode='w',
#     level=logging.DEBUG
# )
logger = logging.getLogger(__name__)


### Named Constants
SUIT_MOUSE = 0
SUIT_RABBIT = 1
SUIT_FOX = 2
SUIT_BIRD = 3
ID_TO_SUIT = {
    SUIT_MOUSE: "Mouse",
    SUIT_RABBIT: "Rabbit",
    SUIT_FOX: "Fox",
    SUIT_BIRD: "Bird"
}

ITEM_HAMMER = 0
ITEM_SWORD = 1
ITEM_BOOT = 2
ITEM_TEA = 3
ITEM_COINS = 4
ITEM_BAG = 5
ITEM_CROSSBOW = 6
ITEM_TORCH = 7
ITEM_NONE = 8
ID_TO_ITEM = {
    ITEM_HAMMER: "Hammer",
    ITEM_SWORD: "Sword",
    ITEM_BOOT: "Boot",
    ITEM_TEA: "Tea",
    ITEM_COINS: "Coins",
    ITEM_BAG: "Bag",
    ITEM_CROSSBOW: "Crossbow",
    ITEM_TORCH: "Torch",
    ITEM_NONE: "None"
}

# Action IDs
AID_GENERIC_SKIP = 0
AID_CHOOSE_CLEARING = 1
AID_BATTLE_MARQUISE = 13
AID_BATTLE_EYRIE = AID_BATTLE_MARQUISE + 12
AID_BATTLE_ALLIANCE = AID_BATTLE_EYRIE + 12
AID_BATTLE_VAGABOND = AID_BATTLE_ALLIANCE + 12
AID_BUILD1 = AID_BATTLE_VAGABOND + 12
AID_BUILD2 = AID_BUILD1 + 12
AID_BUILD3 = AID_BUILD2 + 12
AID_CRAFT_CARD = AID_BUILD3 + 12
AID_CRAFT_ROYAL_CLAIM = AID_CRAFT_CARD + 41
AID_CRAFT_RC_MAPPING = {
    0:  (4,0,0),
    1:  (0,4,0),
    2:  (0,0,4),
    3:  (3,1,0),
    4:  (3,0,1),
    5:  (1,3,0),
    6:  (1,0,3),
    7:  (0,3,1),
    8:  (0,1,3),
    9:  (2,2,0),
    10: (2,0,2),
    11: (0,2,2),
    12: (2,1,1),
    13: (1,2,1),
    14: (1,1,2),
}

N_PLAYERS = 4
TURN_MEMORY = 2
WIN_SCALAR = 1
POINT_WIN_REWARD = 1
DOM_WIN_REWARD = 1
# WIN_SCALAR = 0.011
# POINT_WIN_REWARD = 50
# DOM_WIN_REWARD = 60
GAME_SCALAR = 0.02
MAX_ACTIONS = 1250

PIND_MARQUISE = 0
PIND_EYRIE = 1
PIND_ALLIANCE = 2
PIND_VAGABOND = 3
ID_TO_PLAYER = {
    PIND_MARQUISE: "Marquise de Cat",
    PIND_EYRIE: "Eyrie Dynasties",
    PIND_ALLIANCE: "Woodland Alliance",
    PIND_VAGABOND: "Vagabond",
    -1: "None"
}

# MARQUISE
BIND_SAWMILL = 0
BIND_WORKSHOP = 1
BIND_RECRUITER = 2
TIND_KEEP = 0
TIND_WOOD = 1
ID_TO_MBUILD = {
    BIND_SAWMILL: "Sawmill",
    BIND_WORKSHOP: "Workshop",
    BIND_RECRUITER: "Recruiter"
}
ID_TO_MTOKEN = {
    TIND_KEEP: "THE KEEP",
    TIND_WOOD: "Wood"
}

AID_SPEND_BIRD = AID_CRAFT_ROYAL_CLAIM + 15
AID_RECRUIT = AID_SPEND_BIRD + 10
AID_ORDER_KEEP = AID_RECRUIT + 1
AID_ORDER_WOOD = AID_ORDER_KEEP + 1
AID_ORDER_SAWMILL = AID_ORDER_WOOD + 1
AID_ORDER_WORKSHOP = AID_ORDER_SAWMILL + 1
AID_ORDER_RECRUITER = AID_ORDER_WORKSHOP + 1
AID_OVERWORK = AID_ORDER_RECRUITER + 1

AID_DISCARD_CARD = AID_OVERWORK + 504
AID_MOVE = AID_DISCARD_CARD + 42

AID_AMBUSH_MOUSE = AID_MOVE + 3600
AID_AMBUSH_RABBIT = AID_AMBUSH_MOUSE + 1
AID_AMBUSH_FOX = AID_AMBUSH_RABBIT + 1
AID_AMBUSH_BIRD = AID_AMBUSH_FOX + 1
AID_AMBUSH_NONE = AID_AMBUSH_BIRD + 1

AID_EFFECTS_NONE = AID_AMBUSH_NONE + 1
AID_EFFECTS_ARMORERS = AID_EFFECTS_NONE + 1
AID_EFFECTS_BRUTTACT = AID_EFFECTS_ARMORERS + 1
AID_EFFECTS_SAPPERS = AID_EFFECTS_BRUTTACT + 1
AID_EFFECTS_ARM_BT = AID_EFFECTS_SAPPERS + 1
AID_EFFECTS_ARMSAP = AID_EFFECTS_ARM_BT + 1

AID_CARD_BBB = AID_EFFECTS_ARMSAP + 1
AID_CARD_ROYAL_CLAIM = AID_CARD_BBB + N_PLAYERS
AID_CARD_STAND_DELIVER = AID_CARD_ROYAL_CLAIM + 1
AID_CARD_CODEBREAKERS = AID_CARD_STAND_DELIVER + N_PLAYERS
AID_CARD_TAX_COLLECTOR = AID_CARD_CODEBREAKERS + N_PLAYERS

AID_ACTIVATE_DOM = AID_CARD_TAX_COLLECTOR + 12
AID_TAKE_DOM = AID_ACTIVATE_DOM + 4

# EYRIE
BIND_ROOST = 0
LEADER_BUILDER = 0
LEADER_CHARISMATIC = 1
LEADER_COMMANDER = 2
LEADER_DESPOT = 3
DECREE_RECRUIT = 0
DECREE_MOVE = 1
DECREE_BATTLE = 2
DECREE_BUILD = 3
ID_TO_LEADER = {
    LEADER_BUILDER: "Builder",
    LEADER_CHARISMATIC: "Charismatic",
    LEADER_COMMANDER: "Commander",
    LEADER_DESPOT: "Despot",
    None: "None"
}
ID_TO_DECREE = {
    DECREE_RECRUIT: "Recruit",
    DECREE_MOVE: "Move",
    DECREE_BATTLE: "Battle",
    DECREE_BUILD: "Build"
}

AID_CHOOSE_LEADER = AID_TAKE_DOM + 168
AID_DECREE_RECRUIT = AID_CHOOSE_LEADER + 4
AID_DECREE_MOVE = AID_DECREE_RECRUIT + 42
AID_DECREE_BATTLE = AID_DECREE_MOVE + 42
AID_DECREE_BUILD = AID_DECREE_BATTLE + 42

# ALLIANCE
BIND_MOUSE_BASE = 0
BIND_RABBIT_BASE = 1
BIND_FOX_BASE = 2
TIND_SYMPATHY = 0
ID_TO_ABUILD = {
    BIND_MOUSE_BASE: "Mouse Base",
    BIND_RABBIT_BASE: "Rabbit Base",
    BIND_FOX_BASE: "Fox Base"
}
ID_TO_ATOKEN = {
    TIND_SYMPATHY: "Sympathy",
}

AID_REVOLT = AID_DECREE_BUILD + 42
AID_SPREAD_SYMPATHY = AID_REVOLT + 12
AID_SPEND_SUPPORTER = AID_SPREAD_SYMPATHY + 12
AID_MOBILIZE = AID_SPEND_SUPPORTER + 42
AID_TRAIN = AID_MOBILIZE + 42
AID_RECRUIT_ALLIANCE = AID_TRAIN + 42
AID_ORGANIZE = AID_RECRUIT_ALLIANCE + 12

AID_ORDER_SYMPATHY = AID_ORGANIZE + 12
AID_ORDER_BASE_MOUSE = AID_ORDER_SYMPATHY + 1
AID_ORDER_BASE_RABBIT = AID_ORDER_BASE_MOUSE + 1
AID_ORDER_BASE_FOX = AID_ORDER_BASE_RABBIT + 1

# VAGABOND
CHAR_THIEF = 0
CHAR_TINKER = 1
CHAR_RANGER = 2

AID_CHOOSE_VB_CLASS = AID_ORDER_BASE_FOX + 1
AID_STARTING_FOREST = AID_CHOOSE_VB_CLASS + 3

AID_REFRESH_UNDAM = AID_STARTING_FOREST + 7
AID_REFRESH_DAM = AID_REFRESH_UNDAM + 8
AID_VB_MOVE = AID_REFRESH_DAM + 8
AID_EXPLORE = AID_VB_MOVE + 361

AID_START_AIDING = AID_EXPLORE + 1
AID_CHOOSE_AID_ITEMS = AID_START_AIDING + 126

AID_COMPLETE_QUEST = AID_CHOOSE_AID_ITEMS + 64
AID_STRIKE = AID_COMPLETE_QUEST + 30
AID_REPAIR_UNEXH = AID_STRIKE + 13
AID_REPAIR_EXH = AID_REPAIR_UNEXH + 8
AID_THIEF_ABILITY = AID_REPAIR_EXH + 8
AID_TINKER_ABILITY = AID_THIEF_ABILITY + 3
AID_RANGER_ABILITY = AID_TINKER_ABILITY + 42
AID_DISCARD_ITEM = AID_RANGER_ABILITY + 1
AID_DAMAGE_UNEXH = AID_DISCARD_ITEM + 32
AID_DAMAGE_EXH = AID_DAMAGE_UNEXH + 8
AID_ACTIVATE_COALITION = AID_DAMAGE_EXH + 8

AID_NO_ALLY_MOVE = AID_ACTIVATE_COALITION + 12
AID_MARQUISE_ALLY_MOVE = AID_NO_ALLY_MOVE + 1
AID_EYRIE_ALLY_MOVE = AID_MARQUISE_ALLY_MOVE + 25
AID_ALLIANCE_ALLY_MOVE = AID_EYRIE_ALLY_MOVE + 20
AID_BATTLE_WITH_ALLY = AID_ALLIANCE_ALLY_MOVE + 10
AID_BATTLE_ALLY_HITS = AID_BATTLE_WITH_ALLY + 4


class TurnLog():
    """
    Keeps track of the current turn for the agents to be
    able to observe a sort of "history" of the current game. This info
    will be changed slightly over a turn, and then compiled into a full
    history object to be saved at the end of a turn.
    """
    def __init__(self) -> None:
        self.reset_current_turn()
    
    def reset_current_turn(self):
        self.current_turn = {
            'dp_reset': False,
            'marq_supp_additions': np.zeros((42,3)),
            'eyrie_supp_additions': np.zeros((42,3)),
            'alliance_supp_payments': np.zeros((42,3)),
            'vagabond_supp_payments': np.zeros((42,3))
        }
        for i in range(N_PLAYERS):
            self.current_turn[i] = {
                'hand_size_change': 0,
                'warrior_supply_change': 0,
                'point_change': 0,
                'cards_lost': np.zeros((42,3)),
                'cards_gained': np.zeros((42,3)),
                'persistent_used': np.zeros((11,N_PLAYERS)),
                'cards_crafted': np.zeros(38)
            }
        for i in range(12):
            clr = {'battles': np.zeros((N_PLAYERS,N_PLAYERS))}
            for j in range(N_PLAYERS):
                clr[j] = {
                    'warrior_change': 0,
                    'buildings_change': np.zeros(3),
                    'tokens_change': np.zeros(2)
                }
            self.current_turn[f'c{i}'] = clr

    def get_array(self,priv_player_id:int):
        """
        Get an array of fixed length contining all of the current
        info stored in this object. Meant for giving information
        to the player whose turn it currently is.
        """
        t = self.current_turn
        ret = np.append(np.array([int(t['dp_reset'])]), t['alliance_supp_payments'])
        for player_i in range(N_PLAYERS):
            p_info = t[player_i]
            max_warriors = [25,20,10,1][player_i]
            foo = np.zeros(3)
            foo[0] = p_info['hand_size_change'] / 8
            foo[1] = p_info['warrior_supply_change'] / max_warriors
            foo[2] = min(p_info['point_change'] / 15, 1)
            for att in ('cards_lost','cards_gained','persistent_used','cards_crafted'):
                foo = np.append(foo,p_info[att])
            ret = np.append(ret,foo)            
        # save info stored for each clearing
        for clearing_i in range(12):
            c_info = t[f'c{clearing_i}']
            foo = c_info['battles']
            for player_i in range(N_PLAYERS):
                max_warriors = [25,20,10,1][player_i]
                foo = np.append(foo,c_info[player_i]['warrior_change'] / max_warriors)
                foo = np.append(foo,c_info[player_i]['buildings_change'] / 3)
                bar = c_info[player_i]['tokens_change'].copy()
                if player_i == PIND_MARQUISE:
                    bar[TIND_WOOD] /= 8
                foo = np.append(foo,bar)
            ret = np.append(ret,foo)
        # save this turn in the history
        if priv_player_id == PIND_MARQUISE:
            foo = np.append(t['marq_supp_additions'], np.full((42,3),-1))
        elif priv_player_id == PIND_EYRIE:
            foo = np.append(np.full((42,3),-1), t['eyrie_supp_additions'])
        else:
            foo = np.append(t['marq_supp_additions'], t['eyrie_supp_additions'])

        ret = np.append(ret,foo)
        # logger.debug(f"TurnLog.get_array for id {priv_player_id}: {len(ret)}")

        return ret
    
    def record_battle(self,clr:int,attacker:int,defender:int):
        self.current_turn[f'c{clr}']['battles'][attacker][defender] = 1
    
    def change_clr_warriors(self,clr:int,fac:int,amount:int):
        self.current_turn[f'c{clr}'][fac]['warrior_change'] += amount
    
    def change_clr_building(self,clr:int,fac:int,bld:int,amount:int):
        self.current_turn[f'c{clr}'][fac]['buildings_change'][bld] += amount
    
    def change_clr_tokens(self,clr:int,fac:int,tok:int,amount:int):
        self.current_turn[f'c{clr}'][fac]['tokens_change'][tok] += amount

    def change_plr_hand_size(self,fac:int,amount:int):
        self.current_turn[fac]['hand_size_change'] += amount

    def change_plr_warrior_supply(self,fac:int,amount:int):
        self.current_turn[fac]['warrior_supply_change'] += amount
    
    def change_plr_points(self,fac:int,amount:int):
        self.current_turn[fac]['point_change'] += amount

    def change_plr_cards_lost(self,fac:int,card_id:int):
        a = self.current_turn[fac]['cards_lost']
        if a[card_id][0] == 1:
            a[card_id][0] = 0
            a[card_id][1] = 1
        elif a[card_id][1] == 1:
            a[card_id][1] = 0
            a[card_id][2] = 1
        else:
            a[card_id][0] = 1
    def change_plr_cards_gained(self,fac:int,card_id:int):
        a = self.current_turn[fac]['cards_gained']
        if a[card_id][0] == 1:
            a[card_id][0] = 0
            a[card_id][1] = 1
        elif a[card_id][1] == 1:
            a[card_id][1] = 0
            a[card_id][2] = 1
        else:
            a[card_id][0] = 1
    def change_alliance_payments(self,card_id:int):
        a = self.current_turn['alliance_supp_payments']
        if a[card_id][0] == 1:
            a[card_id][0] = 0
            a[card_id][1] = 1
        elif a[card_id][1] == 1:
            a[card_id][1] = 0
            a[card_id][2] = 1
        else:
            a[card_id][0] = 1
    def change_alliance_supp_addition(self,paying_fac:int,card_id:int):
        if paying_fac == PIND_MARQUISE:
            a = self.current_turn['marq_supp_additions']
        else:
            a = self.current_turn['eyrie_supp_additions']
        if a[card_id][0] == 1:
            a[card_id][0] = 0
            a[card_id][1] = 1
        elif a[card_id][1] == 1:
            a[card_id][1] = 0
            a[card_id][2] = 1
        else:
            a[card_id][0] = 1
    def change_plr_cards_crafted(self,fac:int,card_id:int):
        a = self.current_turn[fac]['cards_crafted']
        a[card_id] = 1
    
    def change_plr_pers_used(self,fac:int,pers_id:int,target:int = -1):
        if target < 0:
            self.current_turn[fac]['persistent_used'][pers_id] = 1
        else:
            self.current_turn[fac]['persistent_used'][pers_id][target] = 1
    
    def discard_pile_was_reset(self):
        self.current_turn['dp_reset'] = True


class Clearing:
    def __init__(self,id:int,suit:int,num_building_slots:int,num_ruins:int,opposite_corner_id:int,adj_clearing_ids:set,adj_forest_ids:set) -> None:
        self.id = id
        self.suit = suit
        self.num_building_slots = num_building_slots
        self.num_ruins = num_ruins
        self.opposite_corner_id = opposite_corner_id
        self.vagabond_present = 0
        self.warriors = {i:0 for i in range(N_PLAYERS)}
        self.tokens = {i:[] for i in range(N_PLAYERS)}
        self.buildings = {i:[] for i in range(N_PLAYERS)}
        self.adjacent_clearing_ids = adj_clearing_ids
        self.adjacent_forest_ids = adj_forest_ids
    
    def __str__(self) -> str:
        ret = f"Clearing {self.id} ({ID_TO_SUIT[self.suit]}) - Ruler: {ID_TO_PLAYER[self.get_ruler()]}"
        # ret += f"\n{self.warriors[PIND_MARQUISE]} Marquise Warriors"
        ret += f"\n{self.warriors[PIND_EYRIE]} Eyrie Warriors"
        ret += f"\n{self.warriors[PIND_ALLIANCE]} Alliance Warriors"
        ret += f"\nAdjacent Clearings: {[self.adjacent_clearing_ids]}"
        ret += f"\n{self.get_num_empty_slots()} Empty Building Spots"

        foo = []
        # for bid in self.buildings[PIND_MARQUISE]:
        #     foo.append(ID_TO_MBUILD[bid])
        for bid in self.buildings[PIND_ALLIANCE]:
            foo.append(ID_TO_ABUILD[bid])
        if len(self.buildings[PIND_EYRIE]) > 0:
            foo.append("Roost")
        if len(foo) > 0:
            ret += "\nBuildings: " + " ".join(foo)
        foo = []
        # for tid in self.tokens[PIND_MARQUISE]:
        #     foo.append(ID_TO_MTOKEN[tid])
        for tid in self.tokens[PIND_ALLIANCE]:
            foo.append(ID_TO_ATOKEN[tid])
        if len(foo) > 0:
            ret += "\nTokens: " + " ".join(foo)
        return ret + "\n"
    
    def get_obs_array(self):
        "Returns an array describing the current state of this clearing."
        # since this information is very important, I'll try to magnify the
        # size of each info piece and scale them more drastically for
        # unusually high amounts of a thing in one clearing
        ret = np.zeros(3)
        # ret[0] = min(self.get_num_warriors(PIND_MARQUISE) / 16, 1)
        ret[0] = self.get_num_warriors(PIND_MARQUISE) / 25
        # ret[1] = min(self.get_num_tokens(PIND_MARQUISE,TIND_WOOD) / 4, 1)
        ret[1] = self.get_num_tokens(PIND_MARQUISE,TIND_WOOD) / 8
        if self.get_num_tokens(PIND_MARQUISE,TIND_KEEP) > 0:
            ret[2] = 1

        foo = np.zeros(3)
        foo[0] = self.get_num_buildings(PIND_MARQUISE,BIND_SAWMILL) / 3
        foo[1] = self.get_num_buildings(PIND_MARQUISE,BIND_WORKSHOP) / 3
        foo[2] = self.get_num_buildings(PIND_MARQUISE,BIND_RECRUITER) / 3
        ret = np.append(ret,foo)

        foo = np.zeros(2)
        # foo[0] = min(self.get_num_warriors(PIND_EYRIE) / 12, 1)
        foo[0] = self.get_num_warriors(PIND_EYRIE) / 20
        if self.get_num_buildings(PIND_EYRIE,BIND_ROOST) > 0:
            foo[1] = 1
        ret = np.append(ret,foo)

        foo = np.zeros(2)
        # foo[0] = min(self.get_num_warriors(PIND_ALLIANCE) / 5, 1)
        foo[0] = self.get_num_warriors(PIND_ALLIANCE) / 10
        if self.get_num_tokens(PIND_ALLIANCE,TIND_SYMPATHY) > 0:
            foo[1] = 1
        ret = np.append(ret,foo)

        foo = np.zeros(9)
        if self.get_num_buildings(PIND_ALLIANCE,BIND_MOUSE_BASE) > 0:
            foo[0] = 1
        elif self.get_num_buildings(PIND_ALLIANCE,BIND_RABBIT_BASE) > 0:
            foo[1] = 1
        elif self.get_num_buildings(PIND_ALLIANCE,BIND_FOX_BASE) > 0:
            foo[2] = 1

        foo[3] = self.vagabond_present
        if self.num_ruins > 0:
            foo[4] = 1
        foo[5] = self.get_num_empty_slots() / 3

        ruler = self.get_ruler()
        if ruler != -1:
            foo[ruler + 6] = 1

        return np.append(ret,foo)
    
    def get_num_empty_slots(self) -> int:
        "Returns the number of empty slots available to build in for the clearing."
        return self.num_building_slots - sum(len(x) for x in self.buildings.values())
    
    def get_ruling_power(self, faction_index:int) -> int:
        "Returns the total ruling power of the given faction: # of Warriors + # of buildings."
        if faction_index == PIND_VAGABOND:
            return 0
        return self.warriors[faction_index] + len(self.buildings[faction_index])
    
    def has_presence(self, faction_index:int) -> bool:
        """
        Returns True if any pieces of the given faction,
        including tokens, warriors, or buildings, are located in
        this clearing, and False otherwise.

        This means that that the given faction could be attacked.
        """
        if faction_index == PIND_VAGABOND:
            return bool(self.vagabond_present)
        return bool(self.get_ruling_power(faction_index) or len(self.tokens[faction_index]))

    def get_ruler(self) -> int:
        """
        Returns the player index of the player that rules the current clearing.
        If nobody returns the current clearing, returns -1.
        """
        num_marquise = self.get_ruling_power(PIND_MARQUISE)
        num_eyrie = self.get_ruling_power(PIND_EYRIE)
        num_alliance = self.get_ruling_power(PIND_ALLIANCE)

        if (num_alliance == 0) and (num_eyrie == 0) and (num_marquise == 0):
            return -1
        # a faction has some piece in it (max rule power > 0)
        max_power = max(num_marquise,num_eyrie,num_alliance)
        # the Eyrie rule tied clearings they have a piece in with "Lords of the Forest"
        if num_eyrie == max_power:
            return PIND_EYRIE
        # The Eyrie have < max rule power, so it's either the alliance or marquise
        if num_marquise > num_alliance:
            return PIND_MARQUISE
        if num_alliance > num_marquise:
            return PIND_ALLIANCE
        # there is a tie between marquise/alliance, so nobody rules
        return -1
    
    def is_ruler(self,faction_index:int) -> bool:
        "Returns True if the given faction IS the ruler of this clearing, False otherwise."
        return self.get_ruler() == faction_index
    
    ### BUILDING METHODS
    def get_num_buildings(self,faction_index:int,building_index:int = -1) -> int:
        """
        Returns the number of buildings of the given type for the given faction in the clearing.

        If no building index is specified, returns the total number of buildings
        for the given faction, regardless of their type.
        """
        return len(self.buildings[faction_index]) if (building_index == -1) else self.buildings[faction_index].count(building_index)
    
    def place_building(self,faction_index:int,building_index:int) -> None:
        "Place the given building in this clearing. Assumes the move is legal, performing no checks."
        # logger.debug(f"\t\tBuilding {building_index} added for {ID_TO_PLAYER[faction_index]} in clearing {self.id}")
        self.buildings[faction_index].append(building_index)
    
    def remove_building(self,faction_index:int,building_index:int) -> None:
        "Removes the given building in this clearing, assuming it exists. Does not handle points."
        # logger.debug(f"\t\tBuilding {building_index} removed for {ID_TO_PLAYER[faction_index]} in clearing {self.id}")
        self.buildings[faction_index].remove(building_index)

    ### TOKEN METHODS
    def get_num_tokens(self,faction_index:int,token_index:int = -1) -> int:
        """Returns the number of tokens of the given type for the given faction in the clearing.

        If no token index is specified, returns the total number of tokens
        for the given faction, regardless of their type.
        """
        return len(self.tokens[faction_index]) if (token_index == -1) else self.tokens[faction_index].count(token_index)

    def place_token(self,faction_index:int,token_index:int) -> None:
        "Place the given token in this clearing. Assumes the move is legal, performing no checks."
        # logger.debug(f"\t\tToken {token_index} added for {ID_TO_PLAYER[faction_index]} in clearing {self.id}")
        self.tokens[faction_index].append(token_index)
    
    def remove_token(self,faction_index:int,token_index:int) -> None:
        "Removes the given token in this clearing, assuming it exists. Does not handle points."
        # logger.debug(f"\t\tToken {token_index} removed for {ID_TO_PLAYER[faction_index]} in clearing {self.id}")
        self.tokens[faction_index].remove(token_index)
    
    ### WARRIOR METHODS
    def get_num_warriors(self,faction_index:int) -> int:
        "Returns the number of warriors of the given faction in the clearing."
        return self.warriors[faction_index]

    def change_num_warriors(self,faction_index:int,change:int) -> None:
        """
        Adds the specified number of warriors of the specified faction to the clearing.
        Use a negative number to remove warriors. Does NOT change faction supply counts.
        """
        # logger.debug(f"\t\tWarriors changed by {change} for {ID_TO_PLAYER[faction_index]} in clearing {self.id}")
        self.warriors[faction_index] += change
    

    def can_start_battle(self,attacker_index:int,defender_index:int) -> bool:
        """
        Returns True if one faction can attack the other as directed
        in this particular clearing, and False otherwise.
        """
        if attacker_index == PIND_VAGABOND:
            return bool( self.vagabond_present and self.has_presence(defender_index) )
        return bool( self.warriors[attacker_index] and self.has_presence(defender_index) )

    def can_place(self,faction_index:int) -> bool:
        """
        Returns True if the faction can place a piece in this clearing. This is mainly affected
        by the Marquise's Keep, which blocks other Factions from placing pieces in its clearing.
        """
        return (TIND_KEEP not in self.tokens[PIND_MARQUISE]) or (faction_index == PIND_MARQUISE)
    
    def is_adjacent_to(self,other_index:int):
        "Returns True only if this clearing is connected to the clearing with the other index given."
        return (other_index in self.adjacent_clearing_ids)
    
    def favor_helper(self,safe_faction_index:int):
        """
        Helps with the process of resolving a favor card. Removes any
        and all warriors, buildings, and tokens of all factions that are
        not the safe faction given.

        Returns a 2-tuple: the total number of points that
        should be scored from removing tokens/buildings in this clearing,
        and the number of Marquise warriors removed (for Field Hospitals).
        """
        ans = marqwar = 0
        for faction_i in {j for j in range(N_PLAYERS) if j != safe_faction_index}:
            ans += self.get_num_buildings(faction_i) + self.get_num_tokens(faction_i)
            if faction_i == PIND_MARQUISE:
                marqwar = self.get_num_warriors(faction_i)
                
        return ans,marqwar
    
    def is_sympathetic(self):
        "Returns True only if this clearing has a sympathy token in it."
        return self.get_num_tokens(PIND_ALLIANCE) > 0
    
    def has_martial_law(self):
        "Returns True only if there are 3+ warriors of a single player that is NOT the Alliance."
        for pid in range(N_PLAYERS):
            if pid == PIND_ALLIANCE or pid == PIND_VAGABOND:
                continue
            if self.get_num_warriors(pid) >= 3:
                return True
        return False


class Forest:
    def __init__(self,id:int,adj_clearing_ids:set,adj_forest_ids:set) -> None:
        self.id = id
        self.adjacent_clearing_ids = adj_clearing_ids
        self.adjacent_forest_ids = adj_forest_ids
        self.vagabond_present = 0


class Board:
    def __init__(self, board_comp:tuple) -> None:
        self.board_comp = board_comp
        self.reset()
    
    def __str__(self) -> str:
        s = "Current Board:\n"
        for c in self.clearings:
            s += str(c) + "\n"
        return s
    
    def get_obs_array(self):
        "Returns an array of the current board's state."
        ret = np.asarray([c.get_obs_array() for c in self.clearings])

        foo = np.zeros(7)
        for i in range(7):
            foo[i] = self.forests[i].vagabond_present

        return np.append(ret,foo)
    
    def randomize_suits(self):
        "Randomizes the suits of each clearing in self.clearings, NOT the board_comp given on creation."
        logger.debug(">--- RANDOMIZING MAP SUITS ---<")
        suits = [SUIT_MOUSE] * 4 + [SUIT_RABBIT] * 4 + [SUIT_FOX] * 4
        for c in self.clearings:
            s = random.choice(suits)
            suits.remove(s)
            c.suit = s
            logger.debug(f"\tClearing {c.id} suit set to {ID_TO_SUIT[s]}")

    def reset(self):
        "Resets the map to the cleared starting state."
        self.clearings: list[Clearing] = copy.deepcopy(self.board_comp[0])
        self.forests: list[Forest] = copy.deepcopy(self.board_comp[1])
        # self.randomize_suits()
        
    def get_total_building_counts(self,faction_index:int,building_index:int = -1):
        """
        Finds the number of buildings of the given faction in each clearing.
        If no building type is specified, returns the total number
        of buildings belonging to the given faction in each clearing.

        Returns a list of integers, with each index matching the corresponding clearing index.
        """
        return [x.get_num_buildings(faction_index,building_index) for x in self.clearings]
    
    def get_clearing_building_counts(self,faction_index:int,clearing_index:int,building_index:int = -1):
        """
        Returns the number of buildings of the given faction of the given type in the GIVEN clearing.
        If no building type is specified, returns the total number
        of buildings belonging to the given faction in the ONE clearing.
        """
        return self.clearings[clearing_index].get_num_buildings(faction_index,building_index)

    def get_total_token_counts(self,faction_index:int,token_index:int = -1):
        """
        Finds the number of tokens of the given faction in each clearing.
        If no token type is specified, returns the total number
        of tokens belonging to the given faction in each clearing.

        Returns a list of integers, with each index matching the corresponding clearing index.
        """
        return [x.get_num_tokens(faction_index,token_index) for x in self.clearings]

    def get_clearing_token_counts(self,faction_index:int,clearing_index:int,token_index:int = -1):
        """
        Returns the number of buildings of the given faction of the given type in the GIVEN clearing.
        If no building type is specified, returns the total number
        of buildings belonging to the given faction in the ONE clearing.
        """
        return self.clearings[clearing_index].get_num_tokens(faction_index,token_index)
    
    def get_empty_building_slot_counts(self):
        """
        Finds the number of empty building slots in each clearing on
        the board. Returns a list of integers, one for each corresponding clearing.
        """
        return [x.get_num_empty_slots() for x in self.clearings]

    def get_rulers(self):
        """
        Finds the current ruling faction of each clearing. Returns a list of integers,
        each corresponding to the faction ID of the ruler in the given clearing, or
        a -1 if there is no ruler of the clearing.
        """
        return [x.get_ruler() for x in self.clearings]
    
    def get_possible_battles(self,attacker_index:int,defender_index:int):
        """
        Returns a list of all of the places where the given faction can start
        a battle with the target faction.

        Returns a list of booleans, one for each clearing.
        """
        return [x.can_start_battle(attacker_index,defender_index) for x in self.clearings]
    
    def get_num_warriors(self,faction_index:int):
        "Returns a list of the number of warriors of the given faction in each clearing."
        return [x.get_num_warriors(faction_index) for x in self.clearings]
    
    def get_crafting_power(self,faction_index:int):
        """
        Returns the crafting power of the given faction. This is
        represented by a list of integers:
        - [Mouse Power / Rabbit Power / Fox Power]
        """
        power = [0,0,0]
        if faction_index == PIND_MARQUISE:
            item_counts = self.get_total_building_counts(PIND_MARQUISE,BIND_WORKSHOP)
        elif faction_index == PIND_EYRIE:
            item_counts = self.get_total_building_counts(PIND_EYRIE,BIND_ROOST)
        else:
            item_counts = self.get_total_token_counts(PIND_ALLIANCE,TIND_SYMPATHY)

        for clearing_i,num in enumerate(item_counts):
            if num:
                c = self.clearings[clearing_i]
                power[c.suit] += num
        return power
    
    def get_legal_move_actions(self,faction_index:int,starting_suits:set):
        """
        Finds every possible distinct legal move that the given faction can currently
        make on the board. The start clearing of each move must have its suit in starting_suits.

        Returns a list of integers: the AID's of each distinct move possible.
        """
        ans = []
        for i,start_clearing in enumerate(self.clearings):
            n_warriors = start_clearing.get_num_warriors(faction_index)
            if (n_warriors == 0) or (start_clearing.suit not in starting_suits):
                continue
            valid_dest_ids = []
            if start_clearing.is_ruler(faction_index):
                valid_dest_ids += list(start_clearing.adjacent_clearing_ids)
            else:
                for dest_id in start_clearing.adjacent_clearing_ids:
                    if self.clearings[dest_id].is_ruler(faction_index):
                        valid_dest_ids.append(dest_id)
            
            ans += [(i*12 + j + GEN_MOVE_CLEARINGS) for j in valid_dest_ids]
        return ans
    
    def move_warriors(self,faction_index:int,amount:int,start_index:int,end_index:int):
        """
        Subtracts warriors of a faction from one clearing, and adds them to another.
        Performs no other checks / assumes the move will be legal.
        """
        logger.debug(f"\t\tMoving {amount} warriors of {ID_TO_PLAYER[faction_index]} from {start_index} to {end_index}")
        start_c,end_c = self.clearings[start_index],self.clearings[end_index]
        
        start_c.change_num_warriors(faction_index,-amount)
        end_c.change_num_warriors(faction_index,amount)

    def move_vagabond(self,start_location:int,end_location:int):
        "Moves the vagabond pawn from one clearing/forest to another."
        if start_location <= 11:
            self.clearings[start_location].vagabond_present = 0
        else:
            self.forests[start_location - 12].vagabond_present = 0
        if end_location <= 11:
            self.clearings[end_location].vagabond_present = 1
        else:
            self.forests[end_location - 12].vagabond_present = 1

    def place_warriors(self,faction_index:int,amount:int,clearing_index:int):
        "Adds the given number of warriors of the faction to the clearing, assuming it is legal to do so."
        logger.debug(f"\t\tPlacing {amount} {ID_TO_PLAYER[faction_index]} warriors in clearing {clearing_index}")
        self.clearings[clearing_index].change_num_warriors(faction_index,amount)
    
    def place_building(self,faction_index:int,building_index:int,clearing_index:int):
        "Adds one of the given building type to the given clearing, assuming it's legal to do so."
        if faction_index == PIND_MARQUISE:
            build = ID_TO_MBUILD[building_index]
        elif faction_index == PIND_EYRIE:
            build = "Roost"
        else:
            build = ID_TO_ABUILD[building_index]
        logger.debug(f"\t\t{build} placed in clearing {clearing_index}")
        self.clearings[clearing_index].place_building(faction_index,building_index)

    def place_token(self,faction_index:int,token_index:int,clearing_index:int):
        "Adds one of the given building type to the given clearing, assuming it's legal to do so."
        if faction_index == PIND_MARQUISE:
            tok = ID_TO_MTOKEN[token_index]
        else:
            tok = ID_TO_ATOKEN[token_index]
        logger.debug(f"\t\t{tok} token placed in clearing {clearing_index}")
        self.clearings[clearing_index].place_token(faction_index,token_index)
    
    def resolve_favor(self,safe_faction_index:int,clearing_indexes:list):
        """
        Resolves the effects of the safe faction crafting a favor card. The
        Effect is carried out in each clearing in the given list of indexes.

        Returns a 2-tuple: the total number of points scored by the
        activating player for removing buildings and tokens, and
        the list of Marquise warriors removed per clearing / the suit (for Field Hospitals)
        """
        ans = 0
        field_hospitals = []
        for i in clearing_indexes:
            pts,marqwars = self.clearings[i].favor_helper(safe_faction_index)
            ans += pts
            if marqwars > 0:
                field_hospitals.append( (marqwars,self.clearings[i].suit) )
        return ans,field_hospitals
    
    def get_wood_available(self):
        """
        For the Marquise, returns a list of integers, one for each clearing.
        The integer for clearing i is the amount of wood tokens available
        to use to Build in clearing i, either from that clearing or using
        any number of connected clearings ruled by the cats.

        If the Marquise do NOT rule clearing i, the amount of wood
        in that clearing is given as -1.
        """
        # logger.debug("GWA Function")
        ans = [-1 for i in range(12)]
        # find out which clearings the Marquise rule - which is where wood actually counts
        foo = [(x == PIND_MARQUISE) for x in self.get_rulers()]
        clearings_left_to_assign = {i for i,ruled in enumerate(foo) if ruled}

        # make groups of connected clearings ruled by the Marquise
        while clearings_left_to_assign:
            i = clearings_left_to_assign.pop()
            # logger.debug(f"\tStarting new group at clearing {i}")
            new_group = {i}
            total = self.clearings[i].get_num_tokens(PIND_MARQUISE,TIND_WOOD)
            c_to_add_to_group = {j for j in self.clearings[i].adjacent_clearing_ids if j in clearings_left_to_assign}
            while c_to_add_to_group:
                j = c_to_add_to_group.pop()
                clearings_left_to_assign.remove(j)
                new_group.add(j)
                total += self.clearings[j].get_num_tokens(PIND_MARQUISE,TIND_WOOD)
                # logger.debug(f"\tAdded clearing {j} to the group. New total: {total}")
                c_to_add_to_group.update( {k for k in self.clearings[j].adjacent_clearing_ids if k in (clearings_left_to_assign - c_to_add_to_group)} )

            for i in new_group:
                ans[i] = total
        # logger.debug(f"GWA Answer: {ans}")
        return ans
    
    def get_wood_to_build_in(self, clearing_index:int):
        """
        For the Marquise, returns a list of integers, one for each clearing.
        Assuming that we are building in the given input clearing index, the
        integer in index i of the returned list is the number of wood tokens
        available to legally take out of clearing i for contributing to the build.

        Thus, a clearing that has 1+ wood tokens, but is NOT connected by rule to
        the building clearing will show up as 0, as it has no usable wood for this build.
        """
        ans = [0 for i in range(12)]
        # logger.debug("GWB Function")
        # it is assumed that the given clearing is ruled by the Marquise (required to build)
        clearings_checked = set()
        clearings_to_check = {clearing_index}
        # logger.debug(f"\tStarting check at clearing {clearing_index}")
        while clearings_to_check:
            i = clearings_to_check.pop()
            clearings_checked.add(i)
            # logger.debug(f"\t  Adding wood in clearing {i}...")
            ans[i] = self.clearings[i].get_num_tokens(PIND_MARQUISE,TIND_WOOD)
            for j in self.clearings[i].adjacent_clearing_ids:
                if (j not in clearings_checked) and (self.clearings[j].get_ruler() == PIND_MARQUISE):
                    # logger.debug(f"\t\tClearing {j} needs to be added...")
                    clearings_to_check.add(j)
        # logger.debug(f"GWB Answer: {ans}")
        return ans
    
    def get_num_sympathetic(self,suit:int):
        "Returns the number of sympathetic clearings currently on the board with the given suit."
        return sum([bool(c.suit == suit and c.is_sympathetic()) for c in self.clearings])
    
    def get_slip_actions(self,vb_location:int):
        """
        Given a location, returns a list of AIDs for all of the valid
        slip actions possible from that location. This is a single, free VB
        movement option that seems to always be valid (ignores rule/relationships).
        """
        ans = []
        if vb_location <= 11: # in clearing
            clearing = self.clearings[vb_location]
            for end_i in clearing.adjacent_clearing_ids:
                ans.append(VB_MOVE + end_i)
            for end_i in clearing.adjacent_forest_ids:
                ans.append(VB_MOVE + (end_i + 12))
        else: # in forest
            forest = self.forests[vb_location - 12]
            for end_i in forest.adjacent_clearing_ids:
                ans.append(VB_MOVE + end_i)
            for end_i in forest.adjacent_forest_ids:
                ans.append(VB_MOVE + (end_i + 12))
        return ans


class Card:
    def __init__(self,id:int,suit:int,name:str,recipe:Recipe,is_ambush:bool,is_dominance:bool,is_persistent:bool,item:int,points:int) -> None:
        self.id = id
        self.suit = suit
        self.name = name
        self.crafting_recipe = recipe
        self.is_ambush = is_ambush
        self.is_dominance = is_dominance
        self.is_persistent = is_persistent
        self.crafting_item = item
        self.points = points
    
    def __str__(self) -> str:
        return f"{self.name} ({ID_TO_SUIT[self.suit]}) (ID {self.id}) ({self.points} Points) (Recipe {self.crafting_recipe})"

class Deck:
    def __init__(self, deck_comp:list):
        self.deck_comp = deck_comp
        self.reset()
    
    def __str__(self) -> str:
        return f" - Deck - {self.size()} Cards\n"
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def reset(self):
        "Remakes the deck from the starting composition and then shuffles it."
        self.cards = []
        for card,amount in self.deck_comp:
            addition = [card for i in range(amount)]
            self.cards += addition
        self.shuffle()

    def draw(self, n):
        """
        Attempts to draw n cards from the deck by popping from the 'cards' list.
        Returns a list of the card objects drawn.

        If the deck runs out, simply returns all of the cards it could draw.
        """
        drawn = []
        for x in range(n):
            try:
                drawn.append(self.cards.pop())
            except:
                pass
        return drawn
    
    def add(self, cards:list):
        "Adds the given cards in the input to the deck and then shuffles the deck."
        for card in cards:
            self.cards.append(card)
        self.shuffle()
                
    def size(self):
        "Returns the current number of cards in the deck."
        return len(self.cards)


class Player:
    def __init__(self,id:int) -> None:
        self.id = id
        self.warrior_storage = 0
        self.buildings = {}
        self.tokens = {}
        self.crafted_items = {i:0 for i in range(7)}
        self.persistent_cards = []
        self.hand = []
    
    def __str__(self) -> str:
        ret = f"""Player ID {self.id}
    Warriors: {self.warrior_storage} - Buildings: {self.buildings} - Tokens: {self.tokens}
    Crafted Items: {self.crafted_items} - Crafted Cards: {[i.name for i in self.persistent_cards]}
    Hand:\n"""
        for card in self.hand:
            ret += f"\t- {str(card)}\n"
        return ret

    def get_num_buildings_on_track(self, building_index:int) -> int:
        "Returns the number of buildings of the given type left on this player's track."
        return self.buildings[building_index]
    
    def get_num_tokens_in_store(self, token_index:int) -> int:
        "Returns the number of tokens of the given type left in this player's store."
        return self.tokens[token_index]

    def change_num_warriors(self, change:int) -> None:
        """
        Changes the number of warriors in this faction's supply by adding 'change'.
        Use a negative number to remove warriors.
        """
        # logger.debug(f"\t\t{ID_TO_PLAYER[self.id]} warrior storage changed by {change}")
        self.warrior_storage += change
    
    def change_num_buildings(self, building_index:int, change:int) -> None:
        """
        Changes the number of buildings of the given type in this faction's supply by adding 'change'.
        Use a negative number to remove buildings of the given type.
        """
        # logger.debug(f"\t\t{ID_TO_PLAYER[self.id]} buildings ID {building_index} changed by {change}")
        self.buildings[building_index] += change
    
    def change_num_tokens(self, token_index:int, change:int) -> None:
        """
        Changes the number of tokens of the given type in this faction's supply by adding 'change'.
        Use a negative number to remove tokens of the given type.
        """
        # logger.debug(f"\t\t{ID_TO_PLAYER[self.id]} tokens ID {token_index} changed by {change}")
        self.tokens[token_index] += change
    
    def change_num_items(self, item_index:int, change:int) -> None:
        """
        Changes the number of items of the given type in this faction's supply by adding 'change'.
        Use a negative number to remove items of the given type.
        """
        # logger.debug(f"\t\t{ID_TO_PLAYER[self.id]} items ID {item_index} changed by {change}")
        self.crafted_items[item_index] += change
    
    def has_suit_in_hand(self, suit_id:int):
        """
        Returns True only if any card in the player's hand is
        either of the exact given suit OR is a BIRD card.
        """
        return any((c.suit in {suit_id, SUIT_BIRD}) for c in self.hand)

    def get_ambush_actions(self, clearing_suit: int):
        ans = set()
        valid_suits = {SUIT_BIRD,clearing_suit}
        for card in self.hand:
            if card.is_ambush and card.suit in valid_suits:
                ans.add(GEN_USE_AMBUSH+4)
                ans.add(card.suit + GEN_USE_AMBUSH)
        return list(ans)
    
    def get_attacker_card_actions(self):
        "Returns a list of all valid attacking AID's this player can do with their current persistent cards."
        ans = set()
        for card in self.persistent_cards:
            if card.id == CID_ARMORERS:
                ans.add(GEN_EFFECTS_NONE)
                ans.add(GEN_EFFECTS_ARMORERS)
            elif card.id == CID_BRUTAL_TACTICS:
                ans.add(GEN_EFFECTS_NONE)
                ans.add(GEN_EFFECTS_BRUTTACT)
        if len(ans) == 3:
            ans.add(GEN_EFFECTS_ARM_BT)
        return list(ans)

    def get_defender_card_actions(self):
        "Returns a list of all valid defending AID's this player can do with their current persistent cards."
        ans = set()
        for card in self.persistent_cards:
            if card.id == CID_ARMORERS:
                ans.add(GEN_EFFECTS_NONE)
                ans.add(GEN_EFFECTS_ARMORERS)
            elif card.id == CID_SAPPERS:
                ans.add(GEN_EFFECTS_NONE)
                ans.add(GEN_EFFECTS_SAPPERS)
        if len(ans) == 3:
            ans.add(GEN_EFFECTS_ARMSAP)
        return list(ans)
    
    def has_card_id_in_hand(self,id:int):
        "Returns True only if this player has a card with the given ID in their hand."
        return any(c.id == id for c in self.hand)
    

class Marquise(Player):
    building_costs = [0,1,2,3,3,4]
    point_tracks = {}
    point_tracks[BIND_SAWMILL] = [0,1,2,3,4,5]
    point_tracks[BIND_WORKSHOP] = [0,2,2,3,4,5]
    point_tracks[BIND_RECRUITER] = [0,1,2,3,3,4]

    def __init__(self, id: int,) -> None:
        super().__init__(id)
        self.warrior_storage = 25
        for i in range(3):
            self.buildings[i] = 6
        self.tokens[TIND_KEEP] = 1
        self.tokens[TIND_WOOD] = 8
        self.keep_clearing_id = -1
    
    def __str__(self) -> str:
        return "--- Marquise de Cat ---\n" + super().__str__() + f"\nKeep placed in clearing {self.keep_clearing_id}"

    def get_obs_array(self):
        foo = np.zeros((3,6))
        for i,a in self.buildings.items():
            if a > 0:
                foo[i][a - 1] = 1
        ret = np.append((self.warrior_storage / 25),foo)
        
        foo = np.zeros(9)
        if self.tokens[TIND_WOOD] > 0:
            foo[self.get_num_tokens_in_store(TIND_WOOD) - 1] = 1
        if self.tokens[TIND_KEEP] > 0:
            foo[8] = 1
        ret = np.append(ret,foo)

        # cards in hand = 0,1,2,3, 4 or more
        hand_size = min(5, len(self.hand)) / 5
        ret = np.append(ret,hand_size)
        
        foo = np.zeros(11)
        foo.put([CID_TO_PERS_INDEX[c.id] for c in self.persistent_cards], 1)
        ret = np.append(ret,foo)
        
        foo = np.zeros(7)
        for i,a in self.crafted_items.items():
            foo[i] = a / 2
        return np.append(ret,foo)

    def get_num_cards_to_draw(self) -> int:
        "Returns the number of cards to draw at the end of the turn (In Evening)."
        recruiters_left = self.get_num_buildings_on_track(BIND_RECRUITER)
        if recruiters_left > 3:
            return 1
        elif recruiters_left > 1:
            return 2
        return 3

    def update_from_building_placed(self, building_index:int) -> tuple:
        """
        Updates the player board as if the given building was placed:
        - Removes 1 building from the corresponding track
        - Wood token amounts should be handled outside when they are
        properly removed from the board

        Returns the number of wood to use and Victory Points scored.
        """
        i = 6 - self.get_num_buildings_on_track(building_index)
        self.change_num_buildings(building_index,-1)

        return Marquise.building_costs[i], Marquise.point_tracks[building_index][i]
    

class Eyrie(Player):
    roost_points = [0,0,1,2,3,4,4,5]
    leader_starting_viziers = { # (Recruit, Move, Battle, Build)
        LEADER_BUILDER:     (1,1,0,0),
        LEADER_CHARISMATIC: (1,0,1,0),
        LEADER_COMMANDER:   (0,1,1,0),
        LEADER_DESPOT:      (0,1,0,1)
    }

    def __init__(self, id: int) -> None:
        super().__init__(id)
        self.warrior_storage = 20
        self.buildings[BIND_ROOST] = 7
        self.available_leaders = [0,1,2,3]
        self.deposed_leaders = []
        self.chosen_leader_index = None
        self.decree = {i:[] for i in range(4)}
        self.viziers = [Card(CID_LOYAL_VIZIER,SUIT_BIRD,"Loyal Vizier",(0,0,0,0),False,False,False,ITEM_NONE,0) for i in range(2)]
    
    def __str__(self) -> str:
        ret = "--- Eyrie Dynasties ---\n" + super().__str__() + f"\nCurrent Leader: {ID_TO_LEADER[self.chosen_leader_index]} - Leaders Available: {[ID_TO_LEADER[i] for i in self.available_leaders]}\n"
        ret += "Decree:\n"
        for i,lst in self.decree.items():
            ret += f"\t{ID_TO_DECREE[i]}: {[(card.name,ID_TO_SUIT[card.suit]) for card in lst]}\n"
        return ret
    
    def get_obs_array(self):
        foo = np.zeros(7)
        if self.buildings[BIND_ROOST] > 0:
            foo[self.buildings[BIND_ROOST] - 1] = 1
        ret = np.append((self.warrior_storage / 20),foo)

        hand_size = min(5, len(self.hand)) / 5
        ret = np.append(ret,hand_size)
        
        foo = np.zeros(11)
        foo.put([CID_TO_PERS_INDEX[c.id] for c in self.persistent_cards], 1)
        ret = np.append(ret,foo)
        
        foo = np.zeros(7)
        for i,a in self.crafted_items.items():
            foo[i] = a / 2
        ret = np.append(ret,foo)

        foo = np.zeros(8)
        foo.put(self.available_leaders, 1)
        if self.chosen_leader_index is not None:
            foo[self.chosen_leader_index + 4] = 1
        ret = np.append(ret,foo)

        foo = np.zeros((4,43)) # how many of each card is in each decree slot
        bar = np.zeros((4,4,5)) # suit counts for each decree item
        for dec_i in range(4):
            for c in self.decree[dec_i]:
                # count one more of this card
                foo[dec_i][c.id] += 1/3
                # count this card's suit
                if sum(bar[dec_i][c.suit]) == 0:
                    bar[dec_i][c.suit][0] = 1
                else:
                    for suit_seen in range(4):
                        if bar[dec_i][c.suit][suit_seen] == 1:
                            bar[dec_i][c.suit][suit_seen] = 0
                            bar[dec_i][c.suit][suit_seen + 1] = 1
                            break
        # logger.debug(f"decree suit counts: {bar}")
        foo = np.append(foo,bar)    
                        
        return np.append(ret,foo)

    def get_num_cards_to_draw(self) -> int:
        "Returns the number of cards to draw at the end of the turn (In Evening)."
        roosts_left = self.get_num_buildings_on_track(BIND_ROOST)
        if roosts_left > 4:
            return 1
        elif roosts_left > 1:
            return 2
        return 3
    
    def get_points_to_score(self) -> int:
        "Returns the number of points that should be scored in the evening phase."
        x = self.get_num_buildings_on_track(BIND_ROOST)
        return self.roost_points[7 - x]

    def place_roost(self) -> None:
        "Removes 1 roost from the track, as if it was placed on the board."
        self.change_num_buildings(BIND_ROOST,-1)
    
    def add_to_decree(self,card_to_add:Card,decree_index:int):
        "Adds the given card object to the decree."
        logger.debug(f"\t\tCard {card_to_add.name} added to decree at {ID_TO_DECREE[decree_index]}")
        self.decree[decree_index].append(card_to_add)
    
    def choose_new_leader(self, leader_index:int) -> None:
        """
        Sets the given leader as the new leader of the Eyries. Assumes that the two
        Loyal Vizier Cards are in the factions list of viziers, and will attempt to
        place them in the corresponding Decree columns for the given leader.
        """
        logger.debug(f"\tNew Leader Chosen: {ID_TO_LEADER[leader_index]}")
        self.chosen_leader_index = leader_index
        self.available_leaders.remove(leader_index)

        # Place the starting viziers for the new leader
        for i, place_vizier in enumerate(Eyrie.leader_starting_viziers[leader_index]):
            if place_vizier:
                self.add_to_decree(self.viziers.pop(),i)
    
    def turmoil_helper(self):
        """
        Resolves several actions as if Turmoil has just occured:
        - Takes out the Loyal Viziers from the Decree
        - Completely empties the decree
        - Removes the current leader and resets the available leaders if needed

        Returns a tuple containing two items:
        - list: contains all of the card objects in the decree that should be discarded
        - int: the number of bird cards that were in the
        decree in total, which is the number of points that the Eyrie should lose
        """
        num_bird_cards = 0
        cards_to_discard = []

        # we want to remove the two viziers, but will
        # simultaneously count the number of bird cards
        # in the entire decree
        for card_list in self.decree.values():
            num_birds_in_this_slot = sum(x.suit == SUIT_BIRD for x in card_list)
            # this slot only matters now if any birds are in it
            if num_birds_in_this_slot:
                num_bird_cards += num_birds_in_this_slot
                for i,card in enumerate(card_list):
                    if card.id == CID_LOYAL_VIZIER:
                        # This is a loyal vizier, which we need to keep for the Eyrie's board
                        # We can safely pop the card because at most one will appear in each of these lists
                        self.viziers.append(card_list.pop(i))
                        break
            # add the non-vizier cards to the list to discard
            cards_to_discard += card_list
        
        # reset the decree
        self.decree = {i:[] for i in range(4)}

        # depose the current leader
        self.deposed_leaders.append(self.chosen_leader_index)
        self.chosen_leader_index = None
        if len(self.deposed_leaders) == 4:
            # reset available leaders if all 4 have been deposed
            self.available_leaders = [0,1,2,3]
            self.deposed_leaders = []

        return cards_to_discard, num_bird_cards
    

class Alliance(Player):
    sympathy_costs = [1,1,1,2,2,2,3,3,3,3,500]
    point_track = [0,1,1,1,2,2,3,4,4,4]

    def __init__(self, id: int,) -> None:
        super().__init__(id)
        self.warrior_storage = 10
        for i in range(3):
            self.buildings[i] = 1
        self.tokens[TIND_SYMPATHY] = 10
        self.supporters = []
        self.num_officers = 0
        self.supporter_suit_counts = [0,0,0]
    
    def __str__(self) -> str:
        return "--- Woodland Alliance ---\n" + super().__str__() + f"\ntodo"

    def get_obs_array(self):
        foo = np.zeros(3)
        for i,a in self.buildings.items():
            if a > 0:
                foo[i] = 1
        ret = np.append((self.warrior_storage / 10),foo)
        
        foo = np.zeros(10)
        if self.tokens[TIND_SYMPATHY] > 0:
            foo[self.get_num_tokens_in_store(TIND_SYMPATHY) - 1] = 1
        ret = np.append(ret,foo)

        foo = np.zeros(6)
        foo[0] = len(self.supporters) / 54

        supp_amount = min(len(self.supporters),5)
        if supp_amount > 0:
            foo[supp_amount] = 1
        ret = np.append(ret,foo)

        foo = np.zeros(10)
        if self.num_officers > 0:
            foo[self.num_officers - 1] = 1
        ret = np.append(ret,foo)

        hand_size = min(5, len(self.hand)) / 5
        ret = np.append(ret,hand_size)
        
        foo = np.zeros(11)
        foo.put([CID_TO_PERS_INDEX[c.id] for c in self.persistent_cards], 1)
        ret = np.append(ret,foo)
        
        foo = np.zeros(7)
        for i,a in self.crafted_items.items():
            foo[i] = a / 2
        return np.append(ret,foo)

    def get_num_cards_to_draw(self) -> int:
        "Returns the number of cards to draw at the end of the turn (In Evening)."
        bases_on_board = sum(self.get_num_buildings_on_track(bind) for bind in range(3))
        return 4 - bases_on_board

    def add_to_supporters(self,card_to_add:Card):
        """
        Given a Card, adds it to the supporters stack and
        updates the supporter suit counts immediately.
        """
        logger.debug(f"\t\t{card_to_add.name} added to supporter pile")
        self.supporters.append(card_to_add)
        suit = card_to_add.suit
        if suit == SUIT_BIRD:
            self.supporter_suit_counts[0] += 1
            self.supporter_suit_counts[1] += 1
            self.supporter_suit_counts[2] += 1
        else:
            self.supporter_suit_counts[suit] += 1
    
    def spend_supporter_helper(self,suit:int):
        "Given a suit ID, updates the suit counts as if a supporter of that suit has been spent."
        if suit == SUIT_BIRD:
            self.supporter_suit_counts[0] -= 1
            self.supporter_suit_counts[1] -= 1
            self.supporter_suit_counts[2] -= 1
        else:
            self.supporter_suit_counts[suit] -= 1
    
    def spread_sympathy_helper(self):
        """
        Changes the number of sympathy tokens on the track as if
        one was placed. Returns the number of points to score.
        """
        n = self.get_num_tokens_in_store(TIND_SYMPATHY)
        self.change_num_tokens(TIND_SYMPATHY,-1)
        return self.point_track[10 - n]


class QuestCard():
    def __init__(self,id:int,suit:int,name:str,requirements:dict) -> None:
        self.id = id
        self.suit = suit
        self.name = name
        self.requirements = requirements

class Vagabond(Player):
    TRACK_IDS = {ITEM_TEA,ITEM_COINS,ITEM_BAG}
    def __init__(self, id: int,) -> None:
        super().__init__(id)
        self.satchel_undamaged = []
        self.satchel_damaged = []
        self.tea_track = 0
        self.coins_track = 0
        self.bag_track = 0
        self.chosen_character = None
        self.relationships = {i:1 for i in range(3)}
        self.completed_quests = {i:[] for i in range(3)}
        self.location = None # 0-11 clearings, 12-18 forests
    
    def __str__(self) -> str:
        return "--- Vagabond ---\n" + super().__str__() + f"\ntodo (if ever)"

    def get_obs_array(self):
        ret = np.zeros(3)
        if self.tea_track > 0:
            ret[0] = self.tea_track / 3
        if self.coins_track > 0:
            ret[1] = self.coins_track / 3
        if self.bag_track > 0:
            ret[2] = self.bag_track / 3
        
        foo = np.zeros((4,8))
        for dam,sat_section in [(0,self.satchel_undamaged),(2,self.satchel_damaged)]:
            exh_counts = {i:0 for i in range(8)}
            unexh_counts = {i:0 for i in range(8)}
            for i,exh in sat_section:
                if exh:
                    exh_counts[i] += 1
                else:
                    unexh_counts[i] += 1

            for item_id,amount in unexh_counts.items():
                foo[dam][item_id] = amount / 4
            for item_id,amount in exh_counts.items():
                foo[dam + 1][item_id] = amount / 4
        ret = np.append(ret,foo)

        item_limit = 6 + 2 * self.bag_track
        num_satchel_items = len(self.satchel_undamaged) + len(self.satchel_damaged)
        ret = np.append(ret, (num_satchel_items / item_limit) - 1)

        foo = np.zeros((3,5))
        for i in range(3):
            foo[i][self.relationships[i]] = 1
        ret = np.append(ret,foo)
        
        foo = np.zeros(18)
        bar = np.zeros(3)
        if self.chosen_character is not None:
            foo[self.chosen_character] = 1

        for suit,qlist in self.completed_quests.items():
            for qcard in qlist:
                foo[qcard.id + 3] = 1
            bar[suit] = min(4,len(qlist)) / 4
        foo = np.append(foo,bar)
        ret = np.append(ret,foo)
        
        foo = np.zeros(19)
        if self.location is not None:
            foo[self.location] = 1
        ret = np.append(ret,foo)

        hand_size = min(5, len(self.hand)) / 5
        ret = np.append(ret,hand_size)
        
        foo = np.zeros(11)
        foo.put([CID_TO_PERS_INDEX[c.id] for c in self.persistent_cards], 1)

        return np.append(ret,foo)
    
    def get_battle_power(self):
        "Returns the total number of undamaged swords in the VB's satchel."
        count = 0
        for i,exh in self.satchel_undamaged:
            if i == ITEM_SWORD:
                count += 1
        return count
    
    def get_refresh_actions(self):
        "Returns a list of AIDs for actions related to refreshing items."
        ans = set()
        for i,exh in self.satchel_undamaged:
            if exh == 1:
                ans.add(i+VB_REFRESH_UNDAM)
        for i,exh in self.satchel_damaged:
            if exh == 1:
                ans.add(i+VB_REFRESH_DAM)
        return list(ans)
    
    def get_damage_actions(self):
        "Returns list of AIDs for choosing what item to damage."
        ans = set()
        # track items
        if self.tea_track > 0:
            ans.add(VB_DAMAGE_UNEXH + ITEM_TEA)
        if self.coins_track > 0:
            ans.add(VB_DAMAGE_UNEXH + ITEM_COINS)
        if self.bag_track > 0:
            ans.add(VB_DAMAGE_UNEXH + ITEM_BAG)
        # satchel items
        for i,exh in self.satchel_undamaged:
            if exh == 1:
                ans.add(VB_DAMAGE_EXH + i)
            else:
                ans.add(VB_DAMAGE_UNEXH + i)
        return list(ans)
    
    def has_exhaustable(self,item_id:int,amount:int=1):
        """
        Returns True only if the Vagabond has at least 'amount' number
        of undamaged AND unexhausted 'item_id' items in their inventory,
        False otherwise.
        """
        if amount == 0:
            return True
        if item_id in Vagabond.TRACK_IDS:
            if item_id == ITEM_TEA:
                return self.tea_track >= amount
            if item_id == ITEM_COINS:
                return self.coins_track >= amount
            return self.bag_track >= amount
        # non track item
        count = 0
        for i,exh in self.satchel_undamaged:
            if i == item_id and exh == 0:
                count += 1
                if count >= amount:
                    return True
        return False
    
    def has_any_damageable(self):
        "Returns True if the VB has ANY damageable item."
        return (self.tea_track + self.coins_track + self.bag_track + len(self.satchel_undamaged)) > 0
    
    def has_any_exhaustable(self):
        "Returns True if the VB has ANY exhaustable item."
        if (self.tea_track + self.coins_track + self.bag_track) > 0:
            return True
        for i,exh in self.satchel_undamaged:
            if exh == 0:
                return True
        return False

    def change_track_amount(self,item_id:int,amount:int):
        if item_id == ITEM_TEA:
            self.tea_track += amount
            word = "Tea"
        if item_id == ITEM_COINS:
            self.coins_track += amount
            word = "Coins"
        if item_id == ITEM_BAG:
            self.bag_track += amount
            word = "Bag"
        logger.debug(f"\t\t\t{word} Track changed by {amount}")
    
    def add_item(self,item_id:int,damaged:int,exhausted:int):
        if damaged == 0:
            self.satchel_undamaged.append((item_id,exhausted))
        else:
            self.satchel_damaged.append((item_id,exhausted))

    def remove_item(self,item_id:int,damaged:int,exhausted:int):
        if damaged == 0:
            self.satchel_undamaged.remove((item_id,exhausted))
        else:
            self.satchel_damaged.remove((item_id,exhausted))

    def damage_item(self,item_id:int,exhausted:int):
        "Damages the target item, moving the tuple to the damaged item part of the satchel."
        if item_id in Vagabond.TRACK_IDS and (exhausted == 0):
            if item_id == ITEM_BAG and self.bag_track == 3 and ((ITEM_BAG,0) in self.satchel_undamaged):
                self.remove_item(ITEM_BAG,0,0)
            else:
                self.change_track_amount(item_id,-1)
        else:
            self.remove_item(item_id,0,exhausted)
        self.add_item(item_id,1,exhausted)
        logger.debug(f"\t\tVagabond damages an {'exhausted' if exhausted else 'unexhausted'} {ID_TO_ITEM[item_id]}")
    
    def repair_item(self,item_id:int,exhausted:int):
        "Repairs the target item, adding it to the track automatically if possible."
        self.remove_item(item_id,1,exhausted)
        if item_id in Vagabond.TRACK_IDS and (exhausted == 0):
            if item_id == ITEM_BAG and self.bag_track == 3:
                self.add_item(ITEM_BAG,0,0)
            else:
                self.change_track_amount(item_id,1)
        else:
            self.add_item(item_id,0,exhausted)
        logger.debug(f"\t\tVagabond repairs an {'exhausted' if exhausted else 'unexhausted'} {ID_TO_ITEM[item_id]}")
    
    def exhaust_item(self,item_id:int):
        "Exhausts one of the given item, assuming it's not damaged, and takes it off the track if needed."
        if item_id in Vagabond.TRACK_IDS:
            if item_id == ITEM_BAG and self.bag_track == 3 and ((ITEM_BAG,0) in self.satchel_undamaged):
                self.remove_item(ITEM_BAG,0,0)
            else:
                self.change_track_amount(item_id,-1)
        else:
            self.remove_item(item_id,0,0)
        self.add_item(item_id,0,1)
        logger.debug(f"\t\tVagabond exhausts a {ID_TO_ITEM[item_id]}")
    
    def refresh_item(self,item_id:int,damaged:int):
        "Refreshes the given item, putting it on the track if it's undamaged."
        self.remove_item(item_id,damaged,1)
        if item_id in Vagabond.TRACK_IDS and (damaged == 0):
            if item_id == ITEM_BAG and self.bag_track == 3:
                self.add_item(ITEM_BAG,0,0)
            else:
                self.change_track_amount(item_id,1)
        else:
            self.add_item(item_id,damaged,0)
        logger.debug(f"\t\tVagabond refreshes {'a damaged' if damaged else 'an undamaged'} {ID_TO_ITEM[item_id]}")

    def is_an_ally(self,player_id:int):
        "Returns True only if the given player is Allied on the Relationships Chart."
        return (self.relationships[player_id] == 4)
    def is_hostile(self,player_id:int):
        "Returns True only if the given player is Hostile on the Relationships Chart."
        return (self.relationships[player_id] == 0)
    

class Battle:
    "Keeps track of info about the current active battle."
    # a choice must be made about using an ambush card
    STAGE_DEF_AMBUSH = 0
    STAGE_ATT_AMBUSH = 1
    # choosing what extra effects/cards to activate
    STAGE_ATT_EFFECTS = 2
    STAGE_DEF_EFFECTS = 3
    # choosing the order in which their pieces will be damaged
    STAGE_ATT_ORDER = 4
    STAGE_DEF_ORDER = 5
    # the cats are given a choice whether to activate field hospitals or not
    STAGE_FIELD_HOSPITALS = 6
    # waiting for the dice roll
    STAGE_DICE_ROLL = 7
    # battle is done
    STAGE_DONE = 8
    # Vagabond is picking their ally
    STAGE_VB_CHOOSE_ALLY = 9

    def __init__(self,att_id:int,def_id:int,clearing_id:int) -> None:
        self.attacker_id = att_id
        self.defender_id = def_id
        self.clearing_id = clearing_id
        self.stage = None
        self.att_rolled_hits = None
        self.att_extra_hits = 0
        self.def_rolled_hits = None
        self.def_extra_hits = 0
        self.att_hits_to_deal = 0
        self.def_hits_to_deal = 0
        self.att_ambush_id = None
        self.def_ambush_id = None
        self.att_cardboard_removed = False
        self.def_cardboard_removed = False

        self.vagabond_battle_ally = None
        self.vagabond_ally_hits_taken = 0
        self.vagabond_hits_taken = 0
    
    def __str__(self) -> str:
        ret = f"--- BATTLE: {ID_TO_PLAYER[self.attacker_id]} attacking {ID_TO_PLAYER[self.defender_id]} in Clearing {self.clearing_id} ---\n"
        ret += f"Roll: {(self.att_rolled_hits,self.def_rolled_hits)}"
        return ret
    
    def get_obs_array(self):
        ret = np.zeros(10)
        if self.stage is not None:
            ret[self.stage] = 1
        if self.stage == self.STAGE_DONE:
            return np.append(ret,np.zeros(33))
        
        foo = np.zeros(11)
        if self.attacker_id >= 0:
            foo[self.attacker_id] = 1
        if self.defender_id >= 0:
            foo[self.defender_id + 4] = 1
        if self.vagabond_battle_ally is not None:
            foo[self.vagabond_battle_ally + 8] = 1
        ret = np.append(ret,foo)

        foo = np.zeros(12)
        if self.clearing_id >= 0:
            foo[self.clearing_id] = 1
        ret = np.append(ret,foo)

        foo = np.zeros(6)
        if self.att_rolled_hits is not None:
            foo[0] = self.att_rolled_hits / 3
        if self.def_rolled_hits is not None:
            foo[1] = self.def_rolled_hits / 3
        foo[2] = self.att_extra_hits / 4
        foo[3] = self.def_extra_hits / 4
        if self.att_ambush_id is not None:
            foo[4] = (self.att_ambush_id + 1) / 4
        if self.def_ambush_id is not None:
            foo[5] = (self.def_ambush_id + 1) / 4
        ret = np.append(ret,foo)

        foo = np.zeros(4)
        foo[0] = min(4, self.att_hits_to_deal) / 4
        foo[1] = min(4, self.def_hits_to_deal) / 4
        foo[2] = min(4, self.vagabond_ally_hits_taken) / 4
        foo[3] = min(4, self.vagabond_hits_taken) / 4
        
        return np.append(ret,foo)


# (Card info, Amount in deck)
# Recipe amounts are (Mouse, Bunny, Fox, Wild)
STANDARD_DECK_COMP = [
    # (id,   Suit,        Name,                    Recipe,    is_ambush, is_dom, is_persistent    item,          points), Amount
    (Card(0, SUIT_BIRD,   "Ambush! (Bird)",        (0,0,0,0),   True,      False,   False,      ITEM_NONE,        0),      2),
    (Card(1, SUIT_RABBIT,  "Ambush! (Rabbit)",     (0,0,0,0),   True,      False,   False,      ITEM_NONE,        0),      1),
    (Card(2, SUIT_FOX,    "Ambush! (Fox)",         (0,0,0,0),   True,      False,   False,      ITEM_NONE,        0),      1),
    (Card(3, SUIT_MOUSE,  "Ambush! (Mouse)",       (0,0,0,0),   True,      False,   False,      ITEM_NONE,        0),      1),
    (Card(4, SUIT_FOX,    "Anvil",                 (0,0,1,0),   False,     False,   False,      ITEM_HAMMER,      2),      1),
    (Card(5, SUIT_BIRD,   "Armorers",              (0,0,1,0),   False,     False,   True,       ITEM_NONE,        0),      2),
    (Card(6, SUIT_BIRD,   "Arms Trader",           (0,0,2,0),   False,     False,   False,      ITEM_SWORD,       2),      1),
    (Card(7, SUIT_RABBIT,  "A Visit to Friends",   (0,1,0,0),   False,     False,   False,      ITEM_BOOT,        1),      1),
    (Card(8, SUIT_RABBIT,  "Bake Sale",            (0,2,0,0),   False,     False,   False,      ITEM_COINS,       3),      1),
    (Card(9, SUIT_RABBIT,  "Better Burrow Bank",   (0,2,0,0),   False,     False,   True,       ITEM_NONE,        0),      2),
    (Card(10,SUIT_BIRD,   "Birdy Bindle",          (1,0,0,0),   False,     False,   False,      ITEM_BAG,         1),      1),
    (Card(11,SUIT_BIRD,   "Brutal Tactics",        (0,0,2,0),   False,     False,   True,       ITEM_NONE,        0),      2),
    (Card(12,SUIT_RABBIT,  "Cobbler",              (0,2,0,0),   False,     False,   True,       ITEM_NONE,        0),      2),
    (Card(13,SUIT_MOUSE,  "Codebreakers",          (1,0,0,0),   False,     False,   True,       ITEM_NONE,        0),      2),
    (Card(14,SUIT_RABBIT,  "Command Warren",       (0,2,0,0),   False,     False,   True,       ITEM_NONE,        0),      2),
    (Card(15,SUIT_BIRD,   "Crossbow (Bird)",       (0,0,1,0),   False,     False,   False,      ITEM_CROSSBOW,    1),      1),
    (Card(16,SUIT_MOUSE,  "Crossbow (Mouse)",      (0,0,1,0),   False,     False,   False,      ITEM_CROSSBOW,    1),      1),
    (Card(17, SUIT_FOX,    "Favor of the Foxes",   (0,0,3,0),   False,     False,   False,      ITEM_NONE,        0),      1),
    (Card(18, SUIT_MOUSE,  "Favor of the Mice",    (3,0,0,0),   False,     False,   False,      ITEM_NONE,        0),      1),
    (Card(19, SUIT_RABBIT, "Favor of the Rabbits", (0,3,0,0),   False,     False,   False,      ITEM_NONE,        0),      1),
    (Card(20, SUIT_FOX,    "Foxfolk Steel",        (0,0,2,0),   False,     False,   False,      ITEM_SWORD,       2),      1),
    # (id,   Suit,        Name,                    Recipe,    is_ambush, is_dom, is_persistent    item,          points), Amount
    (Card(21, SUIT_FOX,    "Gently Used Knapsack", (1,0,0,0),   False,     False,   False,      ITEM_BAG,         1),      1),
    (Card(22, SUIT_MOUSE,   "Investments",         (0,2,0,0),   False,     False,   False,      ITEM_COINS,       3),      1),
    (Card(23, SUIT_MOUSE,    "Mouse-in-a-Sack",    (1,0,0,0),   False,     False,   False,      ITEM_BAG,         1),      1),
    (Card(24, SUIT_FOX,   "Protection Racket",     (0,2,0,0),   False,     False,   False,      ITEM_COINS,       3),      1),
    (Card(25, SUIT_RABBIT,  "Root Tea (Rabbit)",   (1,0,0,0),   False,     False,   False,      ITEM_TEA,         2),      1),
    (Card(26, SUIT_FOX,  "Root Tea (Fox)",         (1,0,0,0),   False,     False,   False,      ITEM_TEA,         2),      1),
    (Card(27, SUIT_MOUSE,  "Root Tea (Mouse)",     (1,0,0,0),   False,     False,   False,      ITEM_TEA,         2),      1),
    (Card(28, SUIT_BIRD,  "Sappers",               (1,0,0,0),   False,     False,   True,       ITEM_NONE,        0),      2),
    (Card(29, SUIT_MOUSE,  "Scouting Party",       (2,0,0,0),   False,     False,   True,       ITEM_NONE,        0),      2),
    (Card(30, SUIT_RABBIT, "Smuggler's Trail",     (1,0,0,0),   False,     False,   False,      ITEM_BAG,         1),      1),
    (Card(31, SUIT_FOX,   "Stand and Deliver!",    (3,0,0,0),   False,     False,   True,       ITEM_NONE,        0),      2),
    (Card(32, SUIT_MOUSE,   "Sword",               (0,0,2,0),   False,     False,   False,      ITEM_SWORD,       2),      1),
    (Card(33, SUIT_FOX,   "Tax Collector",         (1,1,1,0),   False,     False,   True,       ITEM_NONE,        0),      3),
    (Card(34, SUIT_FOX,   "Travel Gear (Fox)",     (0,1,0,0),   False,     False,   False,      ITEM_BOOT,        1),      1),
    (Card(35, SUIT_MOUSE,   "Travel Gear (Mouse)", (0,1,0,0),   False,     False,   False,      ITEM_BOOT,        1),      1),
    (Card(36, SUIT_BIRD,   "Woodland Runners",     (0,1,0,0),   False,     False,   False,      ITEM_BOOT,        1),      1),
    (Card(37, SUIT_BIRD,  "Royal Claim",           (0,0,0,4),   False,     False,   True,       ITEM_NONE,        0),      1),
    (Card(38,SUIT_MOUSE,  "Mouse Dominance",       (0,0,0,0),   False,     True,    False,      ITEM_NONE,        0),      1),
    (Card(39,SUIT_RABBIT,  "Rabbit Dominance",      (0,0,0,0),   False,     True,    False,      ITEM_NONE,        0),      1),
    (Card(40,SUIT_FOX,    "Fox Dominance",         (0,0,0,0),   False,     True,    False,      ITEM_NONE,        0),      1),
    (Card(41,SUIT_BIRD,   "Bird Dominance",        (0,0,0,0),   False,     True,    False,      ITEM_NONE,        0),      1)
]

CID_AMBUSH_BIRD = 0
CID_AMBUSH_RABBIT = 1
CID_AMBUSH_FOX = 2
CID_AMBUSH_MOUSE = 3
CID_ARMORERS = 5
CID_BRUTAL_TACTICS = 11
CID_SAPPERS = 28
CID_SCOUTING_PARTY = 29
CID_COBBLER = 12
CID_COMMAND_WARREN = 14
CID_CODEBREAKERS = 13
CID_STAND_AND_DELIVER = 31
CID_TAX_COLLECTOR = 33
CID_BBB = 9
CID_ROYAL_CLAIM = 37
CID_FAVORS = {17,18,19}
CID_LOYAL_VIZIER = len(STANDARD_DECK_COMP)

CID_TO_PERS_INDEX = {
    CID_ARMORERS: 0,
    CID_BRUTAL_TACTICS: 1,
    CID_SAPPERS: 2,
    CID_SCOUTING_PARTY: 3,
    CID_COBBLER: 4,
    CID_COMMAND_WARREN: 5,
    CID_CODEBREAKERS: 6,
    CID_STAND_AND_DELIVER: 7,
    CID_TAX_COLLECTOR: 8,
    CID_BBB: 9,
    CID_ROYAL_CLAIM: 10
}

ACTION_TO_BIRD_ID = {
    MC_SPEND_BIRD_CARD: CID_AMBUSH_BIRD,
    MC_SPEND_BIRD_CARD + 1: CID_ARMORERS,
    MC_SPEND_BIRD_CARD + 2: 6, # arms trader
    MC_SPEND_BIRD_CARD + 3: 10, # birdy bindle
    MC_SPEND_BIRD_CARD + 4: CID_BRUTAL_TACTICS,
    MC_SPEND_BIRD_CARD + 5: 15, # bird crossbow
    MC_SPEND_BIRD_CARD + 6: CID_SAPPERS,
    MC_SPEND_BIRD_CARD + 7: 36,
    MC_SPEND_BIRD_CARD + 8: CID_ROYAL_CLAIM,
    MC_SPEND_BIRD_CARD + 9: 41,
}
BIRD_ID_TO_ACTION = {i:a for (a,i) in ACTION_TO_BIRD_ID.items()}

MAP_AUTUMN = ([ # board clearings
    #        id, suit,         num_building_slots, num_ruins, opposite_corner_id, set of adj clearings / forests
    Clearing(0,  SUIT_FOX,     1,                 0,         11,                  {1,3,4}, {0,1}),
    Clearing(1,  SUIT_RABBIT,  2,                 0,         -1,                  {0,2}, {0}),
    Clearing(2,  SUIT_MOUSE,   2,                 0,         8,                   {1,3,7}, {0,2}),
    Clearing(3,  SUIT_RABBIT,  1,                 1,         -1,                  {0,2,5}, {0,1,2}),
    Clearing(4,  SUIT_MOUSE,   2,                 0,         -1,                  {0,5,8}, {1,3}),
    Clearing(5,  SUIT_FOX,     1,                 1,         -1,                  {3,4,6,8,10}, {1,2,3,4,5}),
    Clearing(6,  SUIT_MOUSE,   2,                 1,         -1,                  {5,7,11}, {2,5,6}),
    Clearing(7,  SUIT_FOX,     1,                 1,         -1,                  {2,6,11}, {2,6}),
    Clearing(8,  SUIT_RABBIT,  1,                 0,         2,                   {4,5,9}, {3,4}),
    Clearing(9,  SUIT_FOX,     2,                 0,         -1,                  {8,10}, {4}),
    Clearing(10, SUIT_MOUSE,   2,                 0,         -1,                  {5,9,11}, {4,5}),
    Clearing(11, SUIT_RABBIT,  1,                 0,         0,                   {6,7,10}, {5,6})
], [ # board forests
    Forest(0,{0,1,2,3},{1,2}),
    Forest(1,{0,3,4,5},{0,2,3}),
    Forest(2,{2,3,5,6,7},{0,1,5,6}),
    Forest(3,{4,5,8},{1,4}),
    Forest(4,{5,8,9,10},{3,5}),
    Forest(5,{5,6,10,11},{2,4,6}),
    Forest(6,{6,7,11},{2,5})
])

# MAP_WINTER = [
#     #        id, suit,         num_building_slots, num_ruins, opposite_corner_id, set of adj clearings
#     Clearing(0,  SUIT_FOX,     1,                 0,         11,                  {1,4,5}),
#     Clearing(1,  SUIT_RABBIT,  2,                 0,         -1,                  {0,2}),
#     Clearing(2,  SUIT_MOUSE,   2,                 0,         -1,                  {1,3}),
#     Clearing(3,  SUIT_RABBIT,  1,                 0,         8,                   {2,6,7}),
#     Clearing(4,  SUIT_MOUSE,   1,                 0,         -1,                  {0,8}),
#     Clearing(5,  SUIT_FOX,     2,                 1,         -1,                  {0,8,9}),
#     Clearing(6,  SUIT_MOUSE,   2,                 1,         -1,                  {3,10,11}),
#     Clearing(7,  SUIT_FOX,     1,                 0,         -1,                  {3,11}),
#     Clearing(8,  SUIT_RABBIT,  2,                 0,         3,                   {4,5,9}),
#     Clearing(9,  SUIT_FOX,     1,                 1,         -1,                  {5,8,10}),
#     Clearing(10, SUIT_MOUSE,   1,                 1,         -1,                  {6,9,11}),
#     Clearing(11, SUIT_RABBIT,  2,                 0,         0,                   {6,7,10})
# ]

CHOSEN_MAP = MAP_AUTUMN

CLEARING_SUITS = {
    SUIT_FOX: [c.id for c in CHOSEN_MAP[0] if c.suit == SUIT_FOX],
    SUIT_MOUSE: [c.id for c in CHOSEN_MAP[0] if c.suit == SUIT_MOUSE],
    SUIT_RABBIT: [c.id for c in CHOSEN_MAP[0] if c.suit == SUIT_RABBIT]
}

QUEST_DECK_COMP = [
    (QuestCard(0,SUIT_MOUSE,"Expel Bandits",{ITEM_SWORD:2}), 1),
    (QuestCard(1,SUIT_MOUSE,"Guard Duty",{ITEM_SWORD:1,ITEM_TORCH:1}), 1),
    (QuestCard(2,SUIT_MOUSE,"Fend off a Bear",{ITEM_CROSSBOW:1,ITEM_TORCH:1}), 1),
    (QuestCard(3,SUIT_MOUSE,"Escort",{ITEM_BOOT:2}), 1),
    (QuestCard(4,SUIT_MOUSE,"Logistics Help",{ITEM_BOOT:1,ITEM_BAG:1}), 1),
    (QuestCard(5,SUIT_RABBIT,"Guard Duty",{ITEM_SWORD:1,ITEM_TORCH:1}), 1),
    (QuestCard(6,SUIT_RABBIT,"Errand",{ITEM_TEA:1,ITEM_BOOT:1}), 1),
    (QuestCard(7,SUIT_RABBIT,"Give a Speech",{ITEM_TEA:1,ITEM_TORCH:1}), 1),
    (QuestCard(8,SUIT_RABBIT,"Fend off a Bear",{ITEM_CROSSBOW:1,ITEM_TORCH:1}), 1),
    (QuestCard(9,SUIT_RABBIT,"Expel Bandits",{ITEM_SWORD:2}), 1),
    (QuestCard(10,SUIT_FOX,"Fundraising",{ITEM_TEA:1,ITEM_COINS:1}), 1),
    (QuestCard(11,SUIT_FOX,"Errand",{ITEM_TEA:1,ITEM_BOOT:1}), 1),
    (QuestCard(12,SUIT_FOX,"Logistics Help",{ITEM_BOOT:1,ITEM_BAG:1}), 1),
    (QuestCard(13,SUIT_FOX,"Repair a Shed",{ITEM_HAMMER:1,ITEM_TORCH:1}), 1),
    (QuestCard(14,SUIT_FOX,"Give a Speech",{ITEM_TEA:1,ITEM_TORCH:1}), 1)
]