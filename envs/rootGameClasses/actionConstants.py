# Generic / Shared Action numbers
GEN_CRAFT_CARD = 0
GEN_DISCARD_CARD = GEN_CRAFT_CARD + 41
GEN_USE_AMBUSH = GEN_DISCARD_CARD + 42
GEN_EFFECTS_NONE = GEN_USE_AMBUSH + 5
GEN_EFFECTS_ARMORERS = GEN_EFFECTS_NONE + 1
GEN_EFFECTS_BRUTTACT = GEN_EFFECTS_ARMORERS + 1
GEN_EFFECTS_SAPPERS = GEN_EFFECTS_BRUTTACT + 1
GEN_EFFECTS_ARM_BT = GEN_EFFECTS_SAPPERS + 1
GEN_EFFECTS_ARMSAP = GEN_EFFECTS_ARM_BT + 1
GEN_SKIP_BBB = GEN_EFFECTS_ARMSAP + 1
GEN_SKIP_CW = GEN_SKIP_BBB + 1
GEN_SKIP_MOVE = GEN_SKIP_CW + 1
GEN_ACTIVATE_DOMINANCE = GEN_SKIP_MOVE + 1
GEN_TAKE_DOMINANCE = GEN_ACTIVATE_DOMINANCE + 4
GEN_END_TURN = GEN_TAKE_DOMINANCE + 4
GEN_USE_TAX_COLLECTOR = GEN_END_TURN + 1
GEN_CRAFT_ROYAL_CLAIM = GEN_USE_TAX_COLLECTOR + 12
GEN_USE_ROYAL_CLAIM = GEN_CRAFT_ROYAL_CLAIM + 15
GEN_MOVE_CLEARINGS = GEN_USE_ROYAL_CLAIM + 1
GEN_MOVE_AMOUNT = GEN_MOVE_CLEARINGS + 144

# Marquise de Cat Action Numbers
MC_PLACE_KEEP = GEN_MOVE_AMOUNT + 25
MC_PLACE_WOOD = MC_PLACE_KEEP + 12
MC_BATTLE_EYRIE = MC_PLACE_WOOD + 12
MC_BATTLE_ALLIANCE = MC_BATTLE_EYRIE + 12
MC_BATTLE_VAGABOND = MC_BATTLE_ALLIANCE + 12
MC_BUILD_SAWMILL = MC_BATTLE_VAGABOND + 12
MC_BUILD_WORKSHOP = MC_BUILD_SAWMILL + 12
MC_BUILD_RECRUITER = MC_BUILD_WORKSHOP + 12
MC_SPEND_BIRD_CARD = MC_BUILD_RECRUITER + 12
MC_RECRUIT = MC_SPEND_BIRD_CARD + 10
MC_ORDER_KEEP = MC_RECRUIT + 1
MC_ORDER_WOOD = MC_ORDER_KEEP + 1
MC_ORDER_SAWMILL = MC_ORDER_WOOD + 1
MC_ORDER_WORKSHOP = MC_ORDER_SAWMILL + 1
MC_ORDER_RECRUITER = MC_ORDER_WORKSHOP + 1
MC_OVERWORK_CLEARING = MC_ORDER_RECRUITER + 1
MC_SKIP_FH = MC_OVERWORK_CLEARING + 12
MC_FIELD_HOSPITALS = MC_SKIP_FH + 1
MC_CHOOSE_OUTRAGE = MC_FIELD_HOSPITALS + 42
MC_RECRUIT_LOCATION = MC_CHOOSE_OUTRAGE + 42
MC_SPEND_WOOD = MC_RECRUIT_LOCATION + 12
MC_SKIP_TO_DAY = MC_SPEND_WOOD + 12
MC_USE_BBB = MC_SKIP_TO_DAY + 1
MC_USE_STAND_DELIVER = MC_USE_BBB + 3
MC_SKIP_CRAFTING = MC_USE_STAND_DELIVER + 3
MC_SKIP_TO_EVENING = MC_SKIP_CRAFTING + 1
MC_USE_CODEBREAKERS = MC_SKIP_TO_EVENING + 1

# Eyrie Dynasties Action Numbers
EY_RECRUIT_LOCATION = GEN_MOVE_AMOUNT + 25
EY_BUILD_LOCATION = EY_RECRUIT_LOCATION + 12
EY_BATTLE_MARQUISE = EY_BUILD_LOCATION + 12
EY_BATTLE_ALLIANCE = EY_BATTLE_MARQUISE + 12
EY_BATTLE_VAGABOND = EY_BATTLE_ALLIANCE + 12
EY_CHOOSE_LEADER = EY_BATTLE_VAGABOND + 12
EY_DECREE_RECRUIT = EY_CHOOSE_LEADER + 4
EY_DECREE_MOVE = EY_DECREE_RECRUIT + 4
EY_DECREE_BATTLE = EY_DECREE_MOVE + 4
EY_DECREE_BUILD = EY_DECREE_BATTLE + 4
EY_DECREE_CARD = EY_DECREE_BUILD + 4
EY_A_NEW_ROOST = EY_DECREE_CARD + 42
EY_CHOOSE_OUTRAGE = EY_A_NEW_ROOST + 12
EY_USE_BBB = EY_CHOOSE_OUTRAGE + 42
EY_USE_STAND_DELIVER = EY_USE_BBB + 3
EY_USE_CODEBREAKERS = EY_USE_STAND_DELIVER + 3
EY_SKIP_TO_DAY = EY_USE_CODEBREAKERS + 3
EY_SKIP_2ND_DECREE = EY_SKIP_TO_DAY + 1
EY_SKIP_CRAFTING = EY_SKIP_2ND_DECREE + 1
EY_SKIP_TO_EVENING = EY_SKIP_CRAFTING + 1

# Woodland Alliance Action Numbers
WA_RECRUIT_LOCATION = GEN_MOVE_AMOUNT + 25
WA_REVOLT_LOCATION = WA_RECRUIT_LOCATION + 12
WA_ORGANIZE_LOCATION = WA_REVOLT_LOCATION + 12
WA_BATTLE_MARQUISE = WA_ORGANIZE_LOCATION + 12
WA_BATTLE_EYRIE = WA_BATTLE_MARQUISE + 12
WA_BATTLE_VAGABOND = WA_BATTLE_EYRIE + 12
WA_DISCARD_SUPPORTER = WA_BATTLE_VAGABOND + 12
WA_SPREAD_SYMPATHY = WA_DISCARD_SUPPORTER + 42
WA_MOBILIZE = WA_SPREAD_SYMPATHY + 12
WA_TRAIN_OFFICER = WA_MOBILIZE + 42
WA_USE_BBB = WA_TRAIN_OFFICER + 42
WA_USE_STAND_DELIVER = WA_USE_BBB + 3
WA_USE_CODEBREAKERS = WA_USE_STAND_DELIVER + 3
WA_SKIP_TO_DAY = WA_USE_CODEBREAKERS + 3
WA_SKIP_REVOLTING = WA_SKIP_TO_DAY + 1
WA_SKIP_TO_EVENING = WA_SKIP_REVOLTING + 1
WA_SKIP_MILITARY_OPS = WA_SKIP_TO_EVENING + 1
WA_ORDER_SYMPATHY = WA_SKIP_MILITARY_OPS + 1
WA_ORDER_BASE = WA_ORDER_SYMPATHY + 1

# Vagabond Action Numbers
VB_BATTLE_MARQUISE = GEN_END_TURN + 1
VB_BATTLE_EYRIE = VB_BATTLE_MARQUISE + 12
VB_BATTLE_ALLIANCE = VB_BATTLE_EYRIE + 12
VB_USE_BBB = VB_BATTLE_ALLIANCE + 12
VB_USE_STAND_DELIVER = VB_USE_BBB + 3
VB_USE_CODEBREAKERS = VB_USE_STAND_DELIVER + 3
VB_CHOOSE_OUTRAGE = VB_USE_CODEBREAKERS + 3
VB_CHOOSE_CLASS = VB_CHOOSE_OUTRAGE + 42
VB_REFRESH_UNDAM = VB_CHOOSE_CLASS + 3
VB_REFRESH_DAM = VB_REFRESH_UNDAM + 8
VB_MOVE = VB_REFRESH_DAM + 8
VB_EXPLORE = VB_MOVE + 19
VB_START_AIDING = VB_EXPLORE + 1
VB_CHOOSE_AID_EXHAUST = VB_START_AIDING + 126
VB_CHOOSE_AID_TAKE = VB_CHOOSE_AID_EXHAUST + 8
VB_COMPLETE_QUEST = VB_CHOOSE_AID_TAKE + 8
VB_STRIKE = VB_COMPLETE_QUEST + 30
VB_REPAIR_UNEXH = VB_STRIKE + 13
VB_REPAIR_EXH = VB_REPAIR_UNEXH + 8
VB_THIEF_ABILITY = VB_REPAIR_EXH + 8
VB_TINKER_ABILITY = VB_THIEF_ABILITY + 3
VB_RANGER_ABILITY = VB_TINKER_ABILITY + 42
VB_DISCARD_ITEM = VB_RANGER_ABILITY + 1
VB_DAMAGE_UNEXH = VB_DISCARD_ITEM + 32
VB_DAMAGE_EXH = VB_DAMAGE_UNEXH + 8
VB_ACTIVATE_COALITION = VB_DAMAGE_EXH + 8
VB_NO_ALLY_MOVE = VB_ACTIVATE_COALITION + 12
VB_MARQUISE_ALLY_MOVE = VB_NO_ALLY_MOVE + 1
VB_EYRIE_ALLY_MOVE = VB_MARQUISE_ALLY_MOVE + 25
VB_ALLIANCE_ALLY_MOVE = VB_EYRIE_ALLY_MOVE + 20
VB_BATTLE_WITH_ALLY = VB_ALLIANCE_ALLY_MOVE + 10
VB_BATTLE_ALLY_HITS = VB_BATTLE_WITH_ALLY + 4
VB_SKIP_TO_DAY = VB_BATTLE_ALLY_HITS + 3
VB_SKIP_TO_EVENING = VB_SKIP_TO_DAY + 1

AMBUSH_ACTIONS_SET = {GEN_USE_AMBUSH + i for i in range(4)}
CRAFT_ACTIONS_SET = {GEN_CRAFT_CARD + i for i in range(41)}
CRAFT_RC_ACTIONS_SET = {GEN_CRAFT_ROYAL_CLAIM + i for i in range(15)}

FIELD_HOSPITALS_ACTIONS_SET = {MC_FIELD_HOSPITALS + i for i in range(42)}
MC_BUILD_ACTIONS_SET = {(aid + i) for aid in [MC_BUILD_SAWMILL, MC_BUILD_RECRUITER, MC_BUILD_WORKSHOP] for i in range(12)}
MC_BATTLE_ACTIONS_SET = {(aid + i) for aid in [MC_BATTLE_EYRIE, MC_BATTLE_ALLIANCE, MC_BATTLE_VAGABOND] for i in range(12)}
MC_OVERWORK_ACTIONS_SET = {MC_OVERWORK_CLEARING + i for i in range(12)}

SLIP_ACTIONS_SET = {VB_MOVE + i for i in range(19)}
QUEST_ACTIONS_SET = {VB_COMPLETE_QUEST + i for i in range(30)}
FREE_STRIKE_ACTIONS_SET = {VB_STRIKE + i for i in [1,2,3,4,5,7,9,10,11,12]}

SPREAD_SYM_ACTIONS_SET = {WA_SPREAD_SYMPATHY + i for i in range(12)}
REVOLT_ACTIONS_SET = {WA_REVOLT_LOCATION + i for i in range(12)}
ORGANIZE_ACTIONS_SET = {WA_ORGANIZE_LOCATION + i for i in range(12)}

MARQUISE_SKIP_ACTIONS_SET = {
    GEN_SKIP_BBB, GEN_SKIP_CW, GEN_SKIP_MOVE,
    MC_SKIP_CRAFTING, MC_SKIP_FH, MC_SKIP_TO_EVENING, MC_SKIP_TO_DAY,
}

EYRIE_SKIP_ACTIONS_SET = {
    GEN_SKIP_BBB, GEN_SKIP_CW, GEN_SKIP_MOVE,
    EY_SKIP_CRAFTING, EY_SKIP_2ND_DECREE, EY_SKIP_TO_DAY, EY_SKIP_TO_EVENING,
}

ALLIANCE_SKIP_ACTIONS_SET = {
    GEN_SKIP_BBB, GEN_SKIP_CW, GEN_SKIP_MOVE,
    WA_SKIP_MILITARY_OPS, WA_SKIP_REVOLTING, WA_SKIP_TO_DAY, WA_SKIP_TO_EVENING,
}

VAGABOND_SKIP_ACTIONS_SET = {
    GEN_SKIP_BBB, GEN_SKIP_CW, GEN_SKIP_MOVE, 
    VB_SKIP_TO_DAY, VB_SKIP_TO_EVENING,
}