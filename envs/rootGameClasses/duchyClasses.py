from classes import *

BIND_CITADEL = 0
BIND_MARKET = 1
TIND_TUNNEL = 0

MTYPE_SQUIRE = 0
MTYPE_NOBLE = 1
MTYPE_LORD = 2

MIND_FOREMOLE = 0
MIND_CAPTAIN = 1
MIND_MARSHAL = 2
MIND_BRIGADIER = 3
MIND_BANKER = 4
MIND_MAYOR = 5
MIND_DUCHESS_OF_MUD = 6
MIND_BARON_OF_DIRT = 7
MIND_EARL_OF_STONE = 8

class MinisterCard():
    def __init__(self,id:int,name:str,minister_type:int) -> None:
        self.id = id
        self.name = name
        self.minister_type = minister_type

        if minister_type == MTYPE_SQUIRE:
            self.sway_cost = 2
        elif minister_type == MTYPE_NOBLE:
            self.sway_cost = 3
        elif minister_type == MTYPE_LORD:
            self.sway_cost = 4

class Duchy(Player):

    def __init__(self, id: int) -> None:
        super().__init__(id)
        self.warrior_storage = 20
        self.buildings[BIND_CITADEL] = 3
        self.buildings[BIND_MARKET] = 3
        self.tokens[TIND_TUNNEL] = 3

        self.crowns = {MTYPE_SQUIRE: 3, MTYPE_NOBLE: 3, MTYPE_LORD: 3}
        self.unswayed_ministers = [
            MinisterCard(MIND_FOREMOLE,"Foremole",MTYPE_SQUIRE),
            MinisterCard(MIND_CAPTAIN,"Captain",MTYPE_SQUIRE),
            MinisterCard(MIND_MARSHAL,"Marshal",MTYPE_SQUIRE),
            MinisterCard(MIND_BRIGADIER,"Brigadier",MTYPE_NOBLE),
            MinisterCard(MIND_BANKER,"Banker",MTYPE_NOBLE),
            MinisterCard(MIND_MAYOR,"Mayor",MTYPE_NOBLE),
            MinisterCard(MIND_DUCHESS_OF_MUD,"Duchess of Mud",MTYPE_LORD),
            MinisterCard(MIND_BARON_OF_DIRT,"Baron of Dirt",MTYPE_LORD),
            MinisterCard(MIND_EARL_OF_STONE,"Earl of Stone",MTYPE_LORD)
        ]
        self.swayed_ministers = []