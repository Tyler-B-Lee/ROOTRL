from classes import *

TIND_BOMB = 0
TIND_SNARE = 1
TIND_EXTORTION = 2
TIND_RAID = 3

class Corvid(Player):

    def __init__(self, id: int) -> None:
        super().__init__(id)
        self.warrior_storage = 15
        for tid in range(4):
            self.tokens[tid] = 2