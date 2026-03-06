'''
Simple example pokerbot, written in Python.
'''
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot
import eval7

import random



def emc(my_hand_strs, board_strs, opp_known_strs, n=200):
        """
        Returns your win probability (0.0 - 1.0).
        Uses Monte Carlo simulation.
        Automatically includes opponent's revealed card when you win auction.
        """
        my_cards = [eval7.Card(c) for c in my_hand_strs]
        board = [eval7.Card(c) for c in board_strs]
        opp_known = [eval7.Card(c) for c in opp_known_strs]
        dead = set(my_hand_strs + board_strs + opp_known_strs)

        deck = [eval7.Card(r + s) for r in '23456789TJQKA' for s in 'shcd'
                if (r + s) not in dead]

        board_need = 5 - len(board)
        opp_need = 2 - len(opp_known)

        wins = ties = 0
        for _ in range(n):
            draw = random.sample(deck, opp_need + board_need)
            opp_hand = opp_known + draw[:opp_need]
            full_brd = board + draw[opp_need:]

            my_score = eval7.evaluate(my_cards + full_brd)
            opp_score = eval7.evaluate(opp_hand + full_brd)

            if my_score > opp_score:
                wins += 1
            elif my_score == opp_score:
                ties += 1

        return (wins + 0.5 * ties) / n
    
    
def ev (pot, cost_to_call):
    return (cost_to_call/(pot + cost_to_call))


class Player(BaseBot):
    def __init__(self) -> None:
        pass

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        pass
       
    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        pass
        
    def get_move(self, game_info: GameInfo, current_state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        
        
        
       


if __name__ == '__main__':
    run_bot(Player(), parse_args())