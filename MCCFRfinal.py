"""
List of stuff that went wrong the last time and needs to be fixed now.
1. Train and bot circular reference issue
2. My information handling is ...... suboptimal

Change i want to make later:
1. stack to pot ratio + equity = gametree infoset instead of just equity.

My handling of the opponent reach probability issue depends on the idea that the 2 different instances will alternate actions. If not, the logic breaks.
"""
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot
import eval7
import random
from FinalEngine import Trainer

import numpy as np

NUM_BUCKETS = 25
actions = {
    0: 'fold',
    1: 'call',
    2: 'check',
    3: 'raise_small',
    4: 'raise_big',
    5: 'all_in'
}
NUM_ACTIONS = len(actions)
TERATIONS = 10

class Player(BaseBot):

    def __init__(self, trainer : Trainer) -> None:
        self.reach_probab = 1
        self.my_bet = 0 # amount of chips I have put in the pot so far.
        self.opp_bet = 0 # amount of chips opponent has put in the pot so far.
        self.mc_iterations = 500
        self.action_history = []
        self.trainer = trainer
        self._preflop_equity = 0.5
        self.opp_bid_samples = []
        self.opp_style = "unknown"
        
    def _compute_auction_bid(self, equity, my_chips):
            n = len(self.opp_bid_samples)

            # Cold start (first 8 hands)
            if n < 8:
                if equity > 0.72:
                    return max(1, int(my_chips * 0.04)) if my_chips > max(1, int(my_chips * 0.04)) else my_chips
                elif equity > 0.52:
                    return max(1, int(my_chips * 0.07)) if my_chips > max(1, int(my_chips * 0.07)) else my_chips
                else:
                    return max(1, int(my_chips * 0.09)) if my_chips > max(1, int(my_chips * 0.09)) else my_chips

            # Use learned style
            if self.opp_style == "low":
                target_pct = 0.20 if equity > 0.72 else 0.33 if equity > 0.52 else 0.55
            elif self.opp_style == "med":
                target_pct = 0.25 if equity > 0.72 else 0.40 if equity > 0.52 else 0.65
            else:  # high bidder
                target_pct = 0.15 if equity > 0.72 else 0.25 if equity > 0.52 else 0.50

            s = sorted(self.opp_bid_samples)
            idx = max(0, min(int(target_pct * len(s)), len(s) - 1))
            bid = s[idx]
            return max(bid, my_chips) if my_chips > bid else my_chips

    def get_available_actions(self, current_state: PokerState) -> list[int]:
        available = []
        if ActionFold in current_state.legal_actions:
            available.append(0)
        if ActionCall in current_state.legal_actions:
            available.append(1)
        if ActionCheck in current_state.legal_actions:
            available.append(2)
        if ActionRaise in current_state.legal_actions:
            min_r, max_r = current_state.raise_bounds
            available.append(3)  # raise_small
            available.append(4)  # raise_big
            available.append(5)  # all_in
        return available

    def equity_mc(self,current_state: PokerState) -> float: 
    #(my_hand_strs, board_strs, opp_known_strs, n=200):
        my_cards = [eval7.Card(c) for c in current_state.my_hand]
        board = [eval7.Card(c) for c in current_state.board]
        opp_known = [eval7.Card(c) for c in current_state.opp_revealed_cards]
        dead = set(current_state.my_hand + current_state.board + current_state.opp_revealed_cards)

        deck = [eval7.Card(r + s) for r in '23456789TJQKA' for s in 'shcd'
                if (r + s) not in dead]

        board_need = 5 - len(board)
        opp_need = 2 - len(opp_known)

        wins = ties = 0
        for _ in range(self.mc_iterations):
            draw = random.sample(deck, opp_need + board_need)
            opp_hand = opp_known + draw[:opp_need]
            full_brd = board + draw[opp_need:]

            my_score = eval7.evaluate(my_cards + full_brd)
            opp_score = eval7.evaluate(opp_hand + full_brd)

            if my_score > opp_score:
                wins += 1
            elif my_score == opp_score:
                ties += 1

        return (wins + 0.5 * ties) / self.mc_iterations
    
    def get_infoset_key (self, game_info: GameInfo, current_state: PokerState): 
        equity = self.equity_mc(current_state)
        
        bucket = min(int(equity * NUM_BUCKETS), NUM_BUCKETS-1) # bucket equity into 25 buckets.
        if_loss = - self.my_bet
        if_win = self.opp_bet 
        street_map = {
            'pre-flop': 0,
            'auction': 1,
            'flop': 2,
            'turn': 3,
            'river': 4
        }
        #removed the try and except error.
        
        street_int = street_map[current_state.street]
        cards = [
            current_state.my_hand,
            current_state.board,
            current_state.opp_revealed_cards
        ]
        legal_actions = self.get_available_actions(current_state)
        return [
            street_int, 
            bucket, # might replace  bucket with gametree_pos later if i fond more relevant data to include in the infoset (possibly opponent betting profile) 
            cards,
            if_win, 
            if_loss,
            legal_actions,
            current_state.my_chips, 
            current_state.opp_chips
        ]
    
    def on_hand_end(self, game_info, current_state):
        if game_info.round_num % 10 == 0:
            np.savez('trainer_data.npz',
                delta_regret=self.trainer.delta_regret,
                strategy_convergence=self.trainer.strategy_convergence
            )

    def on_hand_start(self, game_info, current_state):
        self._preflop_equity = self.equity_mc(current_state)
        self._street_bet = set()
        self._my_chips_pre_auction = None
        self._opp_chips_pre_auction = None
        self._my_bid_this_hand = 0
        self._auction_processed = False
        pass
    
    def get_move(self, game_info: GameInfo, current_state: PokerState): 
        trainer = self.trainer
        # still not sure if this is a legal way to pass the trainer class 
        street = current_state.street

        if street == 'auction':
            self._my_chips_pre_auction = current_state.my_chips
            self._opp_chips_pre_auction = current_state.opp_chips
            action = self._auction_action(current_state)
            self._my_bid_this_hand = action.amount
            return action
        if not self._auction_processed and self._my_chips_pre_auction is not None:
            self._infer_opp_bid(current_state)
            self._auction_processed = True
            
        
        infoset = self.get_infoset_key(game_info, current_state)

        bucket = infoset[1]
        street_int = infoset[0]
        trainer.mccfr(infoset, self.reach_probab)
        p = trainer.average_strategy[street_int][bucket]
        a = self.get_available_actions(current_state)
        
        
        for i in range(NUM_ACTIONS):
            if i not in a:
                p[i] = 0
        p = np.nan_to_num(p, nan=0.0)
        total = sum(p)
        
        if total == 0:
            # uniform over legal actions if no strategy learned yet
            for i in a:
                p[i] = 1.0 / len(a)
        else:
            p[:] = p[:] / total
            
        chosen = np.random.choice(len(p), p=p)
        
        self.reach_probab *= p[chosen]
        self.equity = self.equity_mc(current_state)
        
        if chosen == 0:
            return ActionFold()
        
        elif chosen == 1:
            return ActionCall()
        
        elif chosen == 2:
            return ActionCheck()
        
        elif chosen == 3:
            min_r, max_r = current_state.raise_bounds
            
            target = int(min_r + (max_r - min_r) * 0.2)
            return ActionRaise(target)
        
        elif chosen == 4:
            min_r, max_r = current_state.raise_bounds
            target = int(min_r + (max_r - min_r) * 0.4)
            
            return ActionRaise(target)
        
        elif chosen == 5:
            min_r, max_r = current_state.raise_bounds
            target = min(current_state.my_chips, current_state.opp_chips)
            if (current_state.my_chips < target or target < min_r or target > max_r) : return ActionCall()
            return ActionRaise(target)
        
        return ActionCall()

    def _infer_opp_bid(self, state: PokerState):
        if self._my_chips_pre_auction is None or self._opp_chips_pre_auction is None:
            return

        my_delta = self._my_chips_pre_auction - state.my_chips
        opp_delta = self._opp_chips_pre_auction - state.opp_chips

        if my_delta > 0 and opp_delta == 0:        # We won
            self.opp_bid_samples.append(my_delta)
        elif my_delta > 0 and opp_delta > 0:       # Tie
            self.opp_bid_samples.append(my_delta)
        elif opp_delta > 0 and my_delta == 0:      # They won
            self.opp_bid_samples.append(self._my_bid_this_hand + 1)

        # Classify opponent style after enough data
        if len(self.opp_bid_samples) >= 8 and self.opp_style == "unknown":
            avg = sum(self.opp_bid_samples) / len(self.opp_bid_samples)
            if avg < 15:
                self.opp_style = "low"
            elif avg < 300:
                self.opp_style = "med"
            else:
                self.opp_style = "high"

    def _auction_action(self, state: PokerState):
        equity = self.equity_mc(state)
        bid = self._compute_auction_bid(equity, state.my_chips)
        return ActionBid(bid)
    
    
    
if __name__ == '__main__':
    run_bot(Player(trainer=Trainer()), parse_args())