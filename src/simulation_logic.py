from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Any
import random
from datetime import datetime
import os
import pandas as pd
from config_schema import Config
from export_results import export_results
from simulation_helpers import *


Action = str  # 'hit','stand','double','split','surrender'

def run_blackjack_mc(
    config,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Monte-Carlo control for infinite-deck blackjack. Returns learned Q, visit counts,
    and basic-strategy grids as simple dict-of-dicts.
    """
    rules = config.rules
    print(f"Game rules: {rules}")
    episodes = config.simulation.episodes
    seed = config.simulation.seed
    print(f"Episodes: {episodes} Seed: {seed}")
    rng = random.Random(seed)

    # Q(s,a) and counts; state is a compact tuple
    Q: Dict[Tuple, Dict[Action, float]] = {}
    N: Dict[Tuple, Dict[Action, int]] = {}

    # helper: epsilon im späteren Verlauf variieren ist der Call
    # def get_epsilon_state(sk, c=1000):
    #     n = sum(N.get(sk, {}).values())
    #     return c / (c + n)
    
    def get_epsilon_state(sk, N0=1000.0, eps_max=0.3, eps_min=0.1):
        N_s = sum(N.get(sk, {}).values())
        e = N0 / (N0 + N_s)
        return max(eps_min, min(eps_max, e))
    
    # state encoder for Q
    def encode_state(pl_total: int, pl_usable_ace: bool, d_up: Any,
                     pr: Optional[Any], num_cards: int, after_split: bool,
                     splits_done: int) -> Tuple:
        can_d = can_double_now(pl_total, num_cards, after_split, rules)
        can_spl = (rules.allow_splits and (num_cards == 2) and (pr is not None)
                   and (splits_done < rules.max_splits))
        # If split aces and one-card rules applies, you cannot hit/double; resplit depends on rules.
        return (pl_total, int(pl_usable_ace), canon_rank(d_up),
                canon_pair_rank(pr) if pr is not None else '0',
                num_cards, int(after_split), splits_done,
                int(can_d), int(can_spl))

    # allowed actions at this decision point
    def allowed_actions(state_key: Tuple, initial_hand: bool, resplittable_aces: bool) -> List[Action]:
        (_, _, _, pr_s, num_cards, after_split_i, splits_done, can_d_i, can_spl_i) = state_key
        actions = []
        after_split = bool(after_split_i)
        can_d = bool(can_d_i)
        can_spl = bool(can_spl_i)

        # Base actions
        actions.extend(['hit', 'stand'])
        if can_d:
            actions.append('double')
        if can_spl:
            actions.append('split')
        # surrender only on original first decision, two cards, not after split
        if rules.allow_surrender.lower() != 'none' and initial_hand and not after_split and num_cards == 2:
            actions.append('surrender')
        return actions

    # epsilon-greedy over allowed actions
    def choose_action(sk, acts, after_split=False):
        if len(acts) == 1:
            return acts[0]
        eps = get_epsilon_state(sk)
        if after_split:   #be greedy if after split
            eps = min(get_epsilon_state(sk), 0.1)
        if rng.random() < eps:
            return rng.choice(acts)
        qsa = Q.get(sk, {})
        vals = [qsa.get(a, 0.0) for a in acts]
        m = max(vals)
        tie_idxs = [i for i,v in enumerate(vals) if v == m]
        return acts[rng.choice(tie_idxs)]

    # update Q first-visit
    def update_q(first_visits: List[Tuple[Tuple, Action]], G: float):
        # Running average per (s,a)
        for sk, a in first_visits:
            row = Q.setdefault(sk, {})
            cnts = N.setdefault(sk, {})
            old = row.get(a, 0.0)
            n = cnts.get(a, 0) + 1
            row[a] = old + (G - old) / n    #update step 
            # row[a] = old + (G - old) * 0.01    #update step 
            cnts[a] = n

    # episode loop
    eps_count = 0
    while eps_count < episodes:
        eps_count += 1

        # Deal
        d_up = draw_card(rng)
        d_hole = draw_card(rng)
        p_cards = [draw_card(rng), draw_card(rng)]

        # Check immediate naturals and peek logic for surrender-late resolution
        player_natural = is_blackjack(p_cards)
        dealer_is_bj = is_blackjack([d_up, d_hole])
        peekable = (d_up == 'A') or (d_up in TEN_RANKS)

        # Prepare player hands queue
        hands = [{
            'cards': p_cards[:],
            'after_split': False,
            'splits_done': 0,
            'doubled': False,
            'from_split_aces': False,  # special one-card rules
            'resolved': False,         # True if surrendered or finished
            'surrendered': False,
            'surrender_pay': 0.0,      # -0.5 if surrender succeeds
            'natural_paid': False,     # True if 3:2 or 6:5 already applied
        }]

        first_decision_logged = False
        first_visits: List[Tuple[Tuple, Action]] = []

        #Solve natural BJ for dealer/player
        if rules.peek_rule == "US" and peekable:
            if rules.allow_surrender == "early":
                surrender_decision = choose_if_surrender()  #Implement later...
                if surrender_decision == "early_surrender":
                    update_q([], -0.5)
            if dealer_is_bj:
                G = 0.0 if player_natural else -1.0
                update_q([], G)
                continue
        if player_natural and not dealer_is_bj:
            bonus = bj_multiplier(rules)
            G = bonus  
            update_q(first_visits, G)
            continue

        # Decision process over possibly multiple hands due to splits
        while True:
            # Find next unresolved hand
            idx = next((i for i, h in enumerate(hands) if not h['resolved']), None)
            if idx is None:
                break
            h = hands[idx]
            cards = h['cards']
            total, usable = hand_value(cards)
            pr = pair_rank(cards) if len(cards) == 2 else None

            # Determine if resplitting aces is currently possible
            resplittable_aces = (
                len(cards) == 2 and
                pair_rank(cards) == 'A' and
                h['splits_done'] < rules.max_splits and
                rules.resplit_aces
            )

            sk = encode_state(
                pl_total=total,
                pl_usable_ace=usable,
                d_up=d_up,
                pr=pr,
                num_cards=len(cards),
                after_split=h['after_split'],
                splits_done=h['splits_done']
            )

            initial_hand = (len(cards) == 2 and not h['after_split'])
            acts = allowed_actions(sk, initial_hand=initial_hand, resplittable_aces=resplittable_aces)

            # Log first decision for grid aggregation (only once per episode, on the very first hand's first choice)
            if not first_decision_logged and initial_hand:
                first_decision_logged = True
                # Choose action with epsilon-greedy but also record which category this first decision belongs to
                category = ('pair' if pr is not None else ('soft' if usable else 'hard'))
                label = (str(pr) if category == 'pair' else total)
                # Choose action now to record; we also need to actually execute it, so reuse below
                action = choose_action(sk, acts)
                # Also first-visit MC bookkeeping
                first_visits.append((sk, action))
            else:
                # Choose action and record first-visit if not seen in this episode
                if h["after_split"]:
                    modified_state= sk[:5] + (0, 0) + sk[7:]
                    action = choose_action(modified_state, acts, after_split=True)
                else:
                    action = choose_action(sk, acts)
                if (sk, action) not in first_visits:
                    first_visits.append((sk, action))

            # Execute action
            if action == 'surrender':
                if rules.allow_surrender.lower() == 'early':
                    h['surrendered'] = True
                    h['surrender_pay'] = -0.5
                    h['resolved'] = True
                else:
                    # late surrender: if dealer has blackjack, surrender not allowed -> full loss
                    if dealer_is_bj:
                        h['surrendered'] = True
                        h['surrender_pay'] = -1.0
                        h['resolved'] = True
                    else:
                        h['surrendered'] = True
                        h['surrender_pay'] = -0.5
                        h['resolved'] = True
                continue

            if action == 'stand':
                h['resolved'] = True
                continue

            if action == 'double':
                # draw exactly one, then stand
                h['cards'].append(draw_card(rng))
                h['doubled'] = True
                # adjust for aces
                t, _ = hand_value(h['cards'])
                h['resolved'] = True
                continue

            if action == 'hit':
                h['cards'].append(draw_card(rng))
                t, _ = hand_value(h['cards'])
                if t >= 21:
                    # 21 or bust -> stand naturally
                    h['resolved'] = True
                continue

            if action == 'split':
                if h['from_split_aces']:
                    assert rules.resplit_aces and pair_rank(cards)=='A', "Only RSA allowed here"
                # Only possible if pair and ruless allow; remove current hand and spawn two
                a, b = cards
                # Each child hand gets one of the pair + one drawn card immediately
                child1 = {
                    'cards': [a, draw_card(rng)],
                    'after_split': True,
                    'splits_done': h['splits_done'] + 1,
                    'doubled': False,
                    'from_split_aces': (a == 'A'),
                    'resolved': False,
                    'surrendered': False,
                    'surrender_pay': 0.0,
                    'natural_paid': False,
                }
                child2 = {
                    'cards': [b, draw_card(rng)],
                    'after_split': True,
                    'splits_done': h['splits_done'] + 1,
                    'doubled': False,
                    'from_split_aces': (b == 'A'),
                    'resolved': False,
                    'surrendered': False,
                    'surrender_pay': 0.0,
                    'natural_paid': False,
                }
                # Split aces: one-card only, auto-stand, but allow immediate re-split if the drawn card is Ace and rules permits.
                for ch in (child1, child2):
                    if ch['from_split_aces']:
                        # if resplit aces and drawn card is Ace and splits remaining, we will allow 'split' on that hand in the loop
                        # else auto-stand after one card
                        if not (rules.resplit_aces and pair_rank(ch['cards']) == 'A' and ch['splits_done'] < rules.max_splits):
                            ch['resolved'] = True  # auto-stand after one card
                # replace current with the two children (order: process new hands next)
                hands.pop(idx)
                hands.insert(idx, child2)
                hands.insert(idx, child1)
                continue

            # Fallback safety
            raise RuntimeError("Unhandled action")

        # Dealer plays once all player hands decided (unless all surrendered)
        any_active = any(not h['surrendered'] for h in hands)
        if any_active:
            # Use the already drawn hole card; re-play dealer from scratch deterministically
            dealer_total, dealer_bj_final = dealer_play(d_up, d_hole, rules, rng)
        else:
            dealer_total, dealer_bj_final = (0, False)

        # Settle all hands and compute episode return per initial unit bet
        G = 0.0
        for h in hands:
            if h['surrendered']:
                G += h['surrender_pay']
            else:
                G += settle_hand(
                    player_cards=h['cards'],
                    dealer_total=dealer_total,
                    dealer_bj=dealer_bj_final,
                    doubled=h['doubled'],
                    natural_already_paid=False,
                    peek_rule=rules.peek_rule
                )

        # MC update
        update_q(first_visits, G)

    # ---------------------------- Policy extraction for grids ----------------------------
    def best_action_for(cat: str, label: Any, up: Any):
        pr = None
        if cat == 'pair':
            pr = label if label != '10' else 10
            total, usable = hand_value([pr, pr])
        else:
            total = int(label)
            usable = (cat == 'soft')

        sk = encode_state(
            pl_total=total,
            pl_usable_ace=usable,
            d_up=up,
            pr=pr,
            num_cards=2,
            after_split=False,
            splits_done=0
        )

        # initial_hand True because this is the first decision table
        acts = allowed_actions(sk, initial_hand=True, resplittable_aces=False)

        qsa = Q.get(sk, {})
        if not qsa:
            return 'unknown', 0.0, 0

        # tie-break preference
        pref = {'split':4,'double':3,'surrender':2,'stand':1,'hit':0}
        best_a = max(
            acts,
            key=lambda a: (qsa.get(a, 0.0), pref.get(a, -1))
        )

        visits = sum(N.get(sk, {}).values())
        return best_a, qsa.get(best_a, 0.0), visits

    up_cols = [2,3,4,5,6,7,8,9,10,'A']
    # Hard 5..21
    hard_grid = {row: {up: best_action_for('hard', row, up)[0] for up in up_cols} for row in range(5, 21)}
    hard_grid[21] = {up: 'stand' for up in up_cols}
    # Soft 13..21 (A,2 .. A,10/soft 21)
    soft_grid = {row: {up: best_action_for('soft', row, up)[0] for up in up_cols} for row in range(13, 21)}
    soft_grid[21] = {up: 'stand' for up in up_cols}
    # Pairs 2..10, A
    pair_rows = [2,3,4,5,6,7,8,9,10,'A']
    pair_grid = {str(row): {up: best_action_for('pair', str(row), up)[0] for up in up_cols} for row in pair_rows}

    result = {
        "Q": Q,
        "N": N,
        "hard_grid": hard_grid,
        "soft_grid": soft_grid,
        "pair_grid": pair_grid,
        "rules": rules
    }
    
    if save_dir is not None:
        export_results(save_dir, result, up_cols)
    return result

def main(cfg, export_dir):
    if not cfg:
        cfg = Config()
    results = run_blackjack_mc(cfg, save_dir=export_dir)

