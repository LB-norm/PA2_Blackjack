import random
from typing import Dict, Tuple, List, Optional, Any

# ---------------------------- Cards and hands ----------------------------
RANKS = [2,3,4,5,6,7,8,9,10,'J','Q','K','A']         # infinite deck categories
TEN_RANKS = {10,'J','Q','K'}

def draw_card(rng: random.Random):
    return rng.choices(RANKS, k=1)[0]

def hand_value(cards):
    total = 0
    aces = 0
    for c in cards:
        if c == 'A':
            total += 11
            aces += 1
        else:
            if isinstance(c, str) and c in {'J','Q','K','T'}:
                v = 10
            else:
                v = int(c)
            total += v
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total, aces > 0

def canon_rank(x: Any) -> str:
    return "10" if x in TEN_RANKS else str(x)

def canon_pair_rank(pr: Optional[Any]) -> str:
    if pr is None:
        return None
    return "10" if pr in TEN_RANKS else str(pr)

def is_blackjack(cards: List[Any]) -> bool:
    if len(cards) != 2:
        return False
    t, _ = hand_value(cards)
    return t == 21 and ('A' in cards)

def pair_rank(cards: List[Any]) -> Optional[Any]:
    if len(cards) != 2:
        return None
    a, b = cards
    return a if a == b else None

def bj_multiplier(rules: Any) -> float:
    return 1.5 if str(rules.blackjack_payout) == "3:2" else 1.2

def hits_soft17(rules: Any) -> bool:
    return str(rules.dealer_rule).upper() == "H17"

def double_range(rules: Any) -> tuple[int, int]:
    if rules.double_allowed == "any_two": return (4, 21)
    if rules.double_allowed == "10-11":   return (10, 11)
    if rules.double_allowed == "9-11":    return (9, 11)
    raise ValueError("unknown double_allowed")

# ---------------------------- Environment helpers ----------------------------
def dealer_play(up: Any, hole: Any, rules: any, rng: random.Random) -> Tuple[int, bool]:
    """Return (dealer_total, dealer_blackjack_flag)."""
    cards = [up, hole]
    if is_blackjack(cards):
        return 21, True
    while True:
        total, usable = hand_value(cards)
        if total < 17:
            cards.append(draw_card(rng))
            continue
        if total == 17 and hits_soft17(rules) and usable:
            cards.append(draw_card(rng))
            continue
        return total, False


def can_double_now(total: int, num_cards: int, after_split: bool, rules: Any) -> bool:
    if num_cards != 2:
        return False
    if after_split and not rules.double_after_split:
        return False
    low, high = double_range(rules)
    return low <= total <= high

def settle_hand(player_cards: List[Any], dealer_total: int, dealer_bj: bool,
                doubled: bool, natural_already_paid: bool, peek_rule: str) -> float:
    stake = 2.0 if doubled else 1.0
    # Note: natural blackjack bonus only on original 2-card hand, not on split hands.
    if natural_already_paid:
        return 0.0
    pt, _ = hand_value(player_cards)
    # bust handled as pt>21
    if pt > 21:
        return -stake
    if dealer_bj and peek_rule == "ENHC":
        return -stake
    if dealer_total > 21:
        return stake
    if pt > dealer_total:
        return stake
    if pt < dealer_total:
        return -stake
    return 0.0  # push