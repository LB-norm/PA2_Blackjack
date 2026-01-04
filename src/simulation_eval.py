from typing import Callable, Dict, Tuple, List, Optional, Any
import math
import random

Action = str
StateKey = Tuple

def make_log_checkpoints(total_episodes: int, start: int = 10_000, num: int = 30) -> List[int]:
    """
    Geometric/log-spaced checkpoints from `start` to `total_episodes` (inclusive).
    Produces `num` points (deduplicated and sorted).
    """
    if total_episodes <= 0:
        return []
    if start >= total_episodes:
        return [total_episodes]
    if num < 2:
        return [start, total_episodes]

    cps = set()
    ratio = total_episodes / start
    for i in range(num):
        t = start * (ratio ** (i / (num - 1)))
        cps.add(int(round(t)))
    cps.add(total_episodes)
    return sorted(cps)


def evaluate_greedy_policy_metrics(
    *,
    Q: Dict[StateKey, Dict[Action, float]],
    rules: Any,
    encode_state: Callable[..., StateKey],
    allowed_actions: Callable[[StateKey, bool, bool], List[Action]],
    eval_episodes: int = 200_000,
    eval_seed: int = 12345,
    prev_policy_map: Optional[Dict[Tuple[str, Any, Any], Action]] = None,
) -> Tuple[Dict[str, Any], Dict[Tuple[str, Any, Any], Action]]:
    """
    Computes:
      A) Performance: mean return per hand under greedy policy (epsilon=0) over `eval_episodes`.
      B) Stability: policy flip-rate on S* = all initial two-card combinations vs dealer upcard (excluding naturals).

    Returns (metrics_dict, current_policy_map).

    `current_policy_map` keys are (category, label, dealer_upcard), where:
      - category in {"hard","soft","pair"}
      - label is int total for hard/soft, and str rank for pairs (e.g. "A", "10", "9", ...)
      - dealer_upcard is canonical (2..10,"A")
    """

    # ---------- small helpers ----------
    def _greedy_action_from_Q(sk: StateKey, acts: List[Action], *, total_for_default: int, usable_for_default: bool) -> Action:
        qsa = Q.get(sk, {})
        # If we have no info for this state, use a sane default to avoid pref-bias artifacts.
        if not qsa:
            # conservative fallback: stand on strong totals, otherwise hit
            if 'stand' in acts and total_for_default >= 17:
                return 'stand'
            if 'hit' in acts:
                return 'hit'
            return acts[0]

        pref = {'split': 4, 'double': 3, 'surrender': 2, 'stand': 1, 'hit': 0}
        best = max(acts, key=lambda a: (qsa.get(a, 0.0), pref.get(a, -1)))
        return best

    def _settle_hand(
        player_cards: List[Any],
        dealer_total: int,
        dealer_bj: bool,
        doubled: bool,
    ) -> float:
        stake = 2.0 if doubled else 1.0
        pt, _ = hand_value(player_cards)
        if pt > 21:
            return -stake
        # ENHC: dealer doesn't peek; if dealer has BJ, doubles/splits can lose immediately (your original logic)
        if dealer_bj and str(rules.peek_rule) == "ENHC":
            return -stake
        if dealer_total > 21:
            return stake
        if pt > dealer_total:
            return stake
        if pt < dealer_total:
            return -stake
        return 0.0

    # ---------- build analysis state set S* and compute current greedy actions ----------
    player_ranks_canon = [2,3,4,5,6,7,8,9,10,'A']
    dealer_ups_canon = [2,3,4,5,6,7,8,9,10,'A']

    # precompute unique initial-hand descriptors from unordered canonical two-card combos
    initial_descriptors: List[Tuple[str, Any, int, bool, Optional[Any]]] = []
    # (cat, label, total, usable, pr_obj)
    seen_desc = set()
    for i, r1 in enumerate(player_ranks_canon):
        for r2 in player_ranks_canon[i:]:
            cards = [r1, r2]
            if is_blackjack(cards):  # no decision; exclude from S*
                continue
            total, usable = hand_value(cards)
            pr = pair_rank(cards)  # returns r1 if pair else None
            cat = 'pair' if pr is not None else ('soft' if usable else 'hard')
            label = (canon_pair_rank(pr) if pr is not None else total)
            key = (cat, label)
            if key in seen_desc:
                continue
            seen_desc.add(key)
            initial_descriptors.append((cat, label, total, usable, pr))

    current_policy_map: Dict[Tuple[str, Any, Any], Action] = {}
    covered_states = 0  # states where we could produce a non-unknown greedy action from Q

    for (cat, label, total, usable, pr) in initial_descriptors:
        for up in dealer_ups_canon:
            sk = encode_state(
                pl_total=total,
                pl_usable_ace=usable,
                d_up=up,
                pr=pr,
                num_cards=2,
                after_split=False,
                splits_done=0
            )
            acts = allowed_actions(sk, initial_hand=True, resplittable_aces=False)

            qsa = Q.get(sk, {})
            if not qsa:
                # for stability metric we keep explicit "unknown"
                a = 'unknown'
            else:
                a = _greedy_action_from_Q(sk, acts, total_for_default=total, usable_for_default=usable)
                covered_states += 1

            current_policy_map[(cat, label, canon_rank(up))] = a

    # ---------- stability metric: flip rate vs previous checkpoint ----------
    flip_rate = None
    flips = 0
    denom = 0
    if prev_policy_map is not None:
        for k, a_now in current_policy_map.items():
            a_prev = prev_policy_map.get(k, 'unknown')
            # only compare where both are known
            if a_now != 'unknown' and a_prev != 'unknown':
                denom += 1
                if a_now != a_prev:
                    flips += 1
        flip_rate = (flips / denom) if denom > 0 else None

    # ---------- performance metric: greedy evaluation rollouts ----------
    rng_eval = random.Random(eval_seed)

    returns: List[float] = []
    for _ in range(max(0, eval_episodes)):
        # Deal
        d_up = draw_card(rng_eval)
        d_hole = draw_card(rng_eval)
        p_cards = [draw_card(rng_eval), draw_card(rng_eval)]

        player_natural = is_blackjack(p_cards)
        dealer_is_bj = is_blackjack([d_up, d_hole])
        peekable = (d_up == 'A') or (d_up in TEN_RANKS)

        # US peek: if dealer has BJ, round ends immediately
        if str(rules.peek_rule) == "US" and peekable and dealer_is_bj:
            G = 0.0 if player_natural else -1.0
            returns.append(G)
            continue

        # Player natural (dealer not BJ, or non-peek variant)
        if player_natural and not dealer_is_bj:
            returns.append(bj_multiplier(rules))
            continue

        # Prepare player hands queue (same structure as training, minus learning bookkeeping)
        hands = [{
            'cards': p_cards[:],
            'after_split': False,
            'splits_done': 0,
            'doubled': False,
            'from_split_aces': False,
            'resolved': False,
            'surrendered': False,
            'surrender_pay': 0.0,
        }]

        # Decision loop (greedy)
        while True:
            idx = next((i for i, h in enumerate(hands) if not h['resolved']), None)
            if idx is None:
                break
            h = hands[idx]
            cards = h['cards']
            total, usable = hand_value(cards)
            pr = pair_rank(cards) if len(cards) == 2 else None

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

            action = _greedy_action_from_Q(sk, acts, total_for_default=total, usable_for_default=usable)

            # Execute action (mirrors your training logic)
            if action == 'surrender':
                if str(rules.allow_surrender).lower() == 'early':
                    h['surrendered'] = True
                    h['surrender_pay'] = -0.5
                else:
                    # late surrender: if dealer has BJ, surrender not allowed -> full loss
                    h['surrendered'] = True
                    h['surrender_pay'] = (-1.0 if dealer_is_bj else -0.5)
                h['resolved'] = True
                continue

            if action == 'stand':
                h['resolved'] = True
                continue

            if action == 'double':
                h['cards'].append(draw_card(rng_eval))
                h['doubled'] = True
                h['resolved'] = True
                continue

            if action == 'hit':
                h['cards'].append(draw_card(rng_eval))
                t, _ = hand_value(h['cards'])
                if t >= 21:
                    h['resolved'] = True
                continue

            if action == 'split':
                a, b = cards
                child1 = {
                    'cards': [a, draw_card(rng_eval)],
                    'after_split': True,
                    'splits_done': h['splits_done'] + 1,
                    'doubled': False,
                    'from_split_aces': (a == 'A'),
                    'resolved': False,
                    'surrendered': False,
                    'surrender_pay': 0.0,
                }
                child2 = {
                    'cards': [b, draw_card(rng_eval)],
                    'after_split': True,
                    'splits_done': h['splits_done'] + 1,
                    'doubled': False,
                    'from_split_aces': (b == 'A'),
                    'resolved': False,
                    'surrendered': False,
                    'surrender_pay': 0.0,
                }
                for ch in (child1, child2):
                    if ch['from_split_aces']:
                        if not (rules.resplit_aces and pair_rank(ch['cards']) == 'A' and ch['splits_done'] < rules.max_splits):
                            ch['resolved'] = True
                hands.pop(idx)
                hands.insert(idx, child2)
                hands.insert(idx, child1)
                continue

            raise RuntimeError("Unhandled action in evaluation")

        # Dealer plays if any non-surrendered hand exists
        any_active = any(not h['surrendered'] for h in hands)
        if any_active:
            dealer_total, dealer_bj_final = dealer_play(d_up, d_hole, rules, rng_eval)
        else:
            dealer_total, dealer_bj_final = (0, False)

        G = 0.0
        for h in hands:
            if h['surrendered']:
                G += h['surrender_pay']
            else:
                G += _settle_hand(
                    player_cards=h['cards'],
                    dealer_total=dealer_total,
                    dealer_bj=dealer_bj_final,
                    doubled=h['doubled'],
                )

        returns.append(G)

    n = len(returns)
    mean_return = (sum(returns) / n) if n > 0 else 0.0
    if n >= 2:
        var = sum((x - mean_return) ** 2 for x in returns) / (n - 1)
        stderr = math.sqrt(var / n)
    else:
        stderr = None

    metrics = {
        "eval_episodes": n,
        "mean_return": mean_return,
        "stderr_return": stderr,
        "flip_rate": flip_rate,
        "flip_denom": denom,
        "policy_states_total": len(current_policy_map),
        "policy_states_covered": covered_states,
    }
    return metrics, current_policy_map
