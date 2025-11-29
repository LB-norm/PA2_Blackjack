from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Tuple

class Rules(BaseModel):
    # Everything in Game Setup
    blackjack_payout: Literal['3:2','6:5'] = '3:2'
    dealer_rule: Literal['S17','H17'] = 'S17'
    peek_rule: Literal['US','ENHC'] = 'US'
    push_22: bool = False
    double_allowed: Literal['any_two','10-11','9-11'] = 'any_two'
    double_after_split: bool = True
    allow_splits: bool = True #True on default
    max_splits: int = Field(default=3, ge=0, le=4)
    resplit_aces: bool = False
    hit_split_aces: bool = False
    allow_surrender: Literal['none','late','early'] = 'none'
    allow_insurance: bool = False
    insurance_payout: Literal['2:1'] = '2:1'


class Shoe(BaseModel):
    n_decks: int = 6                    #Game Setup
    penetration: float = 0.75           #Game Setup
    burn_cards: int = 1                 #Sim
    csm: bool = False                   #Sim
    shuffle_on_cutcard: bool = True     #Sim
    penetration_variance: float = 0.0   #Sim
    rng_seed: Optional[int] = None      #Sim

class Policy(BaseModel):
    # Player Policy
    name: Literal['random','basic','counting'] = 'basic'
    base_bet: float = 1.0
    bet_spread: Tuple[int,int] = (1,12)
    count_system: Literal['HiLo','KO','OmegaII','None'] = 'HiLo'
    true_count_method: Literal['decks','half_decks', "None"] = 'decks'
    deck_estimation: Literal['exact','rounded', "None"] = 'exact'
    index_deviations: Literal['None','Illustrious18'] = 'Illustrious18'
    insurance_by_count: bool = True
    wong_in_tc: Optional[int] = None
    wong_out_tc: Optional[int] = None
    seed: Optional[int] = None

class Table(BaseModel):
    n_spots: int = Field(default=5, ge=1, le=5)                 #Sim
    player_hands_per_round: int = Field(default=1, ge=0 , le=5) #Sim
    min_bet: float = 5.0                                        #Sim
    max_bet: float = 500.0                                      #Sim
    chip_denom: float = 5.0                                     #Sim
    dealer_misdeal_rate: float = 0.0                            #Sim

class Bankroll(BaseModel):
    initial: float = 1000.0                                     #Sim
    risk_of_ruin_target: Optional[float] = None                 #Sim
    kelly_fraction: float = 0.0                                 #Sim
    bet_rounding_to_unit: bool = True                           #Sim

class Simulation(BaseModel):
    #Sim Params
    episodes: int = 10_000_000
    seed: int = 42
    stop_on_ci: bool = True
    ci_level: float = 0.95
    ci_half_width_target: float = 0.005
    max_shoes: Optional[int] = None
    log_every: int = 10_000
    record_hand_histograms: bool = False
    parallel_workers: int = 0

class Config(BaseModel):
    config_version: int = 1
    rules: Rules = Rules()
    shoe: Shoe = Shoe()
    table: Table = Table()
    policy: Policy = Policy()
    bankroll: Bankroll = Bankroll()
    simulation: Simulation = Simulation()
    # reporting: Reporting = Reporting()
