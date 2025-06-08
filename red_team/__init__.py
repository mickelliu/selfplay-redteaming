from enum import Enum, auto

class GameOutcome(Enum):

    # Attacker
    ATTACKER_FAITHFUL_REVISION = "attacker_faithful_revision"
    ATTACKER_UNFAITHFUL_REVISION = "attacker_unfaithful_revision"
    ATTACKER_WIN_SUCCESSFUL_HARMFUL_REQUEST = "attacker_win_successful_harmful_request"
    ATTACKER_WIN_SUCCESSFUL_BENIGN_REQUEST = "attacker_win_successful_benign_request"
    ATTACKER_LOSE_FAILED_HARMFUL_REQUEST = "attacker_lose_failed_harmful_request"
    ATTACKER_LOSE_FAILED_BENIGN_REQUEST = "attacker_lose_failed_benign_request"
    ATTACKER_WIN_BENIGN_REFUSAL = "attacker_win_benign_refusal"
    ATTACKER_LOSE_BENIGN_COMPLIANCE = "attacker_lose_benign_compliance"
    ATTACKER_WIN_HARMFUL_COMPLIANCE = "attacker_win_harmful_compliance"
    ATTACKER_LOSE_HARMFUL_REFUSAL = "attacker_lose_harmful_refusal"

    # Defender
    DEFENDER_WIN_SUCCESSFUL_DEFENSE = "defender_win_successful_defense"
    DEFENDER_LOSE_BROKEN_DEFENSE = "defender_lose_broken_defense" # This was different: "defender_lose_unsuccessful_defense"
    DEFENDER_WIN_CORRECT_REFUSAL = "defender_win_correct_refusal"
    DEFENDER_LOSE_WRONG_REFUSAL = "defender_lose_wrong_refusal"

    TIE = "tie"