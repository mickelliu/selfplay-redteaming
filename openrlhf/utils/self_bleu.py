import sacrebleu
import multiprocessing
from functools import partial
import numpy as np


def compute_self_bleu(responses, group_size=None) -> float:
    if len(responses) <= 1:
        return 0.0
    if group_size is None:
        group_size = len(responses)
    # Use compute_score from self_bleu.py
    # Group size of 5 is typical, but we can adjust based on needs
    # Returns score in range 0-100, so divide by 100 to match original scale
    return compute_score(
        responses, 
        group_size=group_size,
        score_type='self_bleu_score',
        num_workers=len(responses)//group_size,
        is_average_across_prompts=True
    ) / 100.0


def compute_bleu_for_one(candidate_idx, texts):
    """
    Computes BLEU score for one sentence against all others in its group.
    Args:
        candidate_idx: Index of the candidate sentence within its group.
        texts: List of text strings in the group.
    Returns:
        BLEU score for the candidate sentence.
    """
    candidate = texts[candidate_idx]
    references = texts[:candidate_idx] + texts[candidate_idx+1:]

    bleu_score = sacrebleu.sentence_bleu(
        candidate,
        references,
        smooth_method="exp",
        use_effective_order=True
    ).score
    return bleu_score


def compute_group_self_bleu(group_texts):
    """
    Computes Self-BLEU scores for a single group of texts.
    Args:
        group_texts: List of text strings.
    Returns:
        (scores, average): Tuple of individual scores and their average
    """
    group_size = len(group_texts)
    scores = [compute_bleu_for_one(i, group_texts)
              for i in range(group_size)]
    return scores, np.mean(scores)


def compute_all_groups_self_bleu_parallel(all_texts, group_size=5, num_workers=4):
    """
    Computes Self-BLEU scores for all groups in parallel.
    Args:
        all_texts: List of strings.
        group_size: Number of texts in each group.
        num_workers: Number of parallel processes.
    Returns:
        group_scores: List of (scores, average) tuples for each group.
    """
    # Validate input
    if len(all_texts) % group_size != 0:
        raise ValueError(
            f"Length of all_texts ({len(all_texts)}) must be divisible by group_size ({group_size})")

    # Split texts into groups
    groups = [all_texts[i:i+group_size]
              for i in range(0, len(all_texts), group_size)]

    # Create pool and compute scores
    with multiprocessing.Pool(num_workers) as pool:
        group_scores = pool.map(
            partial(compute_group_self_bleu),
            groups
        )
    return group_scores


def compute_score_all_types(all_texts, group_size, num_workers=4, is_average_across_prompts=True, score_types=['self_bleu_score']):
    # , 'pairwise_sentence_embedding_similarity'
    all_prompt_diversity_scores = {score_type: []
                                   for score_type in score_types}

    if 'self_bleu_score' in score_types:
        group_scores = compute_all_groups_self_bleu_parallel(
            all_texts,
            group_size=group_size,
            num_workers=num_workers
        )           
        for group_idx, (scores, avg) in enumerate(group_scores):
            all_prompt_diversity_scores['self_bleu_score'].extend(scores)

    if is_average_across_prompts:
        for score_type in score_types:
            all_prompt_diversity_scores[score_type] = np.mean(
                all_prompt_diversity_scores[score_type])
        return all_prompt_diversity_scores
    else:
        return all_prompt_diversity_scores


def compute_score(all_texts, group_size, score_type='self_bleu_score', num_workers=4, is_average_across_prompts=True):
    """
    Compute diversity scores for texts.
    Args:
        all_texts: List of strings
        group_size: Size of each group
        score_type: Type of score to compute ('self_bleu_score' supported)
        num_workers: Number of parallel processes
        is_average_across_prompts: Whether to return average or all scores
    Returns:
        float or list: Average score or list of scores
    """
    if score_type != 'self_bleu_score':
        raise ValueError(f"Unsupported score type: {score_type}")
    
    # Handle incomplete groups by either:
    # Option 1: Drop incomplete group
    num_complete_groups = len(all_texts) // group_size
    groups = [all_texts[i:i+group_size] 
             for i in range(0, num_complete_groups * group_size, group_size)]
    
    # Option 2: Pad the last group
    # groups = []
    # for i in range(0, len(all_texts), group_size):
    #     group = all_texts[i:i+group_size]
    #     if len(group) < group_size:
    #         # Pad with repeated texts from the same group
    #         group.extend(group[:group_size - len(group)])
    #     groups.append(group)
    
    with multiprocessing.Pool(num_workers) as pool:
        group_scores = pool.map(compute_group_self_bleu, groups)
        all_scores = [score for scores, _ in group_scores for score in scores]
    
    return np.mean(all_scores) if is_average_across_prompts else all_scores


def compute_bleu_score(original, revised) -> float:
    """
    Computes BLEU score between original prompt and revised prompt.
    
    Args:
        original: Original prompt as reference.
        revised: Revised prompt as candidate.
        
    Returns:
        BLEU score for the revised prompt against the original.
    """
    if not original or not revised:
        return 0.0
        
    # Use sacrebleu with the original as reference
    bleu_score = sacrebleu.sentence_bleu(
        revised,
        [original],
        smooth_method="exp",
        use_effective_order=True
    ).score
    
    # Return score normalized to 0-1 range
    return bleu_score / 100.0


# Example usage:
if __name__ == '__main__':
    # Example list of texts (multiple groups)
    all_texts = [
        # Group 1
        "The quick brown fox jumps over the lazy dog.",
        "A fast, brown fox leaps across a sleepy canine.",
        "The rapid fox swiftly jumps over the sluggish dog.",
        "An energetic fox hops over a dozing dog.",
        "A brown fox jumped over a lazy sleeping dog.",
        # Group 2
        "The cat sits on the windowsill watching birds.",
        "A feline perches near the window observing avians.",
        "The kitty is stationed at the window eyeing birds.",
        "A cat lounges by the windowsill looking at birds.",
        "The feline rests near the window watching flying creatures.",
    ] * (32)

    # group_scores = compute_all_groups_self_bleu_parallel(
    #     all_texts,
    #     group_size=5,  # Now configurable
    #     num_workers=4
    # )

    # # Print results
    # for group_idx, (scores, avg) in enumerate(group_scores):
    #     print(f"\nGroup {group_idx + 1}:")
    #     for i, score in enumerate(scores):
    #         print(f"Self-BLEU for sentence {i+1}: {score:.2f}")
    #     print(f"Group average Self-BLEU: {avg:.2f}")
    for group_size in range(2,11):
        print(f"Group size: {group_size}")
        print(compute_score(all_texts, group_size=group_size, num_workers=4, is_average_across_prompts=True))
        print()