from typing import List, Tuple

def to_binary_vector(event_number: int, indices: list[int]) -> list[int]:
    """
    Convert a list of indices to a binary vector.

    Args:
        event_number (int): Length of the vector (total events).
        indices (list[int]): 1-based indices to mark as 1.

    Returns:
        list[int]: Binary vector.
    """
    vec = [0] * event_number
    for i in indices:
        if 1 <= i <= event_number:
            vec[i-1] = 1  # 1-based index → 0-based list
    return vec


def culcurate_recall_precision_module(event_number: int, ground_truth: List[int], predict: List[int]) -> Tuple[float, float, bool]:
    """
    Calculate recall and precision for a single instance.

    Args:
        ground_truth (List[int]): List of ground truth frame indices.
        predict (List[int]): List of predicted frame indices.

    Returns:
        Tuple[float, float]: Recall and precision values.
    """
    gt_vec = to_binary_vector(event_number, ground_truth)
    pred_vec = to_binary_vector(event_number, predict)

    exact_match = False
    exact_match = (gt_vec == pred_vec)
    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0


    for i in range(event_number):
        if gt_vec[i] == 1 and pred_vec[i] == 1:
            true_positive += 1
        elif gt_vec[i] == 1 and pred_vec[i] == 0:
            false_negative += 1
        elif gt_vec[i] == 0 and pred_vec[i] == 1:
            false_positive += 1
        else:
            true_negative += 1

    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0

    return recall, precision, exact_match


def culcurate_recall_precision(event_number:List[int] , ground_truth: List[List[int]], predict: List[List[int]]) -> Tuple[float, float, int]:
    """
    Calculate recall and precision for event detection.

    Args:
        event_nubmer (List[int]): List of event numbers.
        ground_truth (List[List[int]]): List of ground truth events, each represented as a list of frame indices.
        predict (List[List[int]]): List of predicted events, each represented as a list of frame indices.

    Returns:
        Tuple[float, float]: Recall and precision values.
    """

    total_recall = 0.0
    total_precision = 0.0
    total_exact_match = 0

    for en, gt, pred in zip(event_number, ground_truth, predict):
        recall, precision, exact_match = culcurate_recall_precision_module(en, gt, pred)
        total_recall += recall
        total_precision += precision
        if exact_match:
            total_exact_match += 1

    return total_recall / len(event_number), total_precision / len(event_number), total_exact_match


def main():
    event_number = [5, 10, 8]
    ground_truth = [[1, 2], [3, 4, 5], []]
    predict = [[2, 3], [3, 4], []]
    recall, precision, exact_match = culcurate_recall_precision(event_number, ground_truth, predict)
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Exact Match: {exact_match}")


if __name__ == "__main__":
    main()
