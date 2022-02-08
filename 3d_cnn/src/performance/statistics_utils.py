import numpy as np


def get_max_F1(F1_score: list):
    if len(F1_score) > 0:
        max_F1 = np.max(F1_score)
        optimal_peak_number = np.min(np.where(F1_score == max_F1)[0])
    else:
        max_F1 = 0
        optimal_peak_number = np.nan
    return max_F1, optimal_peak_number


def get_clean_points_close2point(point, clean, radius):
    close_to_point = []
    distances = []
    for clean_p in clean:
        dist = np.linalg.norm(clean_p - point)
        if dist <= radius:
            close_to_point.append(clean_p)
            distances.append(dist)
    close_to_point = [tuple(p) for p in close_to_point]
    return close_to_point, distances


def precision_recall_calculator(predicted_coordinates: np.array or list,
                                value_predicted: list,
                                true_coordinates: np.array or list,
                                radius: float):
    true_coordinates = list(true_coordinates)
    predicted_coordinates = list(predicted_coordinates)
    detected_true = list()
    predicted_true_positives = list()
    predicted_redundant = list()
    value_predicted_true_positives = list()
    value_predicted_redundant = list()
    precision = list()
    recall = list()
    total_true_points = len(true_coordinates)
    assert total_true_points > 0, "one empty list here!"
    if len(predicted_coordinates) == 0:
        print("No predicted points")
        precision = []
        recall = []
        detected_true = []
        predicted_true_positives = []
        predicted_false_positives = []
        value_predicted_true_positives = []
        value_predicted_false_positives = []
        predicted_redundant = []
        value_predicted_redundant = []
        false_negatives = total_true_points
    else:
        predicted_false_positives = list()
        value_predicted_false_positives = list()
        for value, point in zip(value_predicted, predicted_coordinates):
            close_to_point, distances = get_clean_points_close2point(
                point,
                true_coordinates,
                radius
            )
            if len(close_to_point) > 0:
                flag = "true_positive_candidate"
                flag_tmp = "not_redundant_yet"
                for dist, clean_p in sorted(zip(distances, close_to_point)):
                    if flag == "true_positive_candidate":
                        if tuple(clean_p) not in detected_true:
                            detected_true.append(tuple(clean_p))
                            flag = "true_positive"
                        else:
                            flag_tmp = "redundant_candidate"
                    # else:
                    # print(point, "is already flagged as true positive")
                if flag == "true_positive":
                    predicted_true_positives.append(tuple(point))
                    value_predicted_true_positives.append(value)
                elif flag == "true_positive_candidate" and \
                        flag_tmp == "redundant_candidate":
                    predicted_redundant.append(tuple(point))
                    value_predicted_redundant.append(value)
                else:
                    print("This should never happen!")
            else:
                predicted_false_positives.append(tuple(point))
                value_predicted_false_positives.append(value)
            true_positives_total = len(predicted_true_positives)
            false_positives_total = len(predicted_false_positives)
            total_current_predicted_points = true_positives_total + \
                                             false_positives_total
            precision.append(true_positives_total / total_current_predicted_points)
            recall.append(true_positives_total)
        false_negatives = [point for point in true_coordinates if tuple(point) not in detected_true]
        N_inv = 1 / total_true_points
        recall = np.array(recall) * N_inv
        recall = list(recall)
    return precision, recall, detected_true, predicted_true_positives, \
           predicted_false_positives, value_predicted_true_positives, \
           value_predicted_false_positives, false_negatives, predicted_redundant, \
           value_predicted_redundant


def f1_score_calculator(precision: list, recall: list):
    f1_score = []
    if len(precision) == 0:
        print("No precision and recall")
        f1_score = [0]
    else:
        for p, r in zip(precision, recall):
            if p + r != 0:
                f1_score.append(2 * p * r / float(p + r))
            else:
                f1_score.append(0)
    return f1_score


def quadrature_calculator(x_points: list, y_points: list) -> float:
    """
    This function computes an approximate value of the integral of a real
    function f in an interval, using the trapezoidal rule.

    Input:
    x_points: is a list of points in the x axis (not necessarily ordered)
    y_points: is a list of points, such that y_points[n] = f(x_points[n]) for
    each n.
    """
    # sorted_y = [p for _, p in sorted(zip(x_points, y_points))]
    sorted_y = [p for _, p in
                sorted(list(zip(x_points, y_points)), key=lambda x: x[0])]
    n = len(y_points)
    sorted_x = sorted(x_points)

    trapezoidal_rule = [
        0.5 * (sorted_x[n + 1] - sorted_x[n]) * (sorted_y[n + 1] + sorted_y[n])
        for n in range(n - 1)]

    return float(np.sum(trapezoidal_rule))


def pr_auc_score(precision: list, recall: list) -> float:
    """
    This function computes an approximate value to the area
    under the precision-recall (PR) curve.
    """
    return quadrature_calculator(recall, precision)
