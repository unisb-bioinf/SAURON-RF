# Author:Kerstin Lenhof
# Date: 18.10.2021
# Script for RF regression with additional variables used for prediction
# Needs a training and a test set as input, as well as the number of trees to be fit, the maximal depth of one tree, and the values for the additional variable to be considered for prediction
# Returns: - the decisions of each leaf node with respect to additional variable
#         - prediction for each training and test sample
import numpy
import math
import time


# from sklearn.datasets import make_regression
from collections import Counter
import scipy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import resample
from scipy.stats import pearsonr
from scipy.stats import spearmanr

def calculate_simple_weights(threshold, response_values):
    sum_weights_sensitive = 0.0
    sum_weights_resistant = 0.0

    for value in response_values:

        if value < threshold:

            sum_weights_sensitive = sum_weights_sensitive + 1

        else:
            sum_weights_resistant = sum_weights_resistant + 1

    weights = []

    factor = 0
    sens_less = True

    if sum_weights_sensitive < sum_weights_resistant:
        factor = (sum_weights_resistant/sum_weights_sensitive)
        sens_less = True

    else:
        factor = (sum_weights_sensitive/sum_weights_resistant)
        sens_less = False

    for value in response_values:
        if sens_less:
            if value < threshold:

                current_weight = factor

                weights.append(current_weight)

            else:

                current_weight = 1
                weights.append(current_weight)

        else:
            if value < threshold:

                current_weight =1

                weights.append(current_weight)

            else:

                current_weight = factor
                weights.append(current_weight)

    #print(sum_weights_sensitive)
    #print(sum_weights_resistant)
    #print(weights)
    return weights

def calculate_quadratic_weights(threshold, response_values):

    #print(response_values)
    #print(threshold)
    sum_weights_sensitive = 0.0
    sum_weights_resistant = 0.0

    for value in response_values:

        if value < threshold:

            sum_weights_sensitive = sum_weights_sensitive + pow(abs(value - threshold), 2)

        else:
            sum_weights_resistant = sum_weights_resistant + pow(abs(value - threshold), 2)

    weights = []

    for value in response_values:

        if value < threshold:

            current_weight = pow(abs(value - threshold), 2) / (2*sum_weights_sensitive)
            #print(current_weight)
            weights.append(current_weight)

        else:

            current_weight = pow(abs(value - threshold), 2) / (2*sum_weights_resistant)
            weights.append(current_weight)

    #print(sum_weights_sensitive)
    #print(sum_weights_resistant)
    #print(weights)
    return weights



def calculate_linear_weights(threshold, response_values):


    sum_weights_sensitive = 0.0
    sum_weights_resistant = 0.0

    for value in response_values:

        if value < threshold:

            sum_weights_sensitive = sum_weights_sensitive + abs(value - threshold)

        else:
            sum_weights_resistant = sum_weights_resistant + abs(value - threshold)

    weights = []

    for value in response_values:

        if value < threshold:

            current_weight = abs(value - threshold)/ (2*sum_weights_sensitive)
            #print(current_weight)
            weights.append(current_weight)

        else:

            current_weight = abs(value - threshold) / (2*sum_weights_resistant)
            weights.append(current_weight)

    #print(sum_weights_sensitive)
    #print(sum_weights_resistant)
    #print(weights)
    return weights

def calculate_weights(threshold, response_values, weighting_scheme):

    if weighting_scheme == "simple":
        return calculate_simple_weights(threshold, response_values)

    elif weighting_scheme == "linear":
        return calculate_linear_weights(threshold, response_values)

    elif weighting_scheme == "quadratic":
        return calculate_quadratic_weights(threshold, response_values)

    else:
        print("The given weighting scheme is not supported. Supported schemes are: simple, linear, quadratic.")
        print("Using weighting sheme simple instead")
        return calculate_simple_weights(threshold, response_values)

def convert_leaf_assignment_to_dict(leaf_assignment_train_data):

    tree_to_leaf_to_samples_dict = {}

    for tree_number in range(0, leaf_assignment_train_data.shape[1]):  # each tree
        for sample in range(0, leaf_assignment_train_data.shape[0]):  # each sample
            leaf_assignment = leaf_assignment_train_data[sample, tree_number]

            if tree_number not in tree_to_leaf_to_samples_dict:

                tree_to_leaf_to_samples_dict[tree_number] = {leaf_assignment: [sample]}
            else:
                if leaf_assignment not in tree_to_leaf_to_samples_dict[tree_number]:

                    tree_to_leaf_to_samples_dict[tree_number][leaf_assignment] = [sample]

                else:
                    if sample not in tree_to_leaf_to_samples_dict[tree_number][leaf_assignment]:

                        tree_to_leaf_to_samples_dict[tree_number][leaf_assignment].append(sample)

                    else:
                        print("Sample: " + str(sample) + " found twice for tree " + str(tree_number))

    return tree_to_leaf_to_samples_dict

def determine_percentage_of_classes(tree_to_leaf_to_samples_dict, class_assignment_samples):

    tree_to_leaf_to_percentage_class_dict = {}

    for tree_number in tree_to_leaf_to_samples_dict.keys():

        for leaf in tree_to_leaf_to_samples_dict[tree_number].keys():

            current_samples_in_leaf = tree_to_leaf_to_samples_dict[tree_number][leaf]

            class_count = {}

            for sample in current_samples_in_leaf:

                current_class = class_assignment_samples[sample]

                if current_class not in class_count:

                    class_count[current_class] = 1

                else:
                    class_count[current_class] = class_count[current_class] + 1

            class_percentage = {}

            for current_class in class_count:
                class_percentage[current_class] = class_count[current_class] / len(current_samples_in_leaf)

            if tree_number not in tree_to_leaf_to_percentage_class_dict:

                tree_to_leaf_to_percentage_class_dict[tree_number] = {leaf: class_percentage}

            else:

                if leaf not in tree_to_leaf_to_percentage_class_dict[tree_number]:
                    tree_to_leaf_to_percentage_class_dict[tree_number][leaf] = class_percentage

    return tree_to_leaf_to_percentage_class_dict

def determine_variance_of_samples(tree_to_leaf_to_samples_dict, y_train):

    tree_to_leaf_to_sample_variance_dict = {}

    for tree_number in tree_to_leaf_to_samples_dict.keys():

        for leaf in tree_to_leaf_to_samples_dict[tree_number].keys():

            current_samples_in_leaf = tree_to_leaf_to_samples_dict[tree_number][leaf]

            values_samples_in_leaf = []

            for sample in current_samples_in_leaf:

                current_value = y_train[sample]

                values_samples_in_leaf.append(current_value)

            leaf_variance = numpy.var(values_samples_in_leaf, ddof=1)

            if tree_number not in tree_to_leaf_to_sample_variance_dict:

                tree_to_leaf_to_sample_variance_dict[tree_number] = {leaf: leaf_variance}

            else:

                if leaf not in tree_to_leaf_to_sample_variance_dict[tree_number]:
                    tree_to_leaf_to_sample_variance_dict[tree_number][leaf] = leaf_variance

    return tree_to_leaf_to_sample_variance_dict


def determine_percentage_of_classes_sample_weights(tree_to_leaf_to_samples_dict, class_assignment_samples, weights_samples):

    tree_to_leaf_to_percentage_class_dict = {}

    for tree_number in tree_to_leaf_to_samples_dict.keys():

        for leaf in tree_to_leaf_to_samples_dict[tree_number].keys():

            current_samples_in_leaf = tree_to_leaf_to_samples_dict[tree_number][leaf]

            class_count = {}

            total_weight_count = 0
            for sample in current_samples_in_leaf:

                current_class = class_assignment_samples[sample]

                current_weight = weights_samples[sample]

                total_weight_count = total_weight_count + current_weight
                if current_class not in class_count:

                    class_count[current_class] = current_weight

                else:
                    class_count[current_class] = class_count[current_class] + current_weight

            class_percentage = {}

            for current_class in class_count:
                class_percentage[current_class] = class_count[current_class] / total_weight_count

            if tree_number not in tree_to_leaf_to_percentage_class_dict:

                tree_to_leaf_to_percentage_class_dict[tree_number] = {leaf: class_percentage}

            else:

                if leaf not in tree_to_leaf_to_percentage_class_dict[tree_number]:
                    tree_to_leaf_to_percentage_class_dict[tree_number][leaf] = class_percentage

    return tree_to_leaf_to_percentage_class_dict



def determine_majority_class(leaf_assignment_current_sample, tree_to_leaf_to_percentage_class_dict):

    class_votes = {}
    for tree in tree_to_leaf_to_percentage_class_dict.keys():

        leaf = leaf_assignment_current_sample[0][tree] # there is only a single sample and several trees


        current_best_percentage = 0.0
        best_prediction_class = None

        for prediction_class in tree_to_leaf_to_percentage_class_dict[tree][leaf]:

            percentage = tree_to_leaf_to_percentage_class_dict[tree][leaf][prediction_class]

            if percentage > current_best_percentage:

                best_prediction_class = prediction_class
                current_best_percentage = percentage


        if best_prediction_class not in class_votes:

            class_votes[best_prediction_class] = 1.0

        else:
            class_votes[best_prediction_class] = class_votes[best_prediction_class] + 1.0


    best_class = None
    best_votes = 0

    for prediction_class in class_votes.keys():
        if class_votes[prediction_class] > best_votes:

            best_votes = class_votes[prediction_class]
            best_class = prediction_class

    return [best_class, best_votes/sum(class_votes.values())]

def is_majority_in_current_tree(majority_class_sample, tree_distribution):


    best_class_of_tree = None
    best_percentage = 0.0
    for prediction_class in tree_distribution.keys():

        current_percentage = tree_distribution[prediction_class]

        if current_percentage > best_percentage:

            best_percentage = current_percentage
            best_class_of_tree = prediction_class


    if majority_class_sample == best_class_of_tree:

        return True

    else:
        return False


def calculate_prediction_single_sample_simple_average(sample_prediction, number_of_trees):


    used_trees = 0.0
    sum_trees = 0.0
    for current_tree in range(0, number_of_trees):

        current_tree_prediction = sample_prediction[current_tree]


        if not math.isnan(current_tree_prediction):
            used_trees = used_trees + 1
            sum_trees = sum_trees + current_tree_prediction

    if used_trees == 0.0:
        print("Something went wrong, no trees were used for a prediction")

    return sum_trees/used_trees


def calculate_prediction_single_sample_simple_weighted_average(sample_prediction, number_of_trees, tree_to_leaf_to_percentage_class_dict, leaf_assignment_current_sample, majority_class):


    weighted_average = 0.0
    sum_weights = 0.0
    for current_tree in range(0, number_of_trees):

        current_tree_prediction = sample_prediction[current_tree]

        if not math.isnan(current_tree_prediction):

            leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][current_tree]

            if majority_class in tree_to_leaf_to_percentage_class_dict[current_tree][leaf_assignment_current_sample_current_tree]:  # can be that a node is purely of the other class, then it is not included


                current_sample_weight = tree_to_leaf_to_percentage_class_dict[current_tree][leaf_assignment_current_sample_current_tree][majority_class]

                sum_weights = sum_weights + current_sample_weight

                weighted_average = weighted_average + current_sample_weight * current_tree_prediction

    if sum_weights == 0.0:
        print("Something went wrong, no trees were used for a prediction")

    return weighted_average/sum_weights


def calculate_final_predictions_binary_simple_weighted_average(X_test, predictions_samples, number_of_trees, tree_to_leaf_to_percentage_class_dict, model, classification_predictions):

    final_predictions_samples = []

    for sample in range(0, len(predictions_samples)):
        current_sample = X_test[sample, :]  # current sample
        leaf_assignment_current_sample = model.apply(current_sample.reshape(1, -1))  # leaf assignment of current sample for all trees
        majority_class = classification_predictions[sample]
        final_prediction_sample = calculate_prediction_single_sample_simple_weighted_average(predictions_samples[sample], number_of_trees, tree_to_leaf_to_percentage_class_dict, leaf_assignment_current_sample, majority_class)
        final_predictions_samples.append(final_prediction_sample)

    return final_predictions_samples


def calculate_final_predictions_binary_sensitive_weighted_average(X_test, predictions_samples, number_of_trees, tree_to_leaf_to_percentage_class_dict, model, classification_predictions):

    final_predictions_samples = []

    for sample in range(0, len(predictions_samples)):
        current_sample = X_test[sample, :]  # current sample
        leaf_assignment_current_sample = model.apply(current_sample.reshape(1, -1))  # leaf assignment of current sample for all trees
        majority_class = classification_predictions[sample]

        if int(majority_class) == 0: # just usual random forest average without weighting
            final_prediction_sample = calculate_prediction_single_sample_simple_average(predictions_samples[sample], number_of_trees)
            final_predictions_samples.append(final_prediction_sample)

        else:
            final_prediction_sample = calculate_prediction_single_sample_simple_weighted_average(predictions_samples[sample], number_of_trees, tree_to_leaf_to_percentage_class_dict, leaf_assignment_current_sample, majority_class)
            final_predictions_samples.append(final_prediction_sample)

    return final_predictions_samples




def get_non_zero_feature_importances(feature_importances):

    non_zero_indices = []
    for idx in range(0, feature_importances.shape[0]):

        if feature_importances[idx] != 0.0:
            non_zero_indices.append(idx)
    return non_zero_indices


#Does not work: somehow distance raises runtime warning that it encountered an invalid argument
def calculate_Mahalanobis_distance_to_leaf_samples(current_sample, tree_number, current_leaf_assignment,tree_to_leaf_to_samples_dict, X_train, indices_relevant_features):

    current_sample_only_important_features = [current_sample[idx] for idx in indices_relevant_features]

    current_samples_of_tree_in_leaf = tree_to_leaf_to_samples_dict[tree_number][current_leaf_assignment]


    current_samples_of_tree_in_leaf_values = [X_train[sample_idx] for sample_idx in current_samples_of_tree_in_leaf]

    current_samples_of_tree_in_leaf_values_important_features = []
    for train_sample in current_samples_of_tree_in_leaf_values:

        current_samples_of_tree_in_leaf_values_important_features.append([train_sample[idx] for idx in indices_relevant_features])


    mean_current_samples = numpy.mean(current_samples_of_tree_in_leaf_values_important_features, axis = 0) # column wise

    current_samples_of_tree_in_leaf_values_t = numpy.transpose(numpy.array(current_samples_of_tree_in_leaf_values_important_features))

    covariance = numpy.cov(current_samples_of_tree_in_leaf_values_t) #rows should be variables

    #print(covariance)
    try:
        icovariance = numpy.linalg.inv(covariance)
    except:
        icovariance = numpy.linalg.pinv(covariance, hermitian= True)
        
    mahalanobis_dist = scipy.spatial.distance.mahalanobis( u = numpy.array(current_sample_only_important_features), v = mean_current_samples, VI = icovariance)

    return mahalanobis_dist


def calculate_Euclidean_distance_to_leaf_samples_simple(current_sample, tree_number, current_leaf_assignment,tree_to_leaf_to_samples_dict, X_train, indices_relevant_features):


    distances = []
    current_samples_of_tree_in_leaf = tree_to_leaf_to_samples_dict[tree_number][current_leaf_assignment]

    for sample_idx in current_samples_of_tree_in_leaf:

        current_leaf_sample = X_train[sample_idx, :]

        current_leaf_sample_only_important_features = [current_leaf_sample[idx] for idx in indices_relevant_features]

        current_sample_only_important_features = [current_sample[idx] for idx in indices_relevant_features]

        distance_samples = numpy.linalg.norm(numpy.array(current_leaf_sample_only_important_features) - numpy.array(current_sample_only_important_features))

        distances.append(distance_samples)

    return numpy.mean(distances)


def calculate_Spearman_distance_to_leaf_samples_simple(current_sample, tree_number, current_leaf_assignment,tree_to_leaf_to_samples_dict, X_train, indices_relevant_features):


    distances = []
    current_samples_of_tree_in_leaf = tree_to_leaf_to_samples_dict[tree_number][current_leaf_assignment]

    for sample_idx in current_samples_of_tree_in_leaf:

        current_leaf_sample = X_train[sample_idx, :]

        current_leaf_sample_only_important_features = [current_leaf_sample[idx] for idx in indices_relevant_features]

        current_sample_only_important_features = [current_sample[idx] for idx in indices_relevant_features]

        #print(scipy.stats.spearmanr(numpy.array(current_leaf_sample_only_important_features), numpy.array(current_sample_only_important_features)))
        distance_samples = 1- abs(scipy.stats.spearmanr(numpy.array(current_leaf_sample_only_important_features), numpy.array(current_sample_only_important_features))[0])

        #print(distance_samples)
        distances.append(distance_samples)

    #print(numpy.mean(distances))
    return numpy.mean(distances)



def calculate_predictions_binary_simple_average_nearest_neighbours(X_test, number_of_trees_in_forest, number_of_samples, model, tree_to_leaf_to_percentage_class_dict, percentage_top_candidate_leafs, tree_to_leaf_to_sample_variance_dict, feature_importances, tree_to_leaf_to_samples_dict, X_train):


    indices_of_relevant_features = get_non_zero_feature_importances(feature_importances)


    predictions_samples = []
    classification_samples = []
    certainty_samples = []
    leaf_purity_samples =[]
    leaf_variance_samples = []
    distances_samples = []


    for sample in range(0, number_of_samples):

        current_sample = X_test[sample, :]  # current sample
        leaf_assignment_current_sample = model.apply(current_sample.reshape(1,-1))  # leaf assignment of current sample for all trees

        majority_class_for_sample_and_percentage = determine_majority_class(leaf_assignment_current_sample,
                                                             tree_to_leaf_to_percentage_class_dict)

        majority_class_for_sample = majority_class_for_sample_and_percentage[0]
        certainty_current_sample = majority_class_for_sample_and_percentage[1]
        classification_samples.append(majority_class_for_sample)
        certainty_samples.append(certainty_current_sample)
        distances_trees = []

        for tree in range(0, number_of_trees_in_forest):

            leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree] # is an ndarray

            is_majority_current_tree = is_majority_in_current_tree(majority_class_for_sample,tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree])

            #print(is_majority_current_tree)
            if is_majority_current_tree:
                #print("Hallo")
                distance_to_leaf_samples = calculate_Euclidean_distance_to_leaf_samples_simple(current_sample, tree, leaf_assignment_current_sample_current_tree, tree_to_leaf_to_samples_dict, X_train, indices_of_relevant_features)
                #distance_to_leaf_samples = calculate_Mahalanobis_distance_to_leaf_samples(current_sample, tree,  leaf_assignment_current_sample_current_tree, tree_to_leaf_to_samples_dict, X_train, indices_of_relevant_features)
                #distance_to_leaf_samples = calculate_Spearman_distance_to_leaf_samples_simple(current_sample, tree, leaf_assignment_current_sample_current_tree, tree_to_leaf_to_samples_dict, X_train, indices_of_relevant_features)

                distances_trees.append(distance_to_leaf_samples)

        #print("Done")
        distances_trees.sort() # ascending order
        #print(distances_trees)
        #print(percentage_top_candidate_leafs)
        #print(len(percentages_trees))
        index_of_top_leafs = round(percentage_top_candidate_leafs * len(distances_trees))

        distance_top_candidate_trees = distances_trees[index_of_top_leafs]

        #print(distance_top_candidate_trees)
        #print(percentage_top_candidate_trees)
        percentages_all_trees  = {}
        predictions_trees = {}
        variances_all_trees = {}
        distances_all_trees = {}


        at_least_one_tree_true = False
        at_least_one_tree_majority = False
        for tree in range(0, number_of_trees_in_forest):

            leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree]  # is an ndarray

            is_majority_current_tree = is_majority_in_current_tree(majority_class_for_sample,
                                                                   tree_to_leaf_to_percentage_class_dict[tree][
                                                                       leaf_assignment_current_sample_current_tree])

            if is_majority_current_tree:
                at_least_one_tree_majority = True
                percentage_current_tree = tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][majority_class_for_sample]


                if tree not in percentages_all_trees:
                    percentages_all_trees[tree] = percentage_current_tree

                else:
                    print("Found tree " + str(tree) + "twice")

                variance_current_tree = tree_to_leaf_to_sample_variance_dict[tree][leaf_assignment_current_sample_current_tree]

                if tree not in variances_all_trees:

                    variances_all_trees[tree] = variance_current_tree

                else:
                    print("Found tree " + str(tree) + "twice")


                distance_current_tree = calculate_Euclidean_distance_to_leaf_samples_simple(current_sample, tree, leaf_assignment_current_sample_current_tree, tree_to_leaf_to_samples_dict, X_train, indices_of_relevant_features)

                if tree not in distances_all_trees:

                    distances_all_trees[tree] = distance_current_tree

                else:
                    print("Found tree " + str(tree) + " twice")

                if distance_current_tree <= distance_top_candidate_trees:

                    at_least_one_tree_true = True
                    tree_x = model.estimators_[tree]
                    pred_single_tree = tree_x.predict(current_sample.reshape(1, -1))


                    if tree not in predictions_trees:

                        predictions_trees[tree] = pred_single_tree[0]  # is an array

                    else:
                        print("Found tree " + str(tree) + " twice for prediction.")


                else:
                    pred_single_tree = float("NaN")

                    if tree not in predictions_trees:
                        predictions_trees[
                            tree] = pred_single_tree  # was not included as a tree because majority was not the right class

                    else:
                        print("Found tree " + str(tree) + " twice for prediction.")
            else:

                pred_single_tree = float("NaN")

                if tree not in predictions_trees:
                    predictions_trees[tree] = pred_single_tree  # was not included as a tree because majority was not the right class

                else:
                    print("Found tree " + str(tree) + " twice for prediction.")

                if tree not in percentages_all_trees:
                    percentages_all_trees[tree] = float("NaN")

                else:
                    print("Found tree " + str(tree) + "twice")

                if tree not in variances_all_trees:
                    variances_all_trees[tree] = float("NaN")

                else:
                    print("Found tree " + str(tree) + "twice")

                if tree not in distances_all_trees:
                    distances_all_trees[tree] = float("NaN")
                else:
                    print("Found tree " + str(tree) + "twice")

        if not at_least_one_tree_true:
            print("Majority " + str(at_least_one_tree_majority))
            print("Tree " + str(at_least_one_tree_true))
        predictions_samples.append(predictions_trees)
        leaf_purity_samples.append(percentages_all_trees)
        leaf_variance_samples.append(variances_all_trees)
        distances_samples.append(distances_all_trees)


    return ([predictions_samples, classification_samples, certainty_samples, leaf_purity_samples, leaf_variance_samples, distances_samples])






def calculate_predictions_binary_simple_average_top_candidates(X_test, number_of_trees_in_forest, number_of_samples, model, tree_to_leaf_to_percentage_class_dict, percentage_top_candidate_leafs, tree_to_leaf_to_sample_variance_dict):

    predictions_samples = []
    classification_samples = []
    certainty_samples = []
    leaf_purity_samples =[]
    leaf_variance_samples = []


    for sample in range(0, number_of_samples):

        current_sample = X_test[sample, :]  # current sample
        leaf_assignment_current_sample = model.apply(current_sample.reshape(1,-1))  # leaf assignment of current sample for all trees

        majority_class_for_sample_and_percentage = determine_majority_class(leaf_assignment_current_sample,
                                                             tree_to_leaf_to_percentage_class_dict)

        majority_class_for_sample = majority_class_for_sample_and_percentage[0]
        certainty_current_sample = majority_class_for_sample_and_percentage[1]
        classification_samples.append(majority_class_for_sample)
        certainty_samples.append(certainty_current_sample)
        percentages_trees = []

        for tree in range(0, number_of_trees_in_forest):

            leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree] # is an ndarray

            is_majority_current_tree = is_majority_in_current_tree(majority_class_for_sample,tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree])

            if is_majority_current_tree:

                percentage_current_tree = tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][majority_class_for_sample]

                #print(percentage_current_tree)
                percentages_trees.append(percentage_current_tree)

        percentages_trees.sort(reverse = True) # descending order

        #print(percentage_top_candidate_leafs)
        #print(len(percentages_trees))
        index_of_top_leafs = round(percentage_top_candidate_leafs * len(percentages_trees))

        percentage_top_candidate_trees = percentages_trees[index_of_top_leafs]
        #print(percentage_top_candidate_trees)
        percentages_all_trees  = {}
        predictions_trees = {}
        variances_all_trees = {}

        for tree in range(0, number_of_trees_in_forest):

            leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree]  # is an ndarray

            is_majority_current_tree = is_majority_in_current_tree(majority_class_for_sample,
                                                                   tree_to_leaf_to_percentage_class_dict[tree][
                                                                       leaf_assignment_current_sample_current_tree])

            if is_majority_current_tree:

                percentage_current_tree = tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][majority_class_for_sample]


                if tree not in percentages_all_trees:
                    percentages_all_trees[tree] = percentage_current_tree

                else:
                    print("Found tree " + str(tree) + "twice")

                variance_current_tree = tree_to_leaf_to_sample_variance_dict[tree][leaf_assignment_current_sample_current_tree]

                if tree not in variances_all_trees:

                    variances_all_trees[tree] = variance_current_tree

                else:
                    print("Found tree " + str(tree) + "twice")

                if percentage_current_tree >= percentage_top_candidate_trees:

                    tree_x = model.estimators_[tree]
                    pred_single_tree = tree_x.predict(current_sample.reshape(1, -1))

                    if tree not in predictions_trees:

                        predictions_trees[tree] = pred_single_tree[0]  # is an array

                    else:
                        print("Found tree " + str(tree) + " twice for prediction.")


                else:
                    pred_single_tree = float("NaN")

                    if tree not in predictions_trees:
                        predictions_trees[
                            tree] = pred_single_tree  # was not included as a tree because majority was not the right class

                    else:
                        print("Found tree " + str(tree) + " twice for prediction.")
            else:

                pred_single_tree = float("NaN")

                if tree not in predictions_trees:
                    predictions_trees[tree] = pred_single_tree  # was not included as a tree because majority was not the right class

                else:
                    print("Found tree " + str(tree) + " twice for prediction.")

                if tree not in percentages_all_trees:
                    percentages_all_trees[tree] = float("NaN")

                else:
                    print("Found tree " + str(tree) + "twice")

                if tree not in variances_all_trees:
                    variances_all_trees[tree] = float("NaN")

                else:
                    print("Found tree " + str(tree) + "twice")

        predictions_samples.append(predictions_trees)
        leaf_purity_samples.append(percentages_all_trees)
        leaf_variance_samples.append(variances_all_trees)


    return ([predictions_samples, classification_samples, certainty_samples, leaf_purity_samples, leaf_variance_samples])

def calculate_predictions_binary_simple_average_top_candidates_variance(X_test, number_of_trees_in_forest, number_of_samples, model, tree_to_leaf_to_percentage_class_dict, percentage_top_candidate_leafs, tree_to_leaf_to_sample_variance_dict):

    predictions_samples = []
    classification_samples = []
    certainty_samples = []
    leaf_purity_samples = []
    leaf_variance_samples = []

    for sample in range(0, number_of_samples):

        current_sample = X_test[sample, :]  # current sample
        leaf_assignment_current_sample = model.apply(
            current_sample.reshape(1, -1))  # leaf assignment of current sample for all trees

        majority_class_for_sample_and_percentage = determine_majority_class(leaf_assignment_current_sample,
                                                                            tree_to_leaf_to_percentage_class_dict)

        majority_class_for_sample = majority_class_for_sample_and_percentage[0]
        certainty_current_sample = majority_class_for_sample_and_percentage[1]
        classification_samples.append(majority_class_for_sample)
        certainty_samples.append(certainty_current_sample)
        variances_trees = []
        for tree in range(0, number_of_trees_in_forest):

            leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree]  # is an ndarray

            is_majority_current_tree = is_majority_in_current_tree(majority_class_for_sample,
                                                                   tree_to_leaf_to_percentage_class_dict[tree][
                                                                       leaf_assignment_current_sample_current_tree])

            if is_majority_current_tree:
               variance_current_tree = tree_to_leaf_to_sample_variance_dict[tree][leaf_assignment_current_sample_current_tree]
               variances_trees.append(variance_current_tree)

        variances_trees.sort()  # increasingly sorted

        index_of_top_leafs = round(percentage_top_candidate_leafs * len(variances_trees))

        variance_top_candidate_tree = variances_trees[index_of_top_leafs]
        #print(variance_top_candidate_tree)
        percentages_all_trees = {}
        variances_all_trees = {}
        predictions_trees = {}
        for tree in range(0, number_of_trees_in_forest):

            leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree]  # is an ndarray

            is_majority_current_tree = is_majority_in_current_tree(majority_class_for_sample,
                                                                   tree_to_leaf_to_percentage_class_dict[tree][
                                                                       leaf_assignment_current_sample_current_tree])

            if is_majority_current_tree:

                percentage_current_tree = tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][majority_class_for_sample]

                if tree not in percentages_all_trees:
                    percentages_all_trees[tree] = percentage_current_tree

                else:
                    print("Found tree " + str(tree) + "twice")

                variance_current_tree = tree_to_leaf_to_sample_variance_dict[tree][leaf_assignment_current_sample_current_tree]


                if tree not in variances_all_trees:

                    variances_all_trees[tree] = variance_current_tree

                else:
                    print("Found tree " + str(tree) + "twice")

                if variance_current_tree <= variance_top_candidate_tree:

                    tree_x = model.estimators_[tree]
                    pred_single_tree = tree_x.predict(current_sample.reshape(1, -1))

                    #if if_test_set_used:
                    #    print("Prediction of tree " + str(tree) + " : " + str(pred_single_tree[0]))
                    if tree not in predictions_trees:

                        predictions_trees[tree] = pred_single_tree[0]  # is an array

                    else:
                        print("Found tree " + str(tree) + " twice for prediction.")


                else:
                    pred_single_tree = float("NaN")

                    if tree not in predictions_trees:
                        predictions_trees[tree] = pred_single_tree  # was not included as a tree because majority was not the right class

                    else:
                        print("Found tree " + str(tree) + " twice for prediction.")
            else:

                pred_single_tree = float("NaN")

                if tree not in predictions_trees:
                    predictions_trees[tree] = pred_single_tree  # was not included as a tree because majority was not the right class

                else:
                    print("Found tree " + str(tree) + " twice for prediction.")

                if tree not in percentages_all_trees:
                    percentages_all_trees[tree] = float("NaN")

                else:
                    print("Found tree " + str(tree) + "twice")

                if tree not in variances_all_trees:

                    variances_all_trees[tree] = float("NaN")
                else:
                    print("Found tree " + str(tree) + "twice")

        predictions_samples.append(predictions_trees)
        leaf_purity_samples.append(percentages_all_trees)
        leaf_variance_samples.append(variances_all_trees)

    return ([predictions_samples, classification_samples, certainty_samples, leaf_purity_samples, leaf_variance_samples])


def calculate_predictions_binary_simple_average_top_candidates_variance_sensitive(X_test, number_of_trees_in_forest, number_of_samples, model, tree_to_leaf_to_percentage_class_dict, percentage_top_candidate_leafs, tree_to_leaf_to_sample_variance_dict):

    predictions_samples = []
    classification_samples = []
    certainty_samples = []
    leaf_purity_samples = []
    leaf_variance_samples = []

    for sample in range(0, number_of_samples):

        current_sample = X_test[sample, :]  # current sample
        leaf_assignment_current_sample = model.apply(
            current_sample.reshape(1, -1))  # leaf assignment of current sample for all trees

        majority_class_for_sample_and_percentage = determine_majority_class(leaf_assignment_current_sample,
                                                                            tree_to_leaf_to_percentage_class_dict)

        majority_class_for_sample = majority_class_for_sample_and_percentage[0]
        certainty_current_sample = majority_class_for_sample_and_percentage[1]
        classification_samples.append(majority_class_for_sample)
        certainty_samples.append(certainty_current_sample)
        variances_trees = []

        percentages_all_trees = {}
        variances_all_trees = {}
        predictions_trees = {}
        if int(majority_class_for_sample) == 0:#resistant sample

            for tree in range(0, number_of_trees_in_forest):
                leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree]  # is an ndarray
                tree_x = model.estimators_[tree]
                pred_single_tree = tree_x.predict(current_sample.reshape(1, -1))

                if tree not in predictions_trees:

                    predictions_trees[tree] = pred_single_tree[0]  # is an array

                else:
                    print("Found tree " + str(tree) + " twice for prediction.")

                percentage_current_tree = 0
                if majority_class_for_sample in tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree]:
                    percentage_current_tree = tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][
                        majority_class_for_sample]

                if tree not in percentages_all_trees:
                    percentages_all_trees[tree] = percentage_current_tree

                else:
                    print("Found tree " + str(tree) + "twice")

                variance_current_tree = tree_to_leaf_to_sample_variance_dict[tree][leaf_assignment_current_sample_current_tree]

                if tree not in variances_all_trees:
                    variances_all_trees[tree] = variance_current_tree

                else:
                    print("Found tree " + str(tree) + "twice")

        else:
            for tree in range(0, number_of_trees_in_forest):

                leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree]  # is an ndarray

                is_majority_current_tree = is_majority_in_current_tree(majority_class_for_sample,
                                                                       tree_to_leaf_to_percentage_class_dict[tree][
                                                                           leaf_assignment_current_sample_current_tree])

                if is_majority_current_tree:
                   variance_current_tree = tree_to_leaf_to_sample_variance_dict[tree][leaf_assignment_current_sample_current_tree]
                   variances_trees.append(variance_current_tree)

            variances_trees.sort()  # increasingly sorted

            index_of_top_leafs = round(percentage_top_candidate_leafs * len(variances_trees))

            variance_top_candidate_tree = variances_trees[index_of_top_leafs]
            #print(variance_top_candidate_tree)

            for tree in range(0, number_of_trees_in_forest):

                leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree]  # is an ndarray

                is_majority_current_tree = is_majority_in_current_tree(majority_class_for_sample,
                                                                       tree_to_leaf_to_percentage_class_dict[tree][
                                                                           leaf_assignment_current_sample_current_tree])

                if is_majority_current_tree:

                    percentage_current_tree = tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][majority_class_for_sample]

                    if tree not in percentages_all_trees:
                        percentages_all_trees[tree] = percentage_current_tree

                    else:
                        print("Found tree " + str(tree) + "twice")

                    variance_current_tree = tree_to_leaf_to_sample_variance_dict[tree][leaf_assignment_current_sample_current_tree]


                    if tree not in variances_all_trees:

                        variances_all_trees[tree] = variance_current_tree

                    else:
                        print("Found tree " + str(tree) + "twice")

                    if variance_current_tree <= variance_top_candidate_tree:

                        tree_x = model.estimators_[tree]
                        pred_single_tree = tree_x.predict(current_sample.reshape(1, -1))

                        #if if_test_set_used:
                        #    print("Prediction of tree " + str(tree) + " : " + str(pred_single_tree[0]))
                        if tree not in predictions_trees:

                            predictions_trees[tree] = pred_single_tree[0]  # is an array

                        else:
                            print("Found tree " + str(tree) + " twice for prediction.")


                    else:
                        pred_single_tree = float("NaN")

                        if tree not in predictions_trees:
                            predictions_trees[tree] = pred_single_tree  # was not included as a tree because majority was not the right class

                        else:
                            print("Found tree " + str(tree) + " twice for prediction.")
                else:

                    pred_single_tree = float("NaN")

                    if tree not in predictions_trees:
                        predictions_trees[tree] = pred_single_tree  # was not included as a tree because majority was not the right class

                    else:
                        print("Found tree " + str(tree) + " twice for prediction.")

                    if tree not in percentages_all_trees:
                        percentages_all_trees[tree] = float("NaN")

                    else:
                        print("Found tree " + str(tree) + "twice")

                    if tree not in variances_all_trees:

                        variances_all_trees[tree] = float("NaN")
                    else:
                        print("Found tree " + str(tree) + "twice")

        predictions_samples.append(predictions_trees)
        leaf_purity_samples.append(percentages_all_trees)
        leaf_variance_samples.append(variances_all_trees)

    return ([predictions_samples, classification_samples, certainty_samples, leaf_purity_samples, leaf_variance_samples])




def calculate_predictions_binary_simple_average(X_test, number_of_trees_in_forest, number_of_samples, model, tree_to_leaf_to_percentage_class_dict, tree_to_leaf_to_sample_variance_dict):
    predictions_samples = []
    classification_samples = []
    certainty_samples = []
    leaf_purity_samples = []
    leaf_variance_samples = []

    for sample in range(0, number_of_samples):

        current_sample = X_test[sample, :]  # current sample
        leaf_assignment_current_sample = model.apply(current_sample.reshape(1,-1))  # leaf assignment of current sample for all trees

        majority_class_for_sample_and_percentage = determine_majority_class(leaf_assignment_current_sample,
                                                             tree_to_leaf_to_percentage_class_dict)

        majority_class_for_sample = majority_class_for_sample_and_percentage[0]
        certainty_current_sample = majority_class_for_sample_and_percentage[1]
        classification_samples.append(majority_class_for_sample)
        certainty_samples.append(certainty_current_sample)
        predictions_trees = {}
        percentages_all_trees = {}
        variances_all_trees = {}
        for tree in range(0, number_of_trees_in_forest):

            leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree] # is an ndarray

            is_majority_current_tree = is_majority_in_current_tree(majority_class_for_sample,tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree])

            if is_majority_current_tree:
                tree_x = model.estimators_[tree]
                pred_single_tree = tree_x.predict(current_sample.reshape(1, -1))


                if tree not in predictions_trees:

                    predictions_trees[tree] = pred_single_tree[0]  # is an array

                else:
                    print("Found tree " + str(tree) + " twice for prediction.")

                percentage_current_tree = tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][majority_class_for_sample]

                if tree not in percentages_all_trees:
                    percentages_all_trees[tree] = percentage_current_tree

                else:
                    print("Found tree " + str(tree) + "twice")


                variance_current_tree = tree_to_leaf_to_sample_variance_dict[tree][leaf_assignment_current_sample_current_tree]

                if tree not in variances_all_trees:
                    variances_all_trees[tree] = variance_current_tree

                else:
                    print("Found tree " + str(tree) + "twice")
            else:

                pred_single_tree = float("NaN")

                if tree not in predictions_trees:
                    predictions_trees[tree] = pred_single_tree  # was not included as a tree because majority was not the right class

                else:
                    print("Found tree " + str(tree) + " twice for prediction.")


                if tree not in percentages_all_trees:
                    percentages_all_trees[tree] = float("NaN")

                else:
                    print("Found tree " + str(tree) + "twice")

                if tree not in variances_all_trees:
                    variances_all_trees[tree] = float("NaN")

                else:
                    print("Found tree " + str(tree) + "twice")

        predictions_samples.append(predictions_trees)
        leaf_purity_samples.append(percentages_all_trees)
        leaf_variance_samples.append(variances_all_trees)

    return ([predictions_samples, classification_samples, certainty_samples, leaf_purity_samples, leaf_variance_samples])


def calculate_predictions_binary_sensitive_average(X_test, number_of_trees_in_forest, number_of_samples, model, tree_to_leaf_to_percentage_class_dict, tree_to_leaf_to_sample_variance_dict):
    predictions_samples = []
    classification_samples = []
    certainty_samples = []
    leaf_purity_samples = []
    leaf_variance_samples = []

    for sample in range(0, number_of_samples):

        current_sample = X_test[sample, :]  # current sample
        leaf_assignment_current_sample = model.apply(current_sample.reshape(1,-1))  # leaf assignment of current sample for all trees

        majority_class_for_sample_and_percentage = determine_majority_class(leaf_assignment_current_sample,
                                                             tree_to_leaf_to_percentage_class_dict)

        majority_class_for_sample = majority_class_for_sample_and_percentage[0]
        certainty_current_sample = majority_class_for_sample_and_percentage[1]
        classification_samples.append(majority_class_for_sample)
        certainty_samples.append(certainty_current_sample)
        predictions_trees = {}
        percentages_all_trees = {}
        variances_all_trees = {}

        if int(majority_class_for_sample) == 0: # this is hardcode shit and assumes that the class with 0 is the resistant and majority class

          for tree in range(0, number_of_trees_in_forest):
              leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree]  # is an ndarray
              tree_x = model.estimators_[tree]
              pred_single_tree = tree_x.predict(current_sample.reshape(1, -1))

              if tree not in predictions_trees:

                  predictions_trees[tree] = pred_single_tree[0]  # is an array

              else:
                  print("Found tree " + str(tree) + " twice for prediction.")

              percentage_current_tree = 0

              if majority_class_for_sample in tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree]:
                percentage_current_tree = tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][majority_class_for_sample]

              if tree not in percentages_all_trees:
                  percentages_all_trees[tree] = percentage_current_tree

              else:
                  print("Found tree " + str(tree) + "twice")

              variance_current_tree = tree_to_leaf_to_sample_variance_dict[tree][leaf_assignment_current_sample_current_tree]

              if tree not in variances_all_trees:
                  variances_all_trees[tree] = variance_current_tree

              else:
                  print("Found tree " + str(tree) + "twice")

        else:   #now the majority is the sensitive class and we would like to perform some other calculations
            for tree in range(0, number_of_trees_in_forest):

                leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree] # is an ndarray

                is_majority_current_tree = is_majority_in_current_tree(majority_class_for_sample,tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree])

                if is_majority_current_tree:
                    tree_x = model.estimators_[tree]
                    pred_single_tree = tree_x.predict(current_sample.reshape(1, -1))


                    if tree not in predictions_trees:

                        predictions_trees[tree] = pred_single_tree[0]  # is an array

                    else:
                        print("Found tree " + str(tree) + " twice for prediction.")

                    percentage_current_tree = tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][majority_class_for_sample]

                    if tree not in percentages_all_trees:
                        percentages_all_trees[tree] = percentage_current_tree

                    else:
                        print("Found tree " + str(tree) + "twice")


                    variance_current_tree = tree_to_leaf_to_sample_variance_dict[tree][leaf_assignment_current_sample_current_tree]

                    if tree not in variances_all_trees:
                        variances_all_trees[tree] = variance_current_tree

                    else:
                        print("Found tree " + str(tree) + "twice")
                else:

                    pred_single_tree = float("NaN")

                    if tree not in predictions_trees:
                        predictions_trees[tree] = pred_single_tree  # was not included as a tree because majority was not the right class

                    else:
                        print("Found tree " + str(tree) + " twice for prediction.")


                    if tree not in percentages_all_trees:
                        percentages_all_trees[tree] = float("NaN")

                    else:
                        print("Found tree " + str(tree) + "twice")

                    if tree not in variances_all_trees:
                        variances_all_trees[tree] = float("NaN")

                    else:
                        print("Found tree " + str(tree) + "twice")

        predictions_samples.append(predictions_trees)
        leaf_purity_samples.append(percentages_all_trees)
        leaf_variance_samples.append(variances_all_trees)

    return ([predictions_samples, classification_samples, certainty_samples, leaf_purity_samples, leaf_variance_samples])



def calculate_predictions_binary_simple_weighted_average(X_test, number_of_trees_in_forest, number_of_samples, model, tree_to_leaf_to_percentage_class_dict, tree_to_leaf_to_sample_variance_dict):

    predictions_samples = []
    classification_samples = []
    certainty_samples = []
    leaf_purity_samples = []
    leaf_variance_samples = []

    for sample in range(0, number_of_samples):

        current_sample = X_test[sample, :]  # current sample
        leaf_assignment_current_sample = model.apply(current_sample.reshape(1,-1))  # leaf assignment of current sample for all trees

        majority_class_for_sample_and_percentage = determine_majority_class(leaf_assignment_current_sample,
                                                             tree_to_leaf_to_percentage_class_dict)

        majority_class_for_sample = majority_class_for_sample_and_percentage[0]
        certainty_sample = majority_class_for_sample_and_percentage[1]

        classification_samples.append(majority_class_for_sample)
        certainty_samples.append(certainty_sample)
        predictions_trees = {}
        percentages_all_trees = {}
        variances_all_trees = {}
        for tree in range(0, number_of_trees_in_forest):

            tree_x = model.estimators_[tree]
            pred_single_tree = tree_x.predict(current_sample.reshape(1, -1))

            if tree not in predictions_trees:

                predictions_trees[tree] = pred_single_tree[0]  # is an array

            else:
                print("Found tree " + str(tree) + " twice for prediction.")

            leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree]  # is an ndarray
            percentage_current_tree = 0
            if majority_class_for_sample in tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree]:
                percentage_current_tree = tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][ majority_class_for_sample]

            if tree not in percentages_all_trees:
                percentages_all_trees[tree] = percentage_current_tree

            else:
                print("Found tree " + str(tree) + "twice")


            variance_current_tree = tree_to_leaf_to_sample_variance_dict[tree][leaf_assignment_current_sample_current_tree]

            if tree not in variances_all_trees:
                variances_all_trees[tree] = variance_current_tree
            else:
                print("Found tree " + str(tree) + "twice")

        leaf_purity_samples.append(percentages_all_trees)

        predictions_samples.append(predictions_trees)
        leaf_variance_samples.append(variances_all_trees)

    return ([predictions_samples, classification_samples, certainty_samples, leaf_purity_samples, leaf_variance_samples])



def calculate_final_predictions_binary_simple_average(predictions_samples, number_of_trees):

    final_predictions_samples = []
    for sample in range(0, len(predictions_samples)):
        final_prediction_sample = calculate_prediction_single_sample_simple_average(predictions_samples[sample], number_of_trees)
        final_predictions_samples.append(final_prediction_sample)

    return final_predictions_samples


def print_prediction_to_file(predictions_samples, final_predictions_samples, number_of_trees, number_of_samples,
                             output_sample_prediction_file, sample_names, certainty_samples, classification_prediction):
    with open(output_sample_prediction_file, "w", encoding="utf-8") as output_file:

        first_line = ["Prediction_Tree_" + str(tree) for tree in range(0, number_of_trees)]

        first_line.append("Total_Prediction")
        first_line.append("Certainty")
        first_line.append("Class_Prediction")

        all_samples_all_predictions = []
        for current_sample in range(0, number_of_samples):

            one_sample_all_predictions = []
            sample_name = sample_names[current_sample]
            one_sample_all_predictions.append(sample_name)

            for current_tree in range(0, number_of_trees):
                one_sample_single_tree_prediction = predictions_samples[current_sample][current_tree]

                one_sample_all_predictions.append(one_sample_single_tree_prediction)

            one_sample_all_predictions.append(final_predictions_samples[current_sample])
            one_sample_all_predictions.append(certainty_samples[current_sample])
            one_sample_all_predictions.append(classification_prediction[current_sample])

            all_samples_all_predictions.append(one_sample_all_predictions)

        output_file.write("\t".join(first_line) + "\n")

        for line in all_samples_all_predictions:
            output_file.write("\t".join([str(x) for x in line]) + "\n")


def print_leaf_purity_per_sample_to_file(leaf_purity_samples,  number_of_trees, number_of_samples,
                             output_leaf_purity_file, sample_names, purity_or_var = "purity"):
    with open(output_leaf_purity_file, "w", encoding="utf-8") as output_file:

        first_line = []
        if purity_or_var == "purity":
            first_line = ["Leaf_Purity_Tree_" + str(tree) for tree in range(0, number_of_trees)]
        elif purity_or_var == "variance":
            first_line = ["Leaf_Variance_Tree_" + str(tree) for tree in range(0, number_of_trees)]
        elif purity_or_var == "distance":
            first_line = ["Leaf_Distance_Tree_" + str(tree) for tree in range(0, number_of_trees)]

        else:
            first_line = ["Leaf_Purity_Tree_" + str(tree) for tree in range(0, number_of_trees)]


        all_samples_all_predictions = []
        for current_sample in range(0, number_of_samples):

            one_sample_all_predictions = []
            sample_name = sample_names[current_sample]
            one_sample_all_predictions.append(sample_name)

            for current_tree in range(0, number_of_trees):
                one_sample_single_tree_prediction = leaf_purity_samples[current_sample][current_tree]

                one_sample_all_predictions.append(one_sample_single_tree_prediction)

            all_samples_all_predictions.append(one_sample_all_predictions)

        output_file.write("\t".join(first_line) + "\n")

        for line in all_samples_all_predictions:
            output_file.write("\t".join([str(x) for x in line]) + "\n")



def print_error_to_file(filename, name_of_analysis, parameters, mse_included, classification_included,
                        predictions_samples, true_values, classification_prediction_samples,
                        true_classification_samples):
    output_line = [name_of_analysis, parameters]
    first_line = ["Name_of_Analysis", "Parameters"]

    #print("Prediction: ")
    #print(classification_prediction_samples)

    #print("Known values: ")
    #print(true_classification_samples)
    with open(filename, "w", encoding="utf-8") as output_file:
        if mse_included:
            mse = mean_squared_error(y_pred=predictions_samples, y_true=true_values)
            pcc = pearsonr(predictions_samples, true_values)

            output_line.append(mse)
            output_line.append(pcc)
            first_line.append("MSE")
            first_line.append("PCC")

        if classification_included:
            confusion_mtx = confusion_matrix(y_true=true_classification_samples,
                                             y_pred=classification_prediction_samples)
            tn = confusion_mtx[0][0]
            fp = confusion_mtx[0][1]
            fn = confusion_mtx[1][0]
            tp = confusion_mtx[1][1]

            accuracy = accuracy_score(y_true=true_classification_samples, y_pred=classification_prediction_samples,
                                      normalize=True)
            mcc = matthews_corrcoef(y_true=true_classification_samples, y_pred=classification_prediction_samples)

            output_line.append(tn)
            output_line.append(tp)
            output_line.append(fn)
            output_line.append(fp)
            output_line.append(accuracy)
            output_line.append(mcc)

            first_line.append("TN")
            first_line.append("TP")
            first_line.append("FN")
            first_line.append("FP")
            first_line.append("Accuracy")
            first_line.append("MCC")

        output_file.write("\t".join(first_line) + "\n")
        output_file.write("\t".join([str(x) for x in output_line]) + "\n")

def print_feature_importance_to_file(feature_imp_fit_model, feature_imp_output_file, feature_names):

    with open(feature_imp_output_file, "w", encoding = "utf-8") as feature_imp_output:
        #print(feature_imp_fit_model.shape[0])
        #print(len(feature_names))
        for feature_idx in range(0, feature_imp_fit_model.shape[0]):
            current_feature_name = feature_names[feature_idx]

            feature_importance = feature_imp_fit_model[feature_idx]

            feature_imp_output.write(current_feature_name + "\t" + str(feature_importance) + "\n")

def add_missing_samples(upsampled_samples, all_samples):

    for sample in all_samples:

        if sample not in upsampled_samples:

            upsampled_samples.append(sample)

    

def upsample_train_data_simple(X_train, y_train, class_assignment_samples_train, sample_names_train):

    class_counts = {}

    for sample_idx in range(0, len(class_assignment_samples_train)):

        class_sample = class_assignment_samples_train[sample_idx]

        if class_sample not in class_counts:

            class_counts[class_sample] = [1, [sample_idx]]

        else:
            class_counts[class_sample][0] = class_counts[class_sample][0] + 1
            class_counts[class_sample][1].append(sample_idx)


    max_class = ""
    max_class_count = 0

    for class_as in class_counts.keys():

        current_class_count = class_counts[class_as][0]

        if current_class_count > max_class_count:
            max_class_count = current_class_count
            max_class = class_as


    upsampled_dict = {}

    for class_as in class_counts.keys():

        current_class_count = class_counts[class_as][0]

        if current_class_count != max_class_count:
            current_samples =  class_counts[class_as][1]

            #print(max_class_count)
            upsampled_samples = resample(current_samples, replace = True, n_samples= max_class_count, random_state=223)

            add_missing_samples(upsampled_samples, current_samples) # no sample of the minority class should be left out
            
            if class_as not in upsampled_dict:
                upsampled_dict[class_as] = upsampled_samples

            else:
                print("Sampled class " + str(class_as) + " twice.")


        else:
            current_samples = class_counts[class_as][1]

            upsampled_samples = current_samples

            if class_as not in upsampled_dict:
                upsampled_dict[class_as] = upsampled_samples

            else:
                print("Sampled class " + str(class_as) + " twice.")

    #print(upsampled_dict)
    new_Xtrain = []
    new_ytrain = []
    new_sample_names = []
    new_class_assignment_samples_train =[]

    for class_as in upsampled_dict.keys():

        for sample_idx in upsampled_dict[class_as]:

            xtrain_sample = X_train[sample_idx]
            new_Xtrain.append(xtrain_sample)

            y_train_sample = y_train[sample_idx]
            new_ytrain.append(y_train_sample)

            new_sample_name = sample_names_train[sample_idx]
            new_sample_names.append(new_sample_name)

            new_class_assignment = class_assignment_samples_train[sample_idx]
            new_class_assignment_samples_train.append(new_class_assignment)

    return [new_Xtrain, new_ytrain, new_sample_names, new_class_assignment_samples_train]

def determine_sample_weights_minority_class(y_train, threshold, max_class, class_counts):#only binary data!

    weights = calculate_linear_weights(threshold, y_train) #could use quadratic weights -> would have to implement case distinction

    minority_class_weights = []

    for class_as in class_counts.keys():

        if not class_as == max_class:

            indices = class_counts[class_as][1]
            minority_class_weights = [(weights[idx]*2) for idx in indices]
            print(minority_class_weights)

    return minority_class_weights

def upsample_minority(current_samples, max_class_count, sample_weights_minority_class):

    upsampled_samples = []
    weight_id_counter = 0
    for sample_id in current_samples:

        sample_weight = sample_weights_minority_class[weight_id_counter]

        sample_frequency = int(round(max_class_count * sample_weight))

        if sample_frequency < 1:
            sample_frequency = 1
      
        upsampled_samples.extend([sample_id for i in range(0, sample_frequency)])

        weight_id_counter = weight_id_counter + 1


    return upsampled_samples

def upsample_train_data_proportional(X_train, y_train, class_assignment_samples_train, sample_names_train, threshold): # works only for binary data and one threshold

    class_counts = {}

    for sample_idx in range(0, len(class_assignment_samples_train)):

        class_sample = class_assignment_samples_train[sample_idx]

        if class_sample not in class_counts:

            class_counts[class_sample] = [1, [sample_idx]]

        else:
            class_counts[class_sample][0] = class_counts[class_sample][0] + 1
            class_counts[class_sample][1].append(sample_idx)

    max_class = ""
    max_class_count = 0

    for class_as in class_counts.keys():

        current_class_count = class_counts[class_as][0]

        if current_class_count > max_class_count:
            max_class_count = current_class_count
            max_class = class_as

    sample_weights_minority_class = determine_sample_weights_minority_class(y_train, threshold, max_class, class_counts) # weights in order of indices of minority class

    upsampled_dict = {}

    for class_as in class_counts.keys():

        current_class_count = class_counts[class_as][0]

        if current_class_count != max_class_count:
            current_samples = class_counts[class_as][1]


            # print(max_class_count)
            upsampled_samples = upsample_minority(current_samples, max_class_count, sample_weights_minority_class)

            if class_as not in upsampled_dict:
                upsampled_dict[class_as] = upsampled_samples

            else:
                print("Sampled class " + str(class_as) + " twice.")


        else:
            current_samples = class_counts[class_as][1]

            upsampled_samples = current_samples

            if class_as not in upsampled_dict:
                upsampled_dict[class_as] = upsampled_samples

            else:
                print("Sampled class " + str(class_as) + " twice.")

    new_Xtrain = []
    new_ytrain = []
    new_sample_names = []
    new_class_assignment_samples_train = []

    for class_as in upsampled_dict.keys():

        for sample_idx in upsampled_dict[class_as]:
            xtrain_sample = X_train[sample_idx]
            new_Xtrain.append(xtrain_sample)

            y_train_sample = y_train[sample_idx]
            new_ytrain.append(y_train_sample)

            new_sample_name = sample_names_train[sample_idx]
            new_sample_names.append(new_sample_name)

            new_class_assignment = class_assignment_samples_train[sample_idx]
            new_class_assignment_samples_train.append(new_class_assignment)

    return [new_Xtrain, new_ytrain, new_sample_names, new_class_assignment_samples_train]


def upsample_train_data(upsampling, X_train, y_train, class_assignment_samples_train, sample_names_train, threshold):

    if upsampling == "simple":

        return upsample_train_data_simple(X_train, y_train, class_assignment_samples_train, sample_names_train)

    elif upsampling == "proportional":

        return upsample_train_data_proportional(X_train, y_train, class_assignment_samples_train, sample_names_train, threshold)
    else:

        print("The given upsampling mode was not supported. Given mode is: " + upsampling + " Supported upsampling modes: simple")
        print("Using mode simple instead.")
        return upsample_train_data_simple(X_train, y_train, class_assignment_samples_train, sample_names_train)


def write_time_to_file(time_file, elapsed_time, name_of_analysis, parameters):

    with open(time_file, "w", encoding = "utf-8") as time_output:
        first_line = ["Name_of_Analysis", "Parameters", "Time"]

        time_output.write("\t".join(first_line) + "\n")
        time_output.write("\t".join([name_of_analysis, parameters, str(elapsed_time)]) + "\n")


def write_leaf_assignment_to_file(leaf_assignment_file_train, tree_to_leaf_to_samples_dict, sample_names_train):

    with open(leaf_assignment_file_train, "w", encoding = "utf-8") as leaf_assignment_output:

        unique_names = numpy.unique(numpy.array(sample_names_train))
        leaf_assignment_output.write("Tree" + "\t" + "\t".join([str(x) for x in unique_names]) + "\n")

        for tree in tree_to_leaf_to_samples_dict.keys():

            current_row = [str(tree)]
            for sample_name in unique_names:

                found_sample = False
                current_row_string = str(float("NaN"))

                for leaf in tree_to_leaf_to_samples_dict[tree].keys():

                    real_current_sample_names = [str(sample_names_train[idx_name]) for idx_name in tree_to_leaf_to_samples_dict[tree][leaf]]

                    if sample_name in real_current_sample_names:

                        if found_sample == True:
                            print("Found a sample twice in the same tree")

                        else:
                            found_sample = True
                            current_row_string = str(leaf)

                current_row.append(current_row_string)


            leaf_assignment_output.write("\t".join(current_row) + "\n")



def my_own_apply(model, X_train, number_of_trees_in_forest):

    #Have to draw bootstrap samples on my own using the same random instance as RandomForestRegressor uses

    #print("blablabla.... dubidibadu")
    tree_to_leaf_to_samples_dict = {}
    for tree_idx in range(0, number_of_trees_in_forest):

        random_instance = numpy.random.RandomState(model.estimators_[tree_idx].random_state)
        #print(str(tree_idx))
        #Tree depth
        #print("Depth tree " + str(tree_idx) + "\t" + str(model.estimators_[tree_idx].get_depth()) + "\n")
        #print("#Leaves trees " + str(tree_idx) + "\t" + str(model.estimators_[tree_idx].get_n_leaves()) + "\n")
        samples_current_tree = random_instance.randint(0, X_train.shape[0], X_train.shape[0])
        bootstrap_samples_current_tree = numpy.array([X_train[sample_idx] for sample_idx in samples_current_tree])

        current_tree_leaf_assignment = model.estimators_[tree_idx].apply(bootstrap_samples_current_tree)

        for sample in range(0, len(samples_current_tree)):

            leaf_of_current_sample = current_tree_leaf_assignment[sample]

            true_sample_id = samples_current_tree[sample]

            if not tree_idx in tree_to_leaf_to_samples_dict:

                tree_to_leaf_to_samples_dict[tree_idx] ={leaf_of_current_sample: [true_sample_id]}

            else:
                if leaf_of_current_sample not in tree_to_leaf_to_samples_dict[tree_idx]:

                    tree_to_leaf_to_samples_dict[tree_idx][leaf_of_current_sample] = [true_sample_id]


                else:
                    tree_to_leaf_to_samples_dict[tree_idx][leaf_of_current_sample].append(true_sample_id)

    return tree_to_leaf_to_samples_dict


def test_equality_by_prediction(tree_to_leaf_to_samples_dict, model, X_train, y_train, weights):

    #actual_predictions = model.predict(X_train)

    actual_leaf_assignments = model.apply(X_train)


    for sample in range(0, X_train.shape[0]):

        leaf_assignment_current_sample = actual_leaf_assignments[sample]

        pred_values = []
        for tree in range(0, len(tree_to_leaf_to_samples_dict.keys())):

            leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[tree]
            leaf_samples = tree_to_leaf_to_samples_dict[tree][leaf_assignment_current_sample_current_tree]


            if len(weights) == 0:
                pred_value = numpy.mean([y_train[sample_idx] for sample_idx in leaf_samples])

            else:
                current_weights = [weights[sample_idx] for sample_idx in leaf_samples]
                current_leaf_samples = [y_train[sample_idx] for sample_idx in leaf_samples]
                sum_weights = sum(current_weights)
                pred_value = (1/sum_weights) * (sum(numpy.multiply(current_leaf_samples, current_weights)))

            actual_pred = model.estimators_[tree].predict(X_train[sample, :].reshape(1, -1))[0]

            if not abs(pred_value-actual_pred) <= 0.0001:

                print("Actual prediction value: " + str(actual_pred))
                print("Our prediction value: " + str(pred_value))


def print_final_prediction_to_file(predictions_samples_train, original_sample_names_train, output_train_file):


    with open(output_train_file, "w", encoding = "utf-8") as output_train:

        output_train.write("Total_Prediction" + "\n")

        for sample_idx in range(0, len(original_sample_names_train)):

            current_sample_name = original_sample_names_train[sample_idx]
            current_prediction = predictions_samples_train[sample_idx]

            output_train.write(current_sample_name + "\t" + str(current_prediction) + "\n")


def print_train_samples_to_file(train_sample_names, train_sample_weights, sample_info_file):

    with open(sample_info_file, "w", encoding = "utf-8") as sample_info_output:

        sample_info_output.write("Sample_name" + "\t" + "Weight" + "\n")
        for train_sample in range(0, len(train_sample_names)):

            sample_info_output.write(str(train_sample_names[train_sample]) + "\t" + str(train_sample_weights[train_sample]) + "\n")


def print_train_samples_to_file_upsample(train_sample_names, sample_info_file):


    counts = Counter(train_sample_names)
    unique_elements = numpy.unique(numpy.array(train_sample_names))


    with open(sample_info_file, "w", encoding = "utf-8") as sample_info_output:

        sample_info_output.write("Sample_name" + "\t" + "Count" + "\n")

        for elem in unique_elements:
            sample_info_output.write(str(elem) + "\t" + str(counts[elem]) + "\n")
        




def train_and_test_rf_model(analysis_mode, X_train, X_test, y_train, y_test, sample_names_train, sample_names_test, output_sample_prediction_file_train, output_sample_prediction_file_test, train_error_file, test_error_file, min_number_of_samples_per_leaf, number_of_trees_in_forest, number_of_features_per_split, class_assignment_samples_train, class_assignment_samples_test, name_of_analysis, mse_included, classification_included, feature_imp_output_file, feature_names, threshold, upsampling, time_file, sample_weights_included, top_candidate_percentage, output_leaf_purity_file_train, output_leaf_purity_file_test, output_variance_file_train, output_variance_file_test, leaf_assignment_file_train, leaf_assignment_file_test, sample_info_file, debug_file):

    #global global_sample_names_train
    #global_sample_names_train= sample_names_train
    #print("Code is new")
    print("Your input parameters are: ")
    print("Analysis mode: " + analysis_mode)
    print("Number of trees: " + str(number_of_trees_in_forest))
    print("Min number of samples per leaf: " + str(min_number_of_samples_per_leaf))
    print("Number of features per split: " + str(number_of_features_per_split))

    print("The dimensions of your training matrix are: " )
    print("Number of rows (samples): " + str(X_train.shape[0]))
    print("Number of columns (features): " + str(X_train.shape[1]))

    original_Xtrain = X_train
    original_ytrain = y_train
    original_sample_names_train = sample_names_train
    original_class_assignment_samples_train = class_assignment_samples_train

    model = RandomForestRegressor(n_estimators=number_of_trees_in_forest, random_state= numpy.random.RandomState(2),
                                  min_samples_leaf=min_number_of_samples_per_leaf,
                                  max_features=number_of_features_per_split, bootstrap=True,
                                  oob_score=True, n_jobs=10)

    train_sample_weights = []
    if sample_weights_included in ["linear", "quadratic", "simple"]:

        if not math.isnan(threshold):
            train_sample_weights = numpy.array(calculate_weights(threshold, y_train, sample_weights_included))

            print_train_samples_to_file(original_sample_names_train, train_sample_weights, sample_info_file)
            start_time = time.perf_counter()
            model.fit(X_train, y_train, sample_weight=train_sample_weights)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
        else:

            print("No Threshold was given to calculate sample weights.")
            print("Please provide a threshold. Calculations aborted.")

    elif not upsampling == "":
        new_training_data = upsample_train_data(upsampling, X_train, y_train, class_assignment_samples_train, sample_names_train, float(threshold))
        up_Xtrain = new_training_data[0]
        up_ytrain = new_training_data[1]
        up_sample_names_train = new_training_data[2]
        new_class_assignment_samples_train = new_training_data[3]

        print_train_samples_to_file_upsample(up_sample_names_train, sample_info_file)

        X_train = numpy.array(up_Xtrain)
        y_train = numpy.array(up_ytrain)
        sample_names_train = up_sample_names_train
        class_assignment_samples_train = new_class_assignment_samples_train

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

    else:
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time


    number_of_samples_train = X_train.shape[0]
    number_of_samples_test = X_test.shape[0]

    feature_imp_fit_model = model.feature_importances_

    print_feature_importance_to_file(feature_imp_fit_model, feature_imp_output_file, feature_names)

    # get table with leaf-assignment for each train sample in each tree

    #Have to write this method on my own because apply does not know which samples were not in a bootstrap sample of a tree
    #This is the code that would do it if if the bootstrap samples were handled correctly
    #leaf_assignment_train_data = model.apply(X_train)
    # convert table to dict
    #tree_to_leaf_to_samples_dict = convert_leaf_assignment_to_dict(leaf_assignment_train_data)

    tree_to_leaf_to_samples_dict = my_own_apply(model, X_train, number_of_trees_in_forest)

    #Can be removed when it has been tested for several samples
    test_equality_by_prediction(tree_to_leaf_to_samples_dict, model, X_train, y_train, train_sample_weights)

    #global global_tree_to_leaf_to_samples_dict
    #global_tree_to_leaf_to_samples_dict = tree_to_leaf_to_samples_dict

    #global if_test_set_used
    #if_test_set_used = False
    write_leaf_assignment_to_file(leaf_assignment_file_train, tree_to_leaf_to_samples_dict, sample_names_train)


    leaf_assignment_all_test_data = model.apply(X_test)
    tree_to_leaf_to_samples_dict_all_test_data = convert_leaf_assignment_to_dict(leaf_assignment_all_test_data)
    write_leaf_assignment_to_file(leaf_assignment_file_test, tree_to_leaf_to_samples_dict_all_test_data, sample_names_test)

    # Determine purity (and majority class) of the leafs per tree
    # Determine percentage of samples in specific class per tree leaf of one tree

    tree_to_leaf_to_percentage_class_dict = {}

    if sample_weights_included in ["linear", "quadratic", "simple"]:
        tree_to_leaf_to_percentage_class_dict = determine_percentage_of_classes_sample_weights(tree_to_leaf_to_samples_dict,
                                                                                class_assignment_samples_train, train_sample_weights)
    else:
         tree_to_leaf_to_percentage_class_dict = determine_percentage_of_classes(tree_to_leaf_to_samples_dict, class_assignment_samples_train)



    #To understand the distribution of values in nodes, we use some variance information
    #This information can also be used to pick specific trees for predicting

    tree_to_leaf_to_sample_variance_dict = determine_variance_of_samples(tree_to_leaf_to_samples_dict, y_train)




    predictions_samples_train_and_classification = None
    predictions_samples_train = None
    classification_prediction_samples_train = None
    certainty_samples_train = None
    final_predictions_samples_train = None
    leaf_purity_samples_train = None
    leaf_variance_samples_train = None
    distances_samples_train = None

    predictions_samples_test_and_classification = None
    predictions_samples_test = None
    classification_prediction_samples_test = None
    certainty_samples_test = None
    final_predictions_samples_test = None
    leaf_purity_samples_test = None
    leaf_variance_samples_test = None
    distances_samples_test = None

    if analysis_mode == "binary_no_weights":
        # Predict with own routine
        predictions_samples_train_and_classification = calculate_predictions_binary_simple_average(original_Xtrain, number_of_trees_in_forest, original_Xtrain.shape[0], model, tree_to_leaf_to_percentage_class_dict, tree_to_leaf_to_sample_variance_dict)
        predictions_samples_train = predictions_samples_train_and_classification[0]
        classification_prediction_samples_train = predictions_samples_train_and_classification[1]
        certainty_samples_train = predictions_samples_train_and_classification[2]
        leaf_purity_samples_train = predictions_samples_train_and_classification[3]
        leaf_variance_samples_train = predictions_samples_train_and_classification[4]
        final_predictions_samples_train = calculate_final_predictions_binary_simple_average(predictions_samples_train, number_of_trees_in_forest)

        predictions_samples_test_and_classification = calculate_predictions_binary_simple_average(X_test, number_of_trees_in_forest, number_of_samples_test, model, tree_to_leaf_to_percentage_class_dict, tree_to_leaf_to_sample_variance_dict)
        predictions_samples_test = predictions_samples_test_and_classification[0]
        classification_prediction_samples_test = predictions_samples_test_and_classification[1]
        certainty_samples_test = predictions_samples_test_and_classification[2]
        leaf_purity_samples_test = predictions_samples_test_and_classification[3]
        leaf_variance_samples_test = predictions_samples_test_and_classification[4]
        final_predictions_samples_test = calculate_final_predictions_binary_simple_average(predictions_samples_test, number_of_trees_in_forest)
    elif analysis_mode == "binary_no_weights_sensitive":
        print("Analysis mode binary_no_weights_sensitive started")
        # Predict with own routine
        predictions_samples_train_and_classification = calculate_predictions_binary_sensitive_average(original_Xtrain,
                                                                                                   number_of_trees_in_forest,
                                                                                                   original_Xtrain.shape[
                                                                                                       0], model,
                                                                                                   tree_to_leaf_to_percentage_class_dict,
                                                                                                   tree_to_leaf_to_sample_variance_dict)
        predictions_samples_train = predictions_samples_train_and_classification[0]
        classification_prediction_samples_train = predictions_samples_train_and_classification[1]
        certainty_samples_train = predictions_samples_train_and_classification[2]
        leaf_purity_samples_train = predictions_samples_train_and_classification[3]
        leaf_variance_samples_train = predictions_samples_train_and_classification[4]
        final_predictions_samples_train = calculate_final_predictions_binary_simple_average(predictions_samples_train,
                                                                                            number_of_trees_in_forest)

        predictions_samples_test_and_classification = calculate_predictions_binary_sensitive_average(X_test,
                                                                                                  number_of_trees_in_forest,
                                                                                                  number_of_samples_test,
                                                                                                  model,
                                                                                                  tree_to_leaf_to_percentage_class_dict,
                                                                                                  tree_to_leaf_to_sample_variance_dict)
        predictions_samples_test = predictions_samples_test_and_classification[0]
        classification_prediction_samples_test = predictions_samples_test_and_classification[1]
        certainty_samples_test = predictions_samples_test_and_classification[2]
        leaf_purity_samples_test = predictions_samples_test_and_classification[3]
        leaf_variance_samples_test = predictions_samples_test_and_classification[4]
        final_predictions_samples_test = calculate_final_predictions_binary_simple_average(predictions_samples_test,
                                                                                           number_of_trees_in_forest)
    elif analysis_mode == "binary_weights":
        # Predict with own routine
        predictions_samples_train_and_classification = calculate_predictions_binary_simple_average(original_Xtrain,
                                                                                                   number_of_trees_in_forest,
                                                                                                   original_Xtrain.shape[0],
                                                                                                   model,
                                                                                                   tree_to_leaf_to_percentage_class_dict, tree_to_leaf_to_sample_variance_dict)
        predictions_samples_train = predictions_samples_train_and_classification[0]
        classification_prediction_samples_train = predictions_samples_train_and_classification[1]
        certainty_samples_train = predictions_samples_train_and_classification[2]
        leaf_purity_samples_train = predictions_samples_train_and_classification[3]
        leaf_variance_samples_train = predictions_samples_train_and_classification[4]
        final_predictions_samples_train = calculate_final_predictions_binary_simple_weighted_average(original_Xtrain, predictions_samples_train, number_of_trees_in_forest, tree_to_leaf_to_percentage_class_dict, model, classification_prediction_samples_train)

        predictions_samples_test_and_classification = calculate_predictions_binary_simple_average(X_test,
                                                                                                  number_of_trees_in_forest,
                                                                                                  number_of_samples_test,
                                                                                                  model,
                                                                                                  tree_to_leaf_to_percentage_class_dict, tree_to_leaf_to_sample_variance_dict)
        predictions_samples_test = predictions_samples_test_and_classification[0]
        classification_prediction_samples_test = predictions_samples_test_and_classification[1]
        certainty_samples_test = predictions_samples_test_and_classification[2]
        leaf_purity_samples_test = predictions_samples_test_and_classification[3]
        leaf_variance_samples_test = predictions_samples_test_and_classification[4]
        final_predictions_samples_test = calculate_final_predictions_binary_simple_weighted_average(X_test, predictions_samples_test, number_of_trees_in_forest, tree_to_leaf_to_percentage_class_dict, model, classification_prediction_samples_test)

    elif analysis_mode == "majority_weights":
        # Predict with own routine
        predictions_samples_train_and_classification = calculate_predictions_binary_simple_weighted_average(original_Xtrain,
                                                                                                   number_of_trees_in_forest,
                                                                                                   original_Xtrain.shape[0],
                                                                                                   model,
                                                                                                   tree_to_leaf_to_percentage_class_dict, tree_to_leaf_to_sample_variance_dict)
        predictions_samples_train = predictions_samples_train_and_classification[0]
        classification_prediction_samples_train = predictions_samples_train_and_classification[1]
        certainty_samples_train = predictions_samples_train_and_classification[2]
        leaf_purity_samples_train = predictions_samples_train_and_classification[3]
        leaf_variance_samples_train = predictions_samples_train_and_classification[4]
        final_predictions_samples_train = calculate_final_predictions_binary_simple_weighted_average(original_Xtrain,
                                                                                                     predictions_samples_train,
                                                                                                     number_of_trees_in_forest,
                                                                                                     tree_to_leaf_to_percentage_class_dict,
                                                                                                     model,
                                                                                                     classification_prediction_samples_train)

        predictions_samples_test_and_classification = calculate_predictions_binary_simple_weighted_average(X_test,
                                                                                                  number_of_trees_in_forest,
                                                                                                  number_of_samples_test,
                                                                                                  model,
                                                                                                  tree_to_leaf_to_percentage_class_dict, tree_to_leaf_to_sample_variance_dict)
        predictions_samples_test = predictions_samples_test_and_classification[0]
        classification_prediction_samples_test = predictions_samples_test_and_classification[1]
        certainty_samples_test = predictions_samples_test_and_classification[2]
        leaf_purity_samples_test = predictions_samples_test_and_classification[3]
        leaf_variance_samples_test = predictions_samples_test_and_classification[4]
        final_predictions_samples_test = calculate_final_predictions_binary_simple_weighted_average(X_test,
                                                                                                    predictions_samples_test,
                                                                                                    number_of_trees_in_forest,
                                                                                                    tree_to_leaf_to_percentage_class_dict,
                                                                                                    model,
                                                                                                    classification_prediction_samples_test)

    elif analysis_mode == "majority_weights_sensitive":
        # Predict with own routine
        predictions_samples_train_and_classification = calculate_predictions_binary_simple_weighted_average(original_Xtrain,
                                                                                                   number_of_trees_in_forest,
                                                                                                   original_Xtrain.shape[0],
                                                                                                   model,
                                                                                                   tree_to_leaf_to_percentage_class_dict, tree_to_leaf_to_sample_variance_dict)
        predictions_samples_train = predictions_samples_train_and_classification[0]
        classification_prediction_samples_train = predictions_samples_train_and_classification[1]
        certainty_samples_train = predictions_samples_train_and_classification[2]
        leaf_purity_samples_train = predictions_samples_train_and_classification[3]
        leaf_variance_samples_train = predictions_samples_train_and_classification[4]
        final_predictions_samples_train = calculate_final_predictions_binary_sensitive_weighted_average(original_Xtrain,
                                                                                                     predictions_samples_train,
                                                                                                     number_of_trees_in_forest,
                                                                                                     tree_to_leaf_to_percentage_class_dict,
                                                                                                     model,
                                                                                                     classification_prediction_samples_train)

        predictions_samples_test_and_classification = calculate_predictions_binary_simple_weighted_average(X_test,
                                                                                                  number_of_trees_in_forest,
                                                                                                  number_of_samples_test,
                                                                                                  model,
                                                                                                  tree_to_leaf_to_percentage_class_dict, tree_to_leaf_to_sample_variance_dict)
        predictions_samples_test = predictions_samples_test_and_classification[0]
        classification_prediction_samples_test = predictions_samples_test_and_classification[1]
        certainty_samples_test = predictions_samples_test_and_classification[2]
        leaf_purity_samples_test = predictions_samples_test_and_classification[3]
        leaf_variance_samples_test = predictions_samples_test_and_classification[4]
        final_predictions_samples_test = calculate_final_predictions_binary_sensitive_weighted_average(X_test,
                                                                                                    predictions_samples_test,
                                                                                                    number_of_trees_in_forest,
                                                                                                    tree_to_leaf_to_percentage_class_dict,
                                                                                                    model,
                                                                                                    classification_prediction_samples_test)

    elif analysis_mode == "binary_top_percentage":

        # Predict with own routine
        predictions_samples_train_and_classification = calculate_predictions_binary_simple_average_top_candidates(original_Xtrain,
                                                                                                   number_of_trees_in_forest,
                                                                                                   original_Xtrain.shape[
                                                                                                       0], model,
                                                                                                   tree_to_leaf_to_percentage_class_dict, top_candidate_percentage, tree_to_leaf_to_sample_variance_dict)
        predictions_samples_train = predictions_samples_train_and_classification[0]
        classification_prediction_samples_train = predictions_samples_train_and_classification[1]
        certainty_samples_train = predictions_samples_train_and_classification[2]
        leaf_purity_samples_train = predictions_samples_train_and_classification[3]
        leaf_variance_samples_train = predictions_samples_train_and_classification[4]
        final_predictions_samples_train = calculate_final_predictions_binary_simple_average(predictions_samples_train,
                                                                                            number_of_trees_in_forest)

        predictions_samples_test_and_classification = calculate_predictions_binary_simple_average_top_candidates(X_test,
                                                                                                  number_of_trees_in_forest,
                                                                                                  number_of_samples_test,
                                                                                                  model,
                                                                                                  tree_to_leaf_to_percentage_class_dict, top_candidate_percentage, tree_to_leaf_to_sample_variance_dict)
        predictions_samples_test = predictions_samples_test_and_classification[0]
        classification_prediction_samples_test = predictions_samples_test_and_classification[1]
        certainty_samples_test = predictions_samples_test_and_classification[2]
        leaf_purity_samples_test = predictions_samples_test_and_classification[3]
        leaf_variance_samples_test = predictions_samples_test_and_classification[4]
        final_predictions_samples_test = calculate_final_predictions_binary_simple_average(predictions_samples_test,
                                                                                           number_of_trees_in_forest)

    elif analysis_mode == "binary_top_variance":
        # Predict with own routine
        print("Binary top variance training predictions")
        predictions_samples_train_and_classification = calculate_predictions_binary_simple_average_top_candidates_variance(
            original_Xtrain,
            number_of_trees_in_forest,
            original_Xtrain.shape[
                0], model,
            tree_to_leaf_to_percentage_class_dict, top_candidate_percentage, tree_to_leaf_to_sample_variance_dict)
        predictions_samples_train = predictions_samples_train_and_classification[0]
        classification_prediction_samples_train = predictions_samples_train_and_classification[1]
        certainty_samples_train = predictions_samples_train_and_classification[2]
        leaf_purity_samples_train = predictions_samples_train_and_classification[3]
        leaf_variance_samples_train = predictions_samples_train_and_classification[4]
        final_predictions_samples_train = calculate_final_predictions_binary_simple_average(predictions_samples_train,
                                                                                            number_of_trees_in_forest)
        print("Binary top variance test predictions")
        #if_test_set_used = True
        predictions_samples_test_and_classification = calculate_predictions_binary_simple_average_top_candidates_variance(X_test,
                                                                                                                 number_of_trees_in_forest,
                                                                                                                 number_of_samples_test,
                                                                                                                 model,
                                                                                                                 tree_to_leaf_to_percentage_class_dict,
                                                                                                                 top_candidate_percentage, tree_to_leaf_to_sample_variance_dict)
        predictions_samples_test = predictions_samples_test_and_classification[0]
        classification_prediction_samples_test = predictions_samples_test_and_classification[1]
        certainty_samples_test = predictions_samples_test_and_classification[2]
        leaf_purity_samples_test = predictions_samples_test_and_classification[3]
        leaf_variance_samples_test = predictions_samples_test_and_classification[4]
        final_predictions_samples_test = calculate_final_predictions_binary_simple_average(predictions_samples_test,
                                                                                           number_of_trees_in_forest)

    elif analysis_mode == "binary_top_variance_sensitive":
        # Predict with own routine
        print("Binary top variance training predictions")
        predictions_samples_train_and_classification = calculate_predictions_binary_simple_average_top_candidates_variance_sensitive(
            original_Xtrain,
            number_of_trees_in_forest,
            original_Xtrain.shape[
                0], model,
            tree_to_leaf_to_percentage_class_dict, top_candidate_percentage, tree_to_leaf_to_sample_variance_dict)
        predictions_samples_train = predictions_samples_train_and_classification[0]
        classification_prediction_samples_train = predictions_samples_train_and_classification[1]
        certainty_samples_train = predictions_samples_train_and_classification[2]
        leaf_purity_samples_train = predictions_samples_train_and_classification[3]
        leaf_variance_samples_train = predictions_samples_train_and_classification[4]
        final_predictions_samples_train = calculate_final_predictions_binary_simple_average(predictions_samples_train,
                                                                                            number_of_trees_in_forest)
        print("Binary top variance test predictions")
        #if_test_set_used = True
        predictions_samples_test_and_classification = calculate_predictions_binary_simple_average_top_candidates_variance_sensitive(X_test,
                                                                                                                 number_of_trees_in_forest,
                                                                                                                 number_of_samples_test,
                                                                                                                 model,
                                                                                                                 tree_to_leaf_to_percentage_class_dict,
                                                                                                                 top_candidate_percentage, tree_to_leaf_to_sample_variance_dict)
        predictions_samples_test = predictions_samples_test_and_classification[0]
        classification_prediction_samples_test = predictions_samples_test_and_classification[1]
        certainty_samples_test = predictions_samples_test_and_classification[2]
        leaf_purity_samples_test = predictions_samples_test_and_classification[3]
        leaf_variance_samples_test = predictions_samples_test_and_classification[4]
        final_predictions_samples_test = calculate_final_predictions_binary_simple_average(predictions_samples_test,
                                                                                           number_of_trees_in_forest)

    elif analysis_mode == "binary_nearest_neighbours":

        # Predict with own routine
        predictions_samples_train_and_classification = calculate_predictions_binary_simple_average_nearest_neighbours(
            original_Xtrain,
            number_of_trees_in_forest,
            original_Xtrain.shape[
                0], model,
            tree_to_leaf_to_percentage_class_dict, top_candidate_percentage, tree_to_leaf_to_sample_variance_dict,  feature_imp_fit_model, tree_to_leaf_to_samples_dict, X_train)
        predictions_samples_train = predictions_samples_train_and_classification[0]
        classification_prediction_samples_train = predictions_samples_train_and_classification[1]
        certainty_samples_train = predictions_samples_train_and_classification[2]
        leaf_purity_samples_train = predictions_samples_train_and_classification[3]
        leaf_variance_samples_train = predictions_samples_train_and_classification[4]
        distances_samples_train = predictions_samples_train_and_classification[5]
        #print(distances_samples_train)
        final_predictions_samples_train = calculate_final_predictions_binary_simple_average(predictions_samples_train,
                                                                                            number_of_trees_in_forest)

        predictions_samples_test_and_classification = calculate_predictions_binary_simple_average_nearest_neighbours(
            X_test,
            number_of_trees_in_forest,
            number_of_samples_test,
            model,
            tree_to_leaf_to_percentage_class_dict,
            top_candidate_percentage, tree_to_leaf_to_sample_variance_dict, feature_imp_fit_model, tree_to_leaf_to_samples_dict, X_train)
        predictions_samples_test = predictions_samples_test_and_classification[0]
        classification_prediction_samples_test = predictions_samples_test_and_classification[1]
        certainty_samples_test = predictions_samples_test_and_classification[2]
        leaf_purity_samples_test = predictions_samples_test_and_classification[3]
        leaf_variance_samples_test = predictions_samples_test_and_classification[4]
        distances_samples_test = predictions_samples_test_and_classification[5]
        final_predictions_samples_test = calculate_final_predictions_binary_simple_average(predictions_samples_test,
                                                                                           number_of_trees_in_forest)


    elif analysis_mode == "vanilla": # just vanilla rf calculations for comparison purposes

        parameters = ["#Trees:" + str(number_of_trees_in_forest),
                      "#MinSamplesLeaf:" + str(min_number_of_samples_per_leaf),
                      "#FeaturesPerSplit:" + str(number_of_features_per_split)]

        predictions_samples_train = model.predict(original_Xtrain) #for vanilla no upsampling is conducted
        print_error_to_file(train_error_file, name_of_analysis, ",".join(parameters), mse_included, False,
                            predictions_samples_train, original_ytrain, [], original_class_assignment_samples_train)
        print_final_prediction_to_file(predictions_samples_train, original_sample_names_train, output_sample_prediction_file_train)

        predictions_samples_test = model.predict(X_test)
        print_error_to_file(test_error_file, name_of_analysis, ",".join(parameters), mse_included, False, predictions_samples_test, y_test, [], class_assignment_samples_test)
        print_final_prediction_to_file(predictions_samples_test, sample_names_test, output_sample_prediction_file_test)

        write_time_to_file(time_file, elapsed_time, name_of_analysis, ",".join(parameters))
        return

    else:
        print("The given analysis mode was not allowed. It was " + analysis_mode + ". Default mode majority_weights is being used now.")

        # Predict with own routine
        predictions_samples_train_and_classification = calculate_predictions_binary_simple_weighted_average(original_Xtrain,
                                                                                                            number_of_trees_in_forest,
                                                                                                            original_Xtrain.shape[0],
                                                                                                            model,
                                                                                                            tree_to_leaf_to_percentage_class_dict, tree_to_leaf_to_sample_variance_dict)
        predictions_samples_train = predictions_samples_train_and_classification[0]
        classification_prediction_samples_train = predictions_samples_train_and_classification[1]
        certainty_samples_train = predictions_samples_train_and_classification[2]
        leaf_purity_samples_train = predictions_samples_train_and_classification[3]
        leaf_variance_samples_train = predictions_samples_train_and_classification[4]
        final_predictions_samples_train = calculate_final_predictions_binary_simple_weighted_average(original_Xtrain,
                                                                                                     predictions_samples_train,
                                                                                                     number_of_trees_in_forest,
                                                                                                     tree_to_leaf_to_percentage_class_dict,
                                                                                                     model,
                                                                                                     classification_prediction_samples_train)

        predictions_samples_test_and_classification = calculate_predictions_binary_simple_weighted_average(X_test,
                                                                                                           number_of_trees_in_forest,
                                                                                                           number_of_samples_test,
                                                                                                           model,
                                                                                                           tree_to_leaf_to_percentage_class_dict, tree_to_leaf_to_sample_variance_dict)
        predictions_samples_test = predictions_samples_test_and_classification[0]
        classification_prediction_samples_test = predictions_samples_test_and_classification[1]
        certainty_samples_test = predictions_samples_test_and_classification[2]
        leaf_purity_samples_test = predictions_samples_test_and_classification[3]
        leaf_variance_samples_test = predictions_samples_test_and_classification[4]
        final_predictions_samples_test = calculate_final_predictions_binary_simple_weighted_average(X_test,
                                                                                                    predictions_samples_test,
                                                                                                    number_of_trees_in_forest,
                                                                                                    tree_to_leaf_to_percentage_class_dict,
                                                                                                    model,
                                                                                                    classification_prediction_samples_test)
    # Print solutions to file
    parameters = ["#Trees:" + str(number_of_trees_in_forest), "#MinSamplesLeaf:" + str(min_number_of_samples_per_leaf), "#FeaturesPerSplit:" + str(number_of_features_per_split)]

    write_time_to_file(time_file, elapsed_time, name_of_analysis, ",".join(parameters))
    print_prediction_to_file(predictions_samples_train, final_predictions_samples_train, number_of_trees_in_forest, original_Xtrain.shape[0], output_sample_prediction_file_train, original_sample_names_train, certainty_samples_train, classification_prediction_samples_train)
    print_error_to_file(train_error_file, name_of_analysis, ",".join(parameters), mse_included, classification_included, final_predictions_samples_train, original_ytrain, classification_prediction_samples_train, original_class_assignment_samples_train)
    print_leaf_purity_per_sample_to_file(leaf_purity_samples_train,  number_of_trees_in_forest, original_Xtrain.shape[0], output_leaf_purity_file_train, original_sample_names_train)
    print_leaf_purity_per_sample_to_file(leaf_variance_samples_train, number_of_trees_in_forest, original_Xtrain.shape[0], output_variance_file_train, original_sample_names_train, "variance")
    print_prediction_to_file(predictions_samples_test, final_predictions_samples_test, number_of_trees_in_forest,
                             number_of_samples_test, output_sample_prediction_file_test, sample_names_test, certainty_samples_test, classification_prediction_samples_test)
    print_error_to_file(test_error_file, name_of_analysis, ",".join(parameters), mse_included,
                        classification_included, final_predictions_samples_test, y_test,
                        classification_prediction_samples_test, class_assignment_samples_test)
    print_leaf_purity_per_sample_to_file(leaf_purity_samples_test,  number_of_trees_in_forest, number_of_samples_test, output_leaf_purity_file_test, sample_names_test)
    print_leaf_purity_per_sample_to_file(leaf_variance_samples_test, number_of_trees_in_forest, number_of_samples_test, output_variance_file_test, sample_names_test, "variance")

    #TODO: fix this
    if analysis_mode == "binary_nearest_neighbours":
        print_leaf_purity_per_sample_to_file(distances_samples_train, number_of_trees_in_forest,
                                             number_of_samples_test, output_leaf_purity_file_test, sample_names_test, "distance")
        print_leaf_purity_per_sample_to_file(distances_samples_test, number_of_trees_in_forest,
                                             number_of_samples_test, output_variance_file_test, sample_names_test,
                                             "distance")

    return

