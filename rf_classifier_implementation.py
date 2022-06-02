# Author:Kerstin Lenhof
# Date: 4.4.2022
# Script for RF classification with additional variables used for prediction
# Needs a training and a test set as input, as well as the number of trees to be fit, the maximal depth of one tree, and the values for the additional variable to be considered for prediction
# Returns: - the decisions of each leaf node with respect to additional variable
#         - prediction for each training and test sample
import numpy
import math
import time

# from sklearn.datasets import make_regression
from collections import Counter
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import resample
from scipy.stats import pearsonr
from scipy.stats import spearmanr

import harf_implementation as harf

def determine_tree_to_class_counter_dict(tree_to_leaf_to_samples_dict, class_assignment_samples):
    tree_to_class_counter_dict = {}

    for tree in tree_to_leaf_to_samples_dict.keys():

        current_tree_samples = []
        for leaf in tree_to_leaf_to_samples_dict[tree].keys():

            for sample in tree_to_leaf_to_samples_dict[tree][leaf]:
                current_tree_samples.append(sample)

        current_class_assignment_samples = []
        for sample in current_tree_samples:
            current_class_assignment_samples.append(class_assignment_samples[sample])

        number_of_samples_per_class_per_tree = Counter(current_class_assignment_samples)

        if not tree in tree_to_class_counter_dict:
            tree_to_class_counter_dict[tree] = number_of_samples_per_class_per_tree

        else:
            print("Found tree " + tree + " twice.")
    return tree_to_class_counter_dict

def determine_percentage_of_classes_sample_weights_and_class_weights_subsample(tree_to_leaf_to_samples_dict, class_assignment_samples, weights_samples):


    tree_to_class_counter_dict = determine_tree_to_class_counter_dict(tree_to_leaf_to_samples_dict, class_assignment_samples)


    tree_to_leaf_to_percentage_class_dict = {}


    for tree_number in tree_to_leaf_to_samples_dict.keys():

        class_counter_current_tree = tree_to_class_counter_dict[tree_number]

        if len(class_counter_current_tree.keys()) == 1: # only one class, according to documentation the class weight will be 1
            class_weight0 = 1.0
            class_weight1 = 1.0

        else:
            class_weight0 = len(class_assignment_samples) / (
                    len(class_counter_current_tree.keys()) * class_counter_current_tree["0"])
            class_weight1 = len(class_assignment_samples) / (
                    len(class_counter_current_tree.keys()) * class_counter_current_tree["1"])

        current_tree_weights = {}
        for leaf in tree_to_leaf_to_samples_dict[tree_number].keys():

            current_samples_in_leaf = tree_to_leaf_to_samples_dict[tree_number][leaf]

            class_count = {}

            total_weight_count = 0
            for sample in current_samples_in_leaf:


                current_class = class_assignment_samples[sample]

                current_weight = weights_samples[sample]

                if current_class == "0":#TODO: hardcode for two classes
                    current_weight = current_weight * class_weight0

                else:
                    current_weight = current_weight * class_weight1

                if leaf not in current_tree_weights:

                    current_tree_weights[leaf] = {sample : current_weight}

                else:
                    current_tree_weights[leaf][sample] = current_weight
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


def calculate_adjusted_weights(weights_samples, class_assignment_samples):
    number_of_samples_per_class = Counter(class_assignment_samples)

    class_weight0 = 1.0
    class_weight1 = 1.0
    if len(number_of_samples_per_class.keys()) != 1:

        class_weight0 = float(len(class_assignment_samples)) / (
                    len(Counter(class_assignment_samples).keys()) * number_of_samples_per_class["0"])
        class_weight1 = float(len(class_assignment_samples)) / (
                    len(Counter(class_assignment_samples).keys()) * number_of_samples_per_class["1"])

    sum_sens = 0.0
    sum_res = 0.0
    new_combined_weights = []
    for sample in range(len(class_assignment_samples)):

        sample_class = class_assignment_samples[sample]
        if sample_class == "0":
            new_weight = weights_samples[sample] * class_weight0
            new_combined_weights.append(new_weight)
            sum_res = sum_res + new_weight
        elif sample_class == "1":
            new_weight = weights_samples[sample] * class_weight1
            new_combined_weights.append(new_weight)
            sum_sens = sum_sens + new_weight
        else:
            print("wrong class detected: " + sample_class)

    #true_sample_weights = []
    #for sample in range(len(class_assignment_samples)):

     #   sample_class = class_assignment_samples[sample]

      #  if sample_class == "0":

       #     new_weight = new_combined_weights[sample] / sum_res
        #    true_sample_weights.append(new_weight)
     #   else:
      #      new_weight = new_combined_weights[sample] / sum_sens
       #     true_sample_weights.append(new_weight)

    return new_combined_weights

def determine_percentage_of_classes_sample_weights_and_class_weights(tree_to_leaf_to_samples_dict, class_assignment_samples, weights_samples):
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





#Take all samples regardless of class
def calculate_continuous_prediction(leaf_assignment_current_sample_current_tree, tree, cont_y_test, tree_to_leaf_to_samples_dict, sample_weights):

    all_samples_in_current_leaf = tree_to_leaf_to_samples_dict[tree][leaf_assignment_current_sample_current_tree]

    total_sum = 0.0
    total_weight_sum = 0.0
    for sample in all_samples_in_current_leaf:

        y_value = cont_y_test[sample]

        current_weight = sample_weights[sample]

        total_sum = total_sum + y_value * current_weight
        total_weight_sum = total_weight_sum + current_weight

    return total_sum/total_weight_sum




def calculate_continuous_prediction_subsample(leaf_assignment_current_sample_current_tree, tree, cont_y_test, tree_to_leaf_to_samples_dict, sample_weights, class_assignment_samples, class_weight0, class_weight1):
    all_samples_in_current_leaf = tree_to_leaf_to_samples_dict[tree][leaf_assignment_current_sample_current_tree]

    total_sum = 0.0
    total_weight_sum = 0.0
    for sample in all_samples_in_current_leaf:
        y_value = cont_y_test[sample]

        current_weight = sample_weights[sample]

        current_class = class_assignment_samples[sample]
        if current_class == "0":

            current_weight = current_weight * class_weight0

        else:
            current_weight = current_weight * class_weight1

        total_sum = total_sum + y_value * current_weight
        total_weight_sum = total_weight_sum + current_weight

    return total_sum / total_weight_sum


def calculate_predictions_binary_simple_average_classification(X_test, number_of_trees_in_forest, number_of_samples, model,
                                                tree_to_leaf_to_percentage_class_dict,
                                                tree_to_leaf_to_sample_variance_dict, cont_y_test, tree_to_leaf_to_samples_dict, train_sample_weights):



    predictions_samples = []
    classification_samples = []
    certainty_samples = []
    leaf_purity_samples = []
    leaf_variance_samples = []

    for sample in range(0, number_of_samples):

        current_sample = X_test[sample, :]  # current sample
        leaf_assignment_current_sample = model.apply(
            current_sample.reshape(1, -1))  # leaf assignment of current sample for all trees

        class_prediction_forest = model.predict(current_sample.reshape(1, -1))[0]

        class_predictions_trees = {}
        predictions_trees = {}
        variances_all_trees = {}
        percentages_all_trees = {}

        class_proba = 0.0
        for tree in range(0, number_of_trees_in_forest):

            leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree]

            tree_x = model.estimators_[tree]
            pred_single_tree = str(int(tree_x.predict(current_sample.reshape(1, -1))[0]))

            #print("Single tree: " + str(pred_single_tree) + "\n")
            #print("Class prediction: " + class_prediction_forest + "\n")
            if tree not in class_predictions_trees:
                class_predictions_trees[tree] = pred_single_tree

            else:
                print("Found tree " + str(tree) + " twice")

            if pred_single_tree == class_prediction_forest:
                class_proba = class_proba + 1.0
                percentage_current_tree = tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][class_prediction_forest]
                variance_current_tree = tree_to_leaf_to_sample_variance_dict[tree][leaf_assignment_current_sample_current_tree]

                cont_pred = calculate_continuous_prediction(leaf_assignment_current_sample_current_tree, tree, cont_y_test, tree_to_leaf_to_samples_dict, train_sample_weights)

                if not tree in predictions_trees:

                    predictions_trees[tree] = cont_pred

                else:
                    print("Found tree " + str(tree) + " twice")


                if not tree in percentages_all_trees:

                    percentages_all_trees[tree] = percentage_current_tree
                else:
                    print("Found tree " + str(tree) + " twice")


                if not tree in variances_all_trees:

                    variances_all_trees[tree] = variance_current_tree
                else:
                    print("Found tree " + str(tree) + " twice")


            else:

                cont_pred = float("NaN")

                if not tree in predictions_trees:

                    predictions_trees[tree] = cont_pred

                else:
                    print("Found tree " + str(tree) + " twice")

                if not tree in percentages_all_trees:

                    percentages_all_trees[tree] = cont_pred
                else:
                    print("Found tree " + str(tree) + " twice")

                if not tree in variances_all_trees:

                    variances_all_trees[tree] = cont_pred
                else:
                    print("Found tree " + str(tree) + " twice")


        certainty_samples.append(class_proba/number_of_trees_in_forest)
        classification_samples.append(class_predictions_trees)
        predictions_samples.append(predictions_trees)
        leaf_purity_samples.append(percentages_all_trees)
        leaf_variance_samples.append(variances_all_trees)

    return ([predictions_samples, classification_samples, certainty_samples, leaf_purity_samples, leaf_variance_samples])



def calculate_predictions_binary_simple_average_classification_subsample(X_test, number_of_trees_in_forest, number_of_samples, model,
                                                tree_to_leaf_to_percentage_class_dict,
                                                tree_to_leaf_to_sample_variance_dict, cont_y_test, tree_to_leaf_to_samples_dict, train_sample_weights, class_assignment_samples):



    predictions_samples = []
    classification_samples = []
    certainty_samples = []
    leaf_purity_samples = []
    leaf_variance_samples = []

    tree_to_class_counter_dict = determine_tree_to_class_counter_dict(tree_to_leaf_to_samples_dict,
                                                                      class_assignment_samples)

    for sample in range(0, number_of_samples):

        current_sample = X_test[sample, :]  # current sample
        leaf_assignment_current_sample = model.apply(
            current_sample.reshape(1, -1))  # leaf assignment of current sample for all trees

        class_prediction_forest = model.predict(current_sample.reshape(1, -1))[0]

        class_predictions_trees = {}
        predictions_trees = {}
        variances_all_trees = {}
        percentages_all_trees = {}

        class_proba = 0.0
        for tree in range(0, number_of_trees_in_forest):

            class_counter_current_tree = tree_to_class_counter_dict[tree]

            if len(class_counter_current_tree.keys()) == 1:  # only one class, according to documentation the class weight will be 1
                class_weight0 = 1.0
                class_weight1 = 1.0

            else:
                class_weight0 = len(class_assignment_samples) / (
                        len(class_counter_current_tree.keys()) * class_counter_current_tree["0"])
                class_weight1 = len(class_assignment_samples) / (
                        len(class_counter_current_tree.keys()) * class_counter_current_tree["1"])

            leaf_assignment_current_sample_current_tree = leaf_assignment_current_sample[0][tree]

            tree_x = model.estimators_[tree]
            pred_single_tree =str(int( tree_x.predict(current_sample.reshape(1, -1))[0]))

            if tree not in class_predictions_trees:
                class_predictions_trees[tree] = pred_single_tree

            else:
                print("Found tree " + str(tree) + " twice")

            if pred_single_tree == class_prediction_forest:
                class_proba = class_proba + 1.0
                percentage_current_tree = tree_to_leaf_to_percentage_class_dict[tree][leaf_assignment_current_sample_current_tree][class_prediction_forest]
                variance_current_tree = tree_to_leaf_to_sample_variance_dict[tree][leaf_assignment_current_sample_current_tree]

                cont_pred = calculate_continuous_prediction_subsample(leaf_assignment_current_sample_current_tree, tree, cont_y_test, tree_to_leaf_to_samples_dict, train_sample_weights, class_assignment_samples, class_weight0, class_weight1)

                if not tree in predictions_trees:

                    predictions_trees[tree] = cont_pred

                else:
                    print("Found tree " + str(tree) + " twice")


                if not tree in percentages_all_trees:

                    percentages_all_trees[tree] = percentage_current_tree
                else:
                    print("Found tree " + str(tree) + " twice")


                if not tree in variances_all_trees:

                    variances_all_trees[tree] = variance_current_tree
                else:
                    print("Found tree " + str(tree) + " twice")


            else:

                cont_pred = float("NaN")

                if not tree in predictions_trees:

                    predictions_trees[tree] = cont_pred

                else:
                    print("Found tree " + str(tree) + " twice")

                if not tree in percentages_all_trees:

                    percentages_all_trees[tree] = cont_pred
                else:
                    print("Found tree " + str(tree) + " twice")

                if not tree in variances_all_trees:

                    variances_all_trees[tree] = cont_pred
                else:
                    print("Found tree " + str(tree) + " twice")


        certainty_samples.append(class_proba/number_of_trees_in_forest)
        classification_samples.append(class_predictions_trees)
        predictions_samples.append(predictions_trees)
        leaf_purity_samples.append(percentages_all_trees)
        leaf_variance_samples.append(variances_all_trees)

    return ([predictions_samples, classification_samples, certainty_samples, leaf_purity_samples, leaf_variance_samples])


def calculate_final_class_prediction(sample_list_to_tree_prediction_dict):

    final_class_predictions = []
    for sample in range(len(sample_list_to_tree_prediction_dict)):
        class_counts = Counter(sample_list_to_tree_prediction_dict[sample].values())

        current_best_class = None
        best_class_count = 0
        for pred in class_counts.keys():

            if class_counts[pred] > best_class_count:
                best_class_count = class_counts[pred]
                current_best_class = pred

        final_class_predictions.append(current_best_class)

    return final_class_predictions


def print_single_tree_class_prediction_to_file(output_sample_prediction_file, number_of_trees, number_of_samples, classification_prediction, sample_names):

    with open(output_sample_prediction_file, "w", encoding="utf-8") as output_file:

        first_line = ["Prediction_Tree_" + str(tree) for tree in range(0, number_of_trees)]


        all_samples_all_predictions = []
        for current_sample in range(0, number_of_samples):

            one_sample_all_predictions = []
            sample_name = sample_names[current_sample]
            one_sample_all_predictions.append(sample_name)

            for current_tree in range(0, number_of_trees):
                one_sample_single_tree_prediction = classification_prediction[current_sample][current_tree]

                one_sample_all_predictions.append(one_sample_single_tree_prediction)


            all_samples_all_predictions.append(one_sample_all_predictions)

        output_file.write("\t".join(first_line) + "\n")

        for line in all_samples_all_predictions:
            output_file.write("\t".join([str(x) for x in line]) + "\n")


def train_and_test_rf_model(analysis_mode, X_train, X_test, y_train, y_test, sample_names_train, sample_names_test,
                            output_sample_prediction_file_train, output_sample_prediction_file_test, train_error_file,
                            test_error_file, min_number_of_samples_per_leaf, number_of_trees_in_forest,
                            number_of_features_per_split, class_assignment_samples_train, class_assignment_samples_test,
                            name_of_analysis, mse_included, classification_included, feature_imp_output_file,
                            feature_names, threshold, upsampling, time_file, sample_weights_included,
                            top_candidate_percentage, output_leaf_purity_file_train, output_leaf_purity_file_test,
                            output_variance_file_train, output_variance_file_test, leaf_assignment_file_train,
                            leaf_assignment_file_test, sample_info_file, classification_single_tree_output_file_train, classification_single_tree_output_file_test, debug_file):
    # global global_sample_names_train
    # global_sample_names_train= sample_names_train
    # print("Code is new")
    print("Your input parameters are: ")
    print("Analysis mode: " + analysis_mode)
    print("Number of trees: " + str(number_of_trees_in_forest))
    print("Min number of samples per leaf: " + str(min_number_of_samples_per_leaf))
    print("Number of features per split: " + str(number_of_features_per_split))

    print("The dimensions of your training matrix are: ")
    print("Number of rows (samples): " + str(X_train.shape[0]))
    print("Number of columns (features): " + str(X_train.shape[1]))

    #print("Class input vector: ")
    #print(class_assignment_samples_train)
    original_Xtrain = X_train
    original_ytrain = y_train
    original_sample_names_train = sample_names_train
    original_class_assignment_samples_train = class_assignment_samples_train

    model = RandomForestClassifier(n_estimators=number_of_trees_in_forest, random_state=numpy.random.RandomState(2),
                                  min_samples_leaf=min_number_of_samples_per_leaf,
                                  max_features=number_of_features_per_split, bootstrap=True,
                                  oob_score=True, n_jobs=10)

    train_sample_weights = []
    if sample_weights_included in ["linear", "quadratic", "simple"]:

        if not math.isnan(threshold):
            train_sample_weights = numpy.array(harf.calculate_weights(threshold, y_train, sample_weights_included))

            harf.print_train_samples_to_file(original_sample_names_train, train_sample_weights, sample_info_file)
            start_time = time.perf_counter()
            model.fit(X_train, class_assignment_samples_train, sample_weight=train_sample_weights)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
        else:

            print("No Threshold was given to calculate sample weights.")
            print("Please provide a threshold. Calculations aborted.")

    elif sample_weights_included in ["balanced", "balanced_subsample"]:

        train_sample_weights = numpy.ones(len(class_assignment_samples_train))
        model = RandomForestClassifier(n_estimators=number_of_trees_in_forest, random_state=numpy.random.RandomState(2),
                                       min_samples_leaf=min_number_of_samples_per_leaf,
                                       max_features=number_of_features_per_split, bootstrap=True,
                                       oob_score=True, n_jobs=10, class_weight = sample_weights_included)
        start_time = time.perf_counter()
        model.fit(X_train, class_assignment_samples_train)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

    elif sample_weights_included in ["linear_balanced"]:

        model = RandomForestClassifier(n_estimators=number_of_trees_in_forest, random_state=numpy.random.RandomState(2),
                                       min_samples_leaf=min_number_of_samples_per_leaf,
                                       max_features=number_of_features_per_split, bootstrap=True,
                                       oob_score=True, n_jobs=10, class_weight="balanced")

        if not math.isnan(threshold):
            train_sample_weights = numpy.array(harf.calculate_weights(threshold, y_train, "linear"))

            harf.print_train_samples_to_file(original_sample_names_train, train_sample_weights, sample_info_file)
            start_time = time.perf_counter()
            model.fit(X_train, class_assignment_samples_train, sample_weight=train_sample_weights)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
        else:

            print("No Threshold was given to calculate sample weights.")
            print("Please provide a threshold. Calculations aborted.")
    elif sample_weights_included in ["linear_balanced_subsample"]:

        model = RandomForestClassifier(n_estimators=number_of_trees_in_forest, random_state=numpy.random.RandomState(2),
                                       min_samples_leaf=min_number_of_samples_per_leaf,
                                       max_features=number_of_features_per_split, bootstrap=True,
                                       oob_score=True, n_jobs=10, class_weight="balanced_subsample")

        if not math.isnan(threshold):
            train_sample_weights = numpy.array(harf.calculate_weights(threshold, y_train, "linear"))

            harf.print_train_samples_to_file(original_sample_names_train, train_sample_weights, sample_info_file)
            start_time = time.perf_counter()
            model.fit(X_train, class_assignment_samples_train, sample_weight=train_sample_weights)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
        else:

            print("No Threshold was given to calculate sample weights.")
            print("Please provide a threshold. Calculations aborted.")

    elif not upsampling == "":
        new_training_data = harf.upsample_train_data(upsampling, X_train, y_train, class_assignment_samples_train,
                                                sample_names_train, float(threshold))
        up_Xtrain = new_training_data[0]
        up_ytrain = new_training_data[1]
        up_sample_names_train = new_training_data[2]
        new_class_assignment_samples_train = new_training_data[3]

        train_sample_weights = numpy.ones(len(new_class_assignment_samples_train))

        harf.print_train_samples_to_file_upsample(up_sample_names_train, sample_info_file)

        X_train = numpy.array(up_Xtrain)
        y_train = numpy.array(up_ytrain)
        sample_names_train = up_sample_names_train
        class_assignment_samples_train = new_class_assignment_samples_train

        start_time = time.perf_counter()
        model.fit(X_train, class_assignment_samples_train)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

    else:
        train_sample_weights = numpy.ones(len(class_assignment_samples_train))
        start_time = time.perf_counter()
        model.fit(X_train, class_assignment_samples_train)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

    number_of_samples_train = X_train.shape[0]
    number_of_samples_test = X_test.shape[0]

    feature_imp_fit_model = model.feature_importances_

    harf.print_feature_importance_to_file(feature_imp_fit_model, feature_imp_output_file, feature_names)

    # get table with leaf-assignment for each train sample in each tree

    # Have to write this method on my own because apply does not know which samples were not in a bootstrap sample of a tree
    # This is the code that would do it if if the bootstrap samples were handled correctly
    # leaf_assignment_train_data = model.apply(X_train)
    # convert table to dict
    # tree_to_leaf_to_samples_dict = convert_leaf_assignment_to_dict(leaf_assignment_train_data)

    tree_to_leaf_to_samples_dict = harf.my_own_apply(model, X_train, number_of_trees_in_forest)

    # Can be removed when it has been tested for several samples
    #harf.test_equality_by_prediction(tree_to_leaf_to_samples_dict, model, X_train, y_train, train_sample_weights)

    # global global_tree_to_leaf_to_samples_dict
    # global_tree_to_leaf_to_samples_dict = tree_to_leaf_to_samples_dict

    # global if_test_set_used
    # if_test_set_used = False
    harf.write_leaf_assignment_to_file(leaf_assignment_file_train, tree_to_leaf_to_samples_dict, sample_names_train)

    leaf_assignment_all_test_data = model.apply(X_test)
    tree_to_leaf_to_samples_dict_all_test_data = harf.convert_leaf_assignment_to_dict(leaf_assignment_all_test_data)
    harf.write_leaf_assignment_to_file(leaf_assignment_file_test, tree_to_leaf_to_samples_dict_all_test_data,
                                  sample_names_test)

    # Determine purity (and majority class) of the leafs per tree
    # Determine percentage of samples in specific class per tree leaf of one tree

    tree_to_leaf_to_percentage_class_dict = {}

    if sample_weights_included in ["linear", "quadratic", "simple"]:
        tree_to_leaf_to_percentage_class_dict = determine_percentage_of_classes_sample_weights_and_class_weights(
            tree_to_leaf_to_samples_dict,
            class_assignment_samples_train, train_sample_weights)

    elif sample_weights_included in ["linear_balanced", "balanced"]:

        true_train_sample_weights = calculate_adjusted_weights(train_sample_weights, class_assignment_samples_train)
        tree_to_leaf_to_percentage_class_dict = determine_percentage_of_classes_sample_weights_and_class_weights(tree_to_leaf_to_samples_dict, class_assignment_samples_train, true_train_sample_weights)
    elif sample_weights_included in ["linear_balanced_subsample", "balanced_subsample"]:

        tree_to_leaf_to_percentage_class_dict = determine_percentage_of_classes_sample_weights_and_class_weights_subsample(tree_to_leaf_to_samples_dict, class_assignment_samples_train, train_sample_weights)

    else:
        tree_to_leaf_to_percentage_class_dict = determine_percentage_of_classes_sample_weights_and_class_weights(
            tree_to_leaf_to_samples_dict,
            class_assignment_samples_train, train_sample_weights)

    # To understand the distribution of values in nodes, we use some variance information
    # This information can also be used to pick specific trees for predicting

    tree_to_leaf_to_sample_variance_dict = harf.determine_variance_of_samples(tree_to_leaf_to_samples_dict, y_train)

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

    if analysis_mode == "binary_no_weights" and sample_weights_included in ["balanced_subsample", "linear_balanced_subsample"]:

        # Predict with own routine
        predictions_samples_train_and_classification = calculate_predictions_binary_simple_average_classification_subsample(
            original_Xtrain,
            number_of_trees_in_forest,
            original_Xtrain.shape[
                0], model,
            tree_to_leaf_to_percentage_class_dict,
            tree_to_leaf_to_sample_variance_dict, y_train, tree_to_leaf_to_samples_dict, train_sample_weights, class_assignment_samples_train)
        predictions_samples_train = predictions_samples_train_and_classification[0]
        classification_prediction_samples_train = predictions_samples_train_and_classification[1]
        certainty_samples_train = predictions_samples_train_and_classification[2]
        leaf_purity_samples_train = predictions_samples_train_and_classification[3]
        leaf_variance_samples_train = predictions_samples_train_and_classification[4]
        final_predictions_samples_train = harf.calculate_final_predictions_binary_simple_average(
            predictions_samples_train,
            number_of_trees_in_forest)

        final_classification_predictions_samples_train = calculate_final_class_prediction(classification_prediction_samples_train)
        predictions_samples_test_and_classification = calculate_predictions_binary_simple_average_classification_subsample(X_test,
                                                                                                                 number_of_trees_in_forest,
                                                                                                                 number_of_samples_test,
                                                                                                                 model,
                                                                                                                 tree_to_leaf_to_percentage_class_dict,
                                                                                                                 tree_to_leaf_to_sample_variance_dict,
                                                                                                                 y_train,
                                                                                                                 tree_to_leaf_to_samples_dict,
                                                                                                                 train_sample_weights, class_assignment_samples_train)
        predictions_samples_test = predictions_samples_test_and_classification[0]
        classification_prediction_samples_test = predictions_samples_test_and_classification[1]
        certainty_samples_test = predictions_samples_test_and_classification[2]
        leaf_purity_samples_test = predictions_samples_test_and_classification[3]
        leaf_variance_samples_test = predictions_samples_test_and_classification[4]
        final_predictions_samples_test = harf.calculate_final_predictions_binary_simple_average(
            predictions_samples_test,
            number_of_trees_in_forest)
        final_classification_predictions_samples_test = calculate_final_class_prediction(classification_prediction_samples_test)

    elif analysis_mode == "binary_no_weights":
        # Predict with own routine
        predictions_samples_train_and_classification = calculate_predictions_binary_simple_average_classification(original_Xtrain,
                                                                                                   number_of_trees_in_forest,
                                                                                                   original_Xtrain.shape[
                                                                                                       0], model,
                                                                                                   tree_to_leaf_to_percentage_class_dict,
                                                                                                   tree_to_leaf_to_sample_variance_dict,y_train, tree_to_leaf_to_samples_dict, train_sample_weights)
        predictions_samples_train = predictions_samples_train_and_classification[0]
        classification_prediction_samples_train = predictions_samples_train_and_classification[1]
        certainty_samples_train = predictions_samples_train_and_classification[2]
        leaf_purity_samples_train = predictions_samples_train_and_classification[3]
        leaf_variance_samples_train = predictions_samples_train_and_classification[4]
        final_predictions_samples_train = harf.calculate_final_predictions_binary_simple_average(predictions_samples_train,
                                                                                            number_of_trees_in_forest)

        #print(classification_prediction_samples_train)
        final_classification_predictions_samples_train = calculate_final_class_prediction(classification_prediction_samples_train)
        #print("\n Calculated output vector \n")
        #print(final_classification_predictions_samples_train)
        predictions_samples_test_and_classification = calculate_predictions_binary_simple_average_classification(X_test,
                                                                                                  number_of_trees_in_forest,
                                                                                                  number_of_samples_test,
                                                                                                  model,
                                                                                                  tree_to_leaf_to_percentage_class_dict,
                                                                                                  tree_to_leaf_to_sample_variance_dict, y_train, tree_to_leaf_to_samples_dict,train_sample_weights )
        predictions_samples_test = predictions_samples_test_and_classification[0]
        classification_prediction_samples_test = predictions_samples_test_and_classification[1]
        certainty_samples_test = predictions_samples_test_and_classification[2]
        leaf_purity_samples_test = predictions_samples_test_and_classification[3]
        leaf_variance_samples_test = predictions_samples_test_and_classification[4]
        final_predictions_samples_test = harf.calculate_final_predictions_binary_simple_average(predictions_samples_test,
                                                                                           number_of_trees_in_forest)
        final_classification_predictions_samples_test = calculate_final_class_prediction(classification_prediction_samples_test)


    else:
        print(
            "The given analysis mode was not allowed. It was " + analysis_mode + ". Default mode binary_no_weights is being used now.")

        # Predict with own routine
        predictions_samples_train_and_classification = calculate_predictions_binary_simple_average_classification(
            original_Xtrain,
            number_of_trees_in_forest,
            original_Xtrain.shape[
                0], model,
            tree_to_leaf_to_percentage_class_dict,
            tree_to_leaf_to_sample_variance_dict, y_train, tree_to_leaf_to_samples_dict, train_sample_weights)
        predictions_samples_train = predictions_samples_train_and_classification[0]
        classification_prediction_samples_train = predictions_samples_train_and_classification[1]
        certainty_samples_train = predictions_samples_train_and_classification[2]
        leaf_purity_samples_train = predictions_samples_train_and_classification[3]
        leaf_variance_samples_train = predictions_samples_train_and_classification[4]
        final_predictions_samples_train = harf.calculate_final_predictions_binary_simple_average(
            predictions_samples_train,
            number_of_trees_in_forest)

        final_classification_predictions_samples_train = calculate_final_class_prediction(classification_prediction_samples_train)

        predictions_samples_test_and_classification = calculate_predictions_binary_simple_average_classification(X_test,
                                                                                                                 number_of_trees_in_forest,
                                                                                                                 number_of_samples_test,
                                                                                                                 model,
                                                                                                                 tree_to_leaf_to_percentage_class_dict,
                                                                                                                 tree_to_leaf_to_sample_variance_dict,
                                                                                                                 y_train,
                                                                                                                 tree_to_leaf_to_samples_dict,
                                                                                                                 train_sample_weights)
        predictions_samples_test = predictions_samples_test_and_classification[0]
        classification_prediction_samples_test = predictions_samples_test_and_classification[1]
        certainty_samples_test = predictions_samples_test_and_classification[2]
        leaf_purity_samples_test = predictions_samples_test_and_classification[3]
        leaf_variance_samples_test = predictions_samples_test_and_classification[4]
        final_predictions_samples_test = harf.calculate_final_predictions_binary_simple_average(
            predictions_samples_test,
            number_of_trees_in_forest)
        final_classification_predictions_samples_test = calculate_final_class_prediction(classification_prediction_samples_test)



    # Print solutions to file
    parameters = ["#Trees:" + str(number_of_trees_in_forest), "#MinSamplesLeaf:" + str(min_number_of_samples_per_leaf),
                  "#FeaturesPerSplit:" + str(number_of_features_per_split)]

    harf.write_time_to_file(time_file, elapsed_time, name_of_analysis, ",".join(parameters))
    harf.print_prediction_to_file(predictions_samples_train, final_predictions_samples_train, number_of_trees_in_forest,
                             original_Xtrain.shape[0], output_sample_prediction_file_train, original_sample_names_train,
                             certainty_samples_train, final_classification_predictions_samples_train)
    harf.print_error_to_file(train_error_file, name_of_analysis, ",".join(parameters), mse_included, classification_included,
                        final_predictions_samples_train, original_ytrain, final_classification_predictions_samples_train,
                        original_class_assignment_samples_train)
    harf.print_leaf_purity_per_sample_to_file(leaf_purity_samples_train, number_of_trees_in_forest, original_Xtrain.shape[0],
                                         output_leaf_purity_file_train, original_sample_names_train)
    harf.print_leaf_purity_per_sample_to_file(leaf_variance_samples_train, number_of_trees_in_forest,
                                         original_Xtrain.shape[0], output_variance_file_train,
                                         original_sample_names_train, "variance")
    print_single_tree_class_prediction_to_file(classification_single_tree_output_file_train, number_of_trees_in_forest, original_Xtrain.shape[0],
                                               classification_prediction_samples_train, original_sample_names_train)

    harf.print_prediction_to_file(predictions_samples_test, final_predictions_samples_test, number_of_trees_in_forest,
                             number_of_samples_test, output_sample_prediction_file_test, sample_names_test,
                             certainty_samples_test, final_classification_predictions_samples_test)
    harf.print_error_to_file(test_error_file, name_of_analysis, ",".join(parameters), mse_included,
                        classification_included, final_predictions_samples_test, y_test,
                        final_classification_predictions_samples_test, class_assignment_samples_test)
    harf.print_leaf_purity_per_sample_to_file(leaf_purity_samples_test, number_of_trees_in_forest, number_of_samples_test,
                                         output_leaf_purity_file_test, sample_names_test)
    harf.print_leaf_purity_per_sample_to_file(leaf_variance_samples_test, number_of_trees_in_forest, number_of_samples_test,
                                         output_variance_file_test, sample_names_test, "variance")

    print_single_tree_class_prediction_to_file(classification_single_tree_output_file_test, number_of_trees_in_forest,
                                               number_of_samples_test,
                                               classification_prediction_samples_test,
                                               sample_names_test)

    return


