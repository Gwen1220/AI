from collections import defaultdict
import numpy as np


def file_reader(file_path, label):
    list_of_lines = []
    list_of_labels = []

    for line in open(file_path):
        line = line.strip()
        if line == "":
            continue
        list_of_lines.append(line)
        list_of_labels.append(label)

    return (list_of_lines, list_of_labels)


def data_reader(source_directory):
    positive_file = source_directory + "Positive.txt"
    (positive_list_of_lines, positive_list_of_labels) = file_reader(file_path=positive_file, label=1)

    negative_file = source_directory + "Negative.txt"
    (negative_list_of_lines, negative_list_of_labels) = file_reader(file_path=negative_file, label=-1)

    neutral_file = source_directory + "Neutral.txt"
    (neutral_list_of_lines, neutral_list_of_labels) = file_reader(file_path=neutral_file, label=0)

    list_of_all_lines = positive_list_of_lines + negative_list_of_lines + neutral_list_of_lines
    list_of_all_labels = np.array(positive_list_of_labels + negative_list_of_labels + neutral_list_of_labels)

    return list_of_all_lines, list_of_all_labels


def evaluate_predictions(test_set, test_labels, trained_classifier):
    correct_predictions = 0
    predictions_list = []
    prediction = -1
    for dataset, label in zip(test_set, test_labels):
        probabilities = trained_classifier.predict(dataset)
        if probabilities[0] >= probabilities[1] and probabilities[0] >= probabilities[-1]:
            prediction = 0
        elif probabilities[1] >= probabilities[0] and probabilities[1] >= probabilities[-1]:
            prediction = 1
        else:
            prediction = -1
        if prediction == label:
            correct_predictions += 1
            predictions_list.append("+")
        else:
            predictions_list.append("-")

    print("Total Sentences correctly: ", len(test_labels))
    print("Predicted correctly: ", correct_predictions)
    print("Accuracy: {}%".format(round(correct_predictions / len(test_labels) * 100, 5)))

    return predictions_list, round(correct_predictions / len(test_labels) * 100)


class NaiveBayesClassifier(object):
    def __init__(self, n_gram=1, printing=False):
        self.prior = {}
        self.conditional = {}
        self.V = {}
        self.n = n_gram

    def word_tokenization_dataset(self, training_sentences):
        training_set = list()
        for sentence in training_sentences:
            cur_sentence = list()
            for word in sentence.split(" "):
                cur_sentence.append(word.lower())
            training_set.append(cur_sentence)
        return training_set

    def word_tokenization_sentence(self, test_sentence):
        cur_sentence = list()
        for word in test_sentence.split(" "):
            cur_sentence.append(word.lower())
        return cur_sentence

    def compute_vocabulary(self, training_set):
        vocabulary = set()
        for sentence in training_set:
            for word in sentence:
                vocabulary.add(word)
        V_dictionary = dict()
        dict_count = 0
        for word in vocabulary:
            V_dictionary[word] = int(dict_count)
            dict_count += 1
        return V_dictionary

    def train(self, training_sentences, training_labels):

        N_sentences = len(training_sentences)
        training_set = self.word_tokenization_dataset(training_sentences)
        self.V = self.compute_vocabulary(training_set)
        all_classes = set(training_labels)

        vocab_size = len(self.V)

        # build  binary BOW matrix
        bow_matrix = np.zeros((N_sentences, vocab_size), dtype=int)
        for i, tokens in enumerate(training_set):
            for word in set(tokens):
                if word in self.V:
                    bow_matrix[i, self.V[word]] = 1
        # Get sorted list of all classes
        self.classes = sorted(list(all_classes))
        # Initialize prior and conditional probability dictionaries
        self.prior = {}
        self.conditional = {}

        for c in self.classes:
            class_indices = np.where(training_labels == c)[0]
            N_c = len(class_indices)
            self.prior[c] = N_c / float(N_sentences)

            class_bow = bow_matrix[class_indices, :]
            word_counts_c = np.sum(class_bow, axis=0)

            # improved smoothing for Bernoulli Naive Bayes
            # Bernoulli: (count + 1) / (N_c + 2)
            p_c = (word_counts_c + 1) / (N_c + 2)

            self.conditional[c] = p_c

    def predict(self, test_sentence):
        label_probability = {0: 0, 1: 0, -1: 0}
        test_sentence = self.word_tokenization_sentence(test_sentence)

        vocab_size = len(self.V)
        x = np.zeros(vocab_size, dtype=int)
        for word in set(test_sentence):
            if word in self.V:
                idx = self.V[word]
                x[idx] = 1

        for c in self.classes:
            p_c = self.conditional[c]
            log_prob = np.log(self.prior[c])
            appear_idx = x == 1
            not_appear_idx = x == 0

            # improved smoothing: add a small constant to avoid log(0)
            if np.any(appear_idx):
                log_prob += np.sum(np.log(p_c[appear_idx] + 1e-12))
            if np.any(not_appear_idx):
                log_prob += np.sum(np.log(1.0 - p_c[not_appear_idx] + 1e-12))

            label_probability[c] = log_prob

        return label_probability


if __name__ == '__main__':
    train_folder = "data-sentiment/train/"
    test_folder = "data-sentiment/test/"

    training_sentences, training_labels = data_reader(train_folder)
    test_sentences, test_labels = data_reader(test_folder)

    NBclassifier = NaiveBayesClassifier(n_gram=1)
    NBclassifier.train(training_sentences, training_labels)

    results, acc = evaluate_predictions(test_sentences, test_labels, NBclassifier)
