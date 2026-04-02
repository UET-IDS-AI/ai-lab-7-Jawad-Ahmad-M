import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


# =========================
# Q1 Naive Bayes
# =========================

def naive_bayes_mle_spam():
    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    test_email = "win cash prize now"

    # 1. Tokenization
    tokenized_texts = [text.split() for text in texts]

    # 2. Vocabulary
    vocab = set()
    for tokens in tokenized_texts:
        vocab.update(tokens)

    # 3. Class priors
    priors = {}
    total_docs = len(labels)
    for c in [0, 1]:
        priors[c] = np.sum(labels == c) / total_docs

    # 4. Word probabilities (MLE, no smoothing)
    word_probs = {0: {}, 1: {}}

    for c in [0, 1]:
        # collect words of class c
        words = []
        for tokens, label in zip(tokenized_texts, labels):
            if label == c:
                words.extend(tokens)

        total_words = len(words)

        for word in vocab:
            count = words.count(word)
            if total_words > 0:
                word_probs[c][word] = count / total_words
            else:
                word_probs[c][word] = 0.0

    # 5. Prediction
    test_tokens = test_email.split()

    scores = {}
    for c in [0, 1]:
        score = np.log(priors[c])  # use log to avoid underflow
        for word in test_tokens:
            prob = word_probs[c].get(word, 0)
            if prob > 0:
                score += np.log(prob)
            else:
                score += -1e9  # simulate log(0)
        scores[c] = score

    prediction = max(scores, key=scores.get)

    return priors, word_probs, prediction


# =========================
# Q2 KNN
# =========================

def knn_iris(k=3, test_size=0.2, seed=0):

    # 1. Load dataset
    data = load_iris()
    X = data.data
    y = data.target

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Euclidean distance
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # KNN prediction function
    def predict(X_train, y_train, X_test):
        predictions = []

        for test_point in X_test:
            distances = []

            for x_train, y_label in zip(X_train, y_train):
                dist = euclidean_distance(test_point, x_train)
                distances.append((dist, y_label))

            # sort by distance
            distances.sort(key=lambda x: x[0])

            # take k nearest
            neighbors = distances[:k]

            # majority vote
            labels = [label for _, label in neighbors]
            pred = max(set(labels), key=labels.count)

            predictions.append(pred)

        return np.array(predictions)

    # 3. Predictions
    train_predictions = predict(X_train, y_train, X_train)
    test_predictions = predict(X_train, y_train, X_test)

    # 4. Accuracy
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    return train_accuracy, test_accuracy, test_predictions
