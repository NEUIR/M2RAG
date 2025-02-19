import json
from sklearn.metrics import classification_report, accuracy_score
import argparse
def read_predictions_and_labels(file_path):

    ids = []
    categories = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            ids.append(data['id'])
            categories.append(data['category'] if 'category' in data else data['text'])
    return ids, categories

def evaluate_classification_metrics(true_file, pred_file):
    """
    Compute evaluation metrics for the classification task, including Recall, F1, Accuracy, etc.
    """

    true_ids, true_labels = read_predictions_and_labels(true_file)
    pred_ids, pred_labels = read_predictions_and_labels(pred_file)

    if true_ids != pred_ids:
        raise ValueError("IDs in the true and predicted files do not match!")


    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, digits=4, output_dict=True)


    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(json.dumps(report, indent=4))


    return accuracy, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--true_file",type=str)
    parser.add_argument("--pred_file", type=str)
    args = parser.parse_args()

    evaluate_classification_metrics(args.true_file, args.pred_file)