from src.classifier.transformer_model import TransformerClassifier


def prediction_all(config, save_dir):
    labels = get_all_labels()
    test_text = read_test_text()
    for label in labels:
        train_data = read_train_file_by_label(label)
        cls = TransformerClassifier(**config).train(train_data["data"], train_data["label"])
        label_prediction_result = cls.predict(test_text["data"])
        save_label_precition_result(label_prediction_result, label, save_dir)

    final = merge_all_prediction_in_dir(save_dir)
    save_result_in_dir(final, save_dir)
    # read train file by labels
    #
