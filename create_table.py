def create_table(y_test, y_predict, accuracy):
    index = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for item in y_predict:
        if item == 3:
            if item == y_test[index]:
                tp += 1
            else:
                fp += 1
        else:
            if item == y_test[index]:
                tn += 1
            else:
                fn += 1
        index += 1
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return [tp, fp, tn, fn, accuracy, tpr, tnr]
