import csv

with open("./data/data.csv", "w") as data_file:
    with open("../sheet_5/data/train.csv", "r") as train_data_file:
        train_data_reader = csv.reader(train_data_file)
        train_data = list(train_data_reader)[1:]
        for row in train_data:
            data_file.write(row[0] + ", " + row[1] + "\n")
    with open("../sheet_5/data/test.csv", "r") as test_data_file:
        test_data_reader = csv.reader(test_data_file)
        test_data = list(test_data_reader)[1:]
        for row in test_data:
            data_file.write(row[0] + ", " + row[1] + "\n")
