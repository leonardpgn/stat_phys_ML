import pandas as pd

data_frame = pd.read_csv("./data/train.csv", index_col="id")


with open("./data/data_head.txt", "w") as data_head_file:
    for row in range(10):
        for column in data_frame.columns:
            print(f"{column}: ", end="", file=data_head_file)
            print(data_frame[column].iloc[row], file=data_head_file)
        print("\n\n", file=data_head_file)

    print(data_frame.shape, file=data_head_file)
