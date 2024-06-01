import pandas as pd
import matplotlib.pyplot as plt

data_frame = pd.read_csv("./data/train.csv", index_col="id")

with open("./data/data_head.txt", "w") as data_head_file:
    for row in range(10):
        for column in data_frame.columns:
            print(f"{column}: ", end="", file=data_head_file)
            print(data_frame[column].iloc[row], file=data_head_file)
        print("\n\n", file=data_head_file)

    print(data_frame.shape, file=data_head_file)


# d
starting_time, ending_time = data_frame["pickup_datetime"], data_frame["dropoff_datetime"]
formatted_starting_time, formatted_ending_time = [], []
for starting_time in starting_time:
    starting_time = starting_time[-8:]
    starting_time = int(starting_time[:2]) + 1/60 * int(starting_time[3:5])
    formatted_starting_time.append(starting_time)

fig, ax = plt.subplots(dpi=500)
ax.hist(formatted_starting_time, bins=24*12)
ax.set(xlabel="Time", ylabel="# pickups / 5 min", xlim=(0, 24), xticks=range(25))
ax.grid(alpha=.2)

fig.tight_layout()
fig.savefig("./figures/wake_up_time.png")
