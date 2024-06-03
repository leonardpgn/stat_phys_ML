import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data_frame = pd.read_csv("./data/train.csv", index_col="id")

with open("./data/data_head.txt", "w") as data_head_file:
    for row in range(10):
        for column in data_frame.columns:
            print(f"{column}: ", end="", file=data_head_file)
            print(data_frame[column].iloc[row], file=data_head_file)
        print("\n\n", file=data_head_file)

    print(data_frame.shape, file=data_head_file)


# d
starting_time, duration = data_frame["pickup_datetime"], data_frame["trip_duration"]
formatted_starting_time, formatted_ending_time = [], []
for starting_time, duration in zip(starting_time, duration):
    starting_time = starting_time[-8:]
    starting_time = int(starting_time[:2]) + 1/60 * int(starting_time[3:5])
    formatted_starting_time.append(starting_time)
    formatted_ending_time.append(min(starting_time + (duration / 3600), 23.99))

formatted_starting_time = np.array(formatted_starting_time)
formatted_ending_time = np.array(formatted_ending_time)

fig, ax = plt.subplots(dpi=500)
ax.hist(formatted_starting_time, bins=24*12)
ax.set(xlabel="Time", ylabel="# pickups / 5 min", xlim=(0, 24), xticks=range(25))
ax.grid(alpha=.2)

fig.tight_layout()
fig.savefig("./figures/wake_up_time.png")

# e
traffic_counts = []
time_instances = np.arange(0, 24, 1/12)
for time_instance in time_instances:
    traffic_count = (
            len([starting_time for starting_time in formatted_starting_time if starting_time <= time_instance])
            - len([ending_time for ending_time in formatted_ending_time if ending_time <= time_instance])
    )
    traffic_counts.append(traffic_count)

fig, ax = plt.subplots(dpi=500)
ax.bar(time_instances, traffic_counts)
ax.set(xlabel="Time", ylabel="traffic count", xlim=(0, 24), xticks=range(25))
ax.grid(alpha=.2)

fig.tight_layout()
fig.savefig("./figures/traffic_counts.png")


# f
passenger_count = np.array(data_frame["passenger_count"])

fig, ax = plt.subplots(dpi=500)
ax.hist(passenger_count, width=0.8, label="passenger count", bins=max(passenger_count))
ax.set(xlabel="# passengers", ylabel="rides", yscale="log")

fig.tight_layout()
fig.savefig("./figures/passenger_data.png")


# g
pca_data = np.array(list(zip(
    list(formatted_starting_time),
    [int(q) for q in data_frame["vendor_id"]],
    [int(q) for q in data_frame["passenger_count"]],
    [float(q) for q in data_frame["pickup_longitude"]],
    [float(q) for q in data_frame["pickup_latitude"]],
    [float(q) for q in data_frame["dropoff_longitude"]],
    [float(q) for q in data_frame["dropoff_latitude"]],
    [int(q) for q in data_frame["trip_duration"]]
)))
scale = StandardScaler()
scaled_data = scale.fit_transform(pca_data)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(reduced_data)

centers = kmeans.cluster_centers_

fig, ax = plt.subplots(dpi=500)
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, s=10)
ax.scatter(centers[:, 0], centers[:, 1], color="dodgerblue", s=10)

ax.set(xlim=(-20, 20), ylim=(-20, 20))

fig.tight_layout()
fig.savefig("./figures/clustering")
