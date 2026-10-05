import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("open_tggates_master_list.csv")
df = df[df['ORGAN'] == 'Kidney'] # Only kidney slides

healthy = df[df['findings'].str.len() == 2]
print("Healthy slides:", len(healthy))
print("Proportion non-unique:", (len(healthy) - len(healthy['subject_UID'].unique())) / len(healthy))

unhealthy = df[df['findings'].str.len() != 2]
print("Unhealthy slides:", len(unhealthy))
print("Proportion non-unique:", (len(unhealthy) - len(unhealthy['subject_UID'].unique())) / len(unhealthy))
print("------")



counts = df['subject_UID'].value_counts()   # slides per animal
dup = counts[counts > 1]                    # only animals with >1 slide
print("Slide count:", len(df))
print("Unique:", (counts == 1).sum())
print("Duplicate:", len(dup))
print("---stats---")
print("Median:", dup.median())
print("Mean:", dup.mean())
print("STD:", dup.std())

dist = dup.value_counts().sort_index()      # index = slides per animal, value = number of animals

plt.bar(dist.index, dist.values)
plt.yscale('log')
plt.xticks(dist.index)
plt.xlabel("Slides per animal")
plt.ylabel("Number of animals (log scale)")
plt.title("Animals with more than one kidney slide")
plt.savefig("results/unique_histogram.png", dpi=150)
