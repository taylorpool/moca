import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot(names, vals, filename, title):
    sns.set_theme("talk", "whitegrid")

    fig, ax = plt.subplots(layout="constrained")

    df = pd.DataFrame({"name": names, "Mean Time (s)": vals})
    sns.barplot(
        data=df,
        x="name",
        y="Mean Time (s)",
        hue="name",
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_title(title)
    plt.savefig(filename)
