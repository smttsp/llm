import matplotlib.pyplot as plt
import seaborn as sns


def plot_categorical(df, key):
    categ = df[key]

    plt.figure(figsize=(8, 6))
    sns.countplot(x=categ, palette='pastel')
    plt.title(f'Distribution of {key}')
    plt.xlabel(key)
    plt.ylabel('Count')

    plt.xticks(rotation=45, ha='right')

    plt.show()


def plot_distributions(counts, x_label, per="visit"):
    plt.figure(figsize=(8, 6))
    sns.histplot(counts, bins=100, kde=False, color='salmon')
    plt.title(f'Distribution of {x_label} per {per}')
    plt.xlabel(x_label)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')

    plt.show()


def plot_distributions_xy(x, y, x_label, per="visit"):
    plt.figure(figsize=(8, 6))

    plt.bar(x, y, color='salmon')
    # sns.histplot(counts, bins=100, kde=False, color='salmon')
    plt.title(f'Distribution of {x_label} per {per}')
    plt.xlabel(x_label)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
