import matplotlib.pyplot as plt
import seaborn as sns


def plot_categorical(df, key):
    categ = df[key]

    plt.figure(figsize=(8, 6))
    sns.countplot(x=categ, palette='pastel')
    plt.title(f'Distribution of {key}')
    plt.xlabel(key)
    plt.ylabel('Count')

    plt.xticks(
        rotation=45,
        ha='right'
    )

    plt.show()
