import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from .generic_plots import plot_categorical


def get_labitem_statistics(labitem_df: pandas.DataFrame):
    plot_categorical(labitem_df, "fluid")
    plot_categorical(labitem_df, "category")
    pass
