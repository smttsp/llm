import matplotlib.pyplot as plt
import pandas
import seaborn as sns


def get_admissions_statistics(admissions_df: pandas.DataFrame):
    admissions_df.loc[
        admissions_df["field_name"] == "Consult Status Time", "field_value"
    ] = "0"
    unique_pairs_df = (
        admissions_df.groupby(["field_name", "field_value"])
        .size()
        .reset_index(name="counts")
    )

    # Sort the DataFrame by counts in descending order
    unique_pairs_df = unique_pairs_df.sort_values(by="counts", ascending=False)

    # Display the sorted unique pairs DataFrame
    # print(unique_pairs_df.head(10))

    plt.figure(figsize=(16, 10))

    # Create a bar plot using Seaborn
    sns.barplot(
        x=unique_pairs_df["field_name"]
        + " - "
        + unique_pairs_df["field_value"],
        y=unique_pairs_df["counts"],
        palette="viridis",
    )

    plt.title("Distribution of Field Name + Field Value")
    plt.xlabel("Field Name + Field Value")
    plt.ylabel("Count")
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=60, ha="right")  # Adjust the rotation and ha parameters

    # Automatically adjust subplot parameters for better layout
    plt.tight_layout()

    plt.show()
