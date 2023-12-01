import pandas
from scipy import stats

from llm_projects.healthcare.statistical_analyses import (
    plot_distributions,
    plot_distributions_xy,
)


def visit_frequency_by_race(merged_df):
    race_distribution = (
        merged_df.groupby(["race"])[["subject_id", "hadm_id"]]
        .nunique()
        .reset_index()
    )

    race_distribution["hadm_subject_ratio"] = (
        race_distribution["hadm_id"] / race_distribution["subject_id"]
    )

    # Sort the DataFrame by the ratio in descending order
    sorted_race_distribution = race_distribution.sort_values(
        by="hadm_subject_ratio", ascending=False
    )
    plot_distributions_xy(
        x=sorted_race_distribution["race"],
        y=sorted_race_distribution["hadm_subject_ratio"],
        x_label="hadm_subject_ratio",
        top=30,
    )

    return None


def get_length_of_stay(merged_df):
    merged_df["length_of_stay_days"] = round(
        (
            pandas.to_datetime(merged_df["dischtime"])
            - pandas.to_datetime(merged_df["admittime"])
        ).dt.total_seconds()
        / 3600
        / 24,
        2,
    )

    merged_df[merged_df["length_of_stay_days"].isna()] = 0
    return merged_df


def plot_length_of_stay(merged_df):
    unique_times = (
        merged_df.groupby(["subject_id", "hadm_id"])["length_of_stay_days"]
        .first()
        .reset_index()
    )

    plot_distributions(
        unique_times["length_of_stay_days"], "length_of_stay_days"
    )

    return None


def get_single_distribution(merged_df, column_name, categ):
    a_distribution = (
        merged_df[column_name].value_counts().reset_index(name="counts")
    )
    x = a_distribution[column_name]
    y = a_distribution["counts"]
    plot_distributions_xy(x, y, column_name, categ)

    return None


def get_outlier_stays(merged_df):
    merged_df["length_of_stay_zscore"] = stats.zscore(
        merged_df["length_of_stay_days"]
    )

    outliers = merged_df[abs(merged_df["length_of_stay_zscore"]) > 3]
    long_stays_df = outliers.drop_duplicates(subset=["subject_id", "hadm_id"])

    return merged_df, long_stays_df


def long_stay_analysis(merged_df, long_stays_df):
    plot_distributions(long_stays_df["careunit"], "careunit")

    # outliers = merged_df[abs(merged_df['length_of_stay_days']) > 5]
    #
    # long_stays_df = outliers.drop_duplicates(subset=['subject_id', 'hadm_id'])
    # unique_long_stays_df = long_stays_df.drop_duplicates(subset='subject_id',
    #                                                      keep='first')
    # # Calculate and visualize overall mortality rate among unique subject_ids
    # total_unique_patients = len(unique_long_stays_df)
    # unique_deaths = unique_long_stays_df['hospital_expire_flag'].sum()
    # mortality_rate_unique = unique_deaths / long_stays_df.shape[0] * 100
    # print(
    #     f"Overall Mortality Rate among Unique subject_ids: {mortality_rate_unique:.2f}%"
    # )

    long_stay_per_user = long_stays_df.subject_id.value_counts().reset_index(
        name="counts"
    )


def admission_to_transfer(admissions_df, transfers_df):
    transfers_df["hadm_id"] = transfers_df["hadm_id"].astype("Int64")

    merged_df = pandas.merge(
        transfers_df,
        admissions_df,
        on=["subject_id", "hadm_id"],
        how="left",
        suffixes=("_transfer", "_admission"),
    )

    merged_df = get_length_of_stay(merged_df)
    merged_df, long_stays_df = get_outlier_stays(merged_df)
    long_stay_analysis(long_stays_df)
    plot_distributions(long_stays_df["careunit"], "careunit")
    long_stay_per_user = long_stays_df.subject_id.value_counts().reset_index(
        name="counts"
    )
    # get_single_distribution(merged_df, column_name='admission_type', categ="admission")
    # get_single_distribution(
    #     merged_df, column_name='discharge_location', categ="admission"
    # )
    # get_single_distribution(merged_df, column_name='careunit', categ="transfer")
    # visit_frequency_by_race(merged_df)
    # plot_length_of_stay(merged_df)

    return None
