import pandas

from .generic_plots import plot_distributions


def get_diagnosis_statistics(diagnoses_icd_df: pandas.DataFrame):
    grouped_data = (
        diagnoses_icd_df.groupby(["subject_id", "hadm_id"])
        .size()
        .reset_index(name="counts")
    )

    # print(grouped_data)

    plot_distributions(grouped_data["counts"], "num_icd_codes", "admission")
