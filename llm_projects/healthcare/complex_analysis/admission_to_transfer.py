from llm_projects.healthcare.statistical_analyses import (
    plot_distributions, plot_distributions_xy
)


def get_admission_transfer_stats(
    admission_type_distribution,
    discharge_location_distribution,
    care_unit_distribution,
):
    x = admission_type_distribution['admission_type']
    y = admission_type_distribution['counts']
    plot_distributions_xy(x, y, "admission_type", "admission")

    x = discharge_location_distribution['discharge_location']
    y = discharge_location_distribution['counts']
    plot_distributions_xy(x, y, "discharge_location", "admission")

    x = care_unit_distribution['careunit']
    y = care_unit_distribution['counts']
    plot_distributions_xy(x, y, "care_unit", "transfer")


def get_distributions(admissions_df, transfers_df):
    admission_type_distribution = admissions_df[
        'admission_type'
    ].value_counts().reset_index(name='counts')

    discharge_location_distribution = admissions_df[
        'discharge_location'
    ].value_counts().reset_index(name='counts')

    care_unit_distribution = transfers_df[
        'careunit'
    ].value_counts().reset_index(name='counts')

    return (
        admission_type_distribution,
        discharge_location_distribution,
        care_unit_distribution,
    )


def admission_to_transfer(admissions_df, transfers_df):
    (
        admission_type_distribution,
        discharge_location_distribution,
        care_unit_distribution,
    ) = get_distributions(admissions_df, transfers_df)

    get_admission_transfer_stats(
        admission_type_distribution,
        discharge_location_distribution,
        care_unit_distribution,
    )
