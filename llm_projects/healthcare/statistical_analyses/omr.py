from .generic_plots import plot_distributions


def get_omr_statistics(omr_df):
    bmi_list = omr_df[
        omr_df["result_name"].isin(["BMI", "BMI (kg/m2)"])
    ].result_value.to_list()

    # bmi_list = list(map(float, bmi_list))
    bmi_list = [float(val) for val in bmi_list if float(val) < 80]

    plot_distributions(bmi_list, "BMI", "entry")
