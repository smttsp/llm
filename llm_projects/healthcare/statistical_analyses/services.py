import matplotlib.pyplot as plt
import pandas
import seaborn as sns


def plot_num_services_per_patient(services_df: pandas.DataFrame):
    num_services_each_patient = (
        services_df.groupby("subject_id").size().reset_index(name="counts")
    )
    num_services_each_patient = num_services_each_patient.sort_values(
        by="counts", ascending=False
    )

    max_service = num_services_each_patient["counts"].max() + 1
    sns.histplot(
        num_services_each_patient["counts"],
        bins=range(1, max_service),
        kde=False,
        color="skyblue",
    )

    plt.title("Distribution of Number of Services per Patient")
    plt.xlabel("Number of Services")
    plt.ylabel("Number of Patients")
    plt.show()


def plot_distribution_of_service_type(
    services_df: pandas.DataFrame, service_column
):
    services_df[service_column].value_counts().plot(kind="bar", color="skyblue")
    plt.title(f"Distribution of {service_column}")
    plt.xlabel(service_column)
    plt.ylabel("Count")
    plt.show()


def get_services_statistics(services_df: pandas.DataFrame):
    plot_num_services_per_patient(services_df)
    plot_distribution_of_service_type(services_df, "prev_service")
    plot_distribution_of_service_type(services_df, "curr_service")
