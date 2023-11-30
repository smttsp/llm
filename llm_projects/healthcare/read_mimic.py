import pandas
from glob import glob

from llm_projects.healthcare import (
    get_patient_statistics,
    get_admissions_statistics,
    get_services_statistics,
    get_labitem_statistics,
    get_diagnosis_statistics,
    get_omr_statistics,
)
from pprint import pprint

from .complex_analysis import admission_to_transfer


def read_csvs(folder):
    files = glob(folder + "*.csv")
    # completed = [
    #     "patients", "admissions", "provider", "services", "d_labitems", "omr"
    # ]
    # files = [f for f in files if f.split("/")[-1].split(".")[0] not in completed]

    pair = ["admissions", "transfers"]

    df_dict = {}
    for idx, file in enumerate(files):
        filename = file.split("/")[-1].split(".")[0]
        if filename not in pair:
            continue

        df = pandas.read_csv(file)
        print(idx, filename, len(df.columns), df.shape)
        # print(df.columns)
        # pprint(df.head(5).to_dict())

        # print("\n\n")
        # print(df.columns)
        # print()
        df_dict[filename] = df

    df0 = df_dict[pair[0]]
    df1 = df_dict[pair[1]]

    admission_to_transfer(df0, df1)

    pass


def read_csvs_old(folder):
    files = glob(folder + "*.csv")

    completed = [
        "patients", "admissions", "provider", "services", "d_labitems", "omr"
    ]

    skipped_for_now = ["d_hcpcs"]

    df_dict = {}
    for idx, file in enumerate(files):
        filename = file.split("/")[-1].split(".")[0]
        if filename not in completed + skipped_for_now:
            continue
        # if filename in skipped_for_now:
        #     continue
        #
        # if filename != "emar_detail":
        #     continue
        # df = pandas.read_csv(file, nrows=1000)
        # get_omr_statistics(df)
        df = pandas.read_csv(file, nrows=100)
        print(idx, filename, len(df.columns), df.shape)
        print(df.columns)
        pprint(df.head(5).to_dict())

        print("\n\n")
        # print(df.columns)
        # print()
        df_dict[filename] = df

    pass


# small files, single column
# provider_df = pandas.read_csv(files[1])  # no information, only provider_id

# large files
# pharmacy_df = pandas.read_csv(files[2], nrows=10000)


# completed functions

# admissions_df = pandas.read_csv(files[0])
# patient_df = pandas.read_csv(files[18])

# get_patient_statistics(df_dict, patient_df)
# get_admissions_statistics(admissions_df)


# services_df = pandas.read_csv(files[15])  # no information, only provider_id
# get_services_statistics(services_df)
