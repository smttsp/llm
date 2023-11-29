import pandas
from glob import glob

from llm_projects.healthcare.patients import get_patient_statistics
from llm_projects.healthcare.admissions import get_admissions_statistics

def read_csvs(folder):
    files = glob(folder + "*.csv")
    df_dict = {}
    for file in files:
        filename = file.split("/")[-1].split(".")[0]
        print(filename)
        df = pandas.read_csv(file, nrows=1000)
        print(df.columns)
        print()
        df_dict[filename] = df

    admissions_df = pandas.read_csv(files[0])
    get_admissions_statistics(admissions_df)
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
