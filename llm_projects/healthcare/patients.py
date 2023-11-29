import pandas
import seaborn as sns
import matplotlib.pyplot as plt


def fnc_gender(patient_df: pandas.DataFrame, key="gender"):
    gender = patient_df[key]

    plt.figure(figsize=(8, 6))
    sns.countplot(x=gender, palette='pastel')
    plt.title(f'Distribution of {key}')
    plt.xlabel(key)
    plt.ylabel('Count')
    plt.show()


def fnc_age(patient_df, key="anchor_age"):
    gender_counts = patient_df.groupby('gender')[key].value_counts().unstack().T
    labels = gender_counts.keys().to_list()
    gender_counts.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'])

    plt.title(f'Distribution of {key} by Gender')
    plt.xlabel(key)
    plt.ylabel('Count')
    plt.legend(
        title='Gender', loc='upper right', labels=labels
    )
    plt.xticks(
        range(0, len(gender_counts), 10),
        gender_counts.index[::10],
        rotation=45,
        ha='right'
    )

    plt.show()



def fnc_dod(patient_df):
    death_age = patient_df.anchor_age[patient_df.dod.notnull()]
    plt.figure(figsize=(8, 6))
    sns.histplot(death_age, bins=100, kde=False, color='salmon')
    plt.title(f'Distribution of age that patient died')
    plt.xlabel("death_age")
    plt.ylabel('Count')
    plt.show()


def get_patient_statistics(df_dict, patient_df):
    fnc_gender(patient_df)
    fnc_age(patient_df)
    fnc_dod(patient_df)

    pass