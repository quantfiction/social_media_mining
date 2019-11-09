import pandas as pd, numpy as np
import string


def format_cols(df):
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def col_to_datetime(df, col, unit="s"):
    df[col] = pd.to_datetime(df[col], unit=unit)
    return df


def lowercase_cols(df, cols):
    for col in cols:
        df[col] = df[col].str.lower()
    return df


def remove_punctuation(df, col):
    punct = string.punctuation.replace("|", "") + "\\"
    transtab = str.maketrans(dict.fromkeys(punct, ""))

    translated = df[col].str.translate(transtab).replace(r"\n\n", " ", regex=True)
    df[col] = translated
    return df


def ffill_cols(df, cols):
    df.loc[:, cols] = df.loc[:, cols].ffill()
    return df


def create_regex_filter(terms):
    filt = ""
    for term in terms:
        filt += term.lower() + "|"
    return filt


def contains_str(df, col, _str):
    df = df.loc[(df[col].fillna("").str.contains(_str, regex=True))]
    return df


def isin_col(df, col, strings):
    str_list = [term.lower() for term in strings]
    condition = df[col].isin(str_list)
    return df.loc[condition]


def format_datetime_index(df):
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()
