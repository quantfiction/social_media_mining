import pandas as pd, numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


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
    p = r"\b(?:{})\b".format("|".join(map(re.escape, strings)))
    condition = df[col].contains(p)
    return df.loc[condition]


def format_datetime_index(df):
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def rename_index(df, name):
    df.index.name = name
    return df


################################
# NLTK / Text Mining Functions #
################################


def remove_punctuation(df, col):
    punct = string.punctuation.replace("|", "") + r"\n\n" + "”" + "’"
    transtab = str.maketrans(dict.fromkeys(punct, ""))

    translated = df[col].str.translate(transtab).replace("”|’", "", regex=True)
    df[col] = translated
    return df


def _tokenize(x, remove_duplicates=True):
    tokenizer = RegexpTokenizer(r"\w+")
    raw_tokens = tokenizer.tokenize(x)
    tokens = set(raw_tokens) if remove_duplicates else raw_tokens
    return list(tokens)


def remove_stopwords(df, col):
    to_remove = stopwords.words("english")
    pat = r"\b(?:{})\b".format("|".join(to_remove))
    df[col] = df[col].str.replace(pat, "", regex=True)
    return df
