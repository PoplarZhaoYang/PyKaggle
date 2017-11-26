import pandas as pd

def convert_object_into_label(df):
    """label the object feature
    """
    df_num = df.select_dtypes(exclude=['object'])
    df_obj = df.select_dtypes(include=['object']).copy()
    for c in df_obj:
        df_obj[c] = pd.factorize(df_obj[c])[0]
    return pd.concat([df_num, df_obj], axis=1)


