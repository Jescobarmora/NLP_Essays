import pandas as pd

def load_data(filepath):
    data = pd.read_csv(filepath, sep='\t', encoding='ISO-8859-1')
    data = data[['essay_id', 'essay', 'essay_set']]
    data = data.rename(columns={'essay_set': 'topic'})
    return data