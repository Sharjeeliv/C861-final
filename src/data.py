import pandas as pd
import numpy as np
from pathlib import Path


ROOT = Path.cwd()



def clean_tweets():
    # Phase 1
    # Remove @'s and user references
    #   Start at @ until whitespace   
    # Keep hashtags (jsut remove hashtag?)

    # Phase 2
    # Remove extra punctuation and symbols
    # Special character and digit removal
    # tokenization
    # lowercase all
    # remove stopwrods
    # lemmatization (?)
    # join back, remove extra spacing
    pass



def preprocess(file_path: Path):
    # Extract data
    df = pd.read_csv(file_path)
    df.columns = ['id', 'entity', 'score', 'text']
    print(df.head())
    
    # I may need to make two splits, augmented and non
    print(df['id'].count())
    print(df['id'].nunique())
    
    
    


if __name__ == "__main__":
    # We can pass each csv
    file = ROOT / 'data' / 'training.csv'
    
    preprocess(file)
    print(file)