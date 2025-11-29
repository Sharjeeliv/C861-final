import pandas as pd
import numpy as np
from pathlib import Path
import csv

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
    
    # Manual processing of csv:
    # First three commas correctly separate id, entity, score
    # We need to process text properly
    # text = []
    
    # text = csv.reader(open(file_path, 'r', encoding='utf-8'))
    # for row in text:
    #     print(row)
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         parts = line.strip().split(',', 3)
    #         if len(parts) == 4:
    #             text.append(parts)
    #         else:
    #             print(f"Skipping malformed line: {line}")
    
    rows = []

    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, quotechar='"', escapechar='\\')
        
        for row in reader:
            # Skip completely malformed lines
            if len(row) < 4:
                print(f"Skipping malformed line: {row}")
                continue

            # If there are more than 4 columns, merge extras into text
            id_, game, sentiment = row[:3]
            text = ",".join(row[3:])  # combine the rest
            
            
            rows.append([int(id_.strip()), game.strip(), sentiment.strip(), text.strip()])
    df = pd.DataFrame(rows, columns=["id", "game", "sentiment", "text"])
    # print(df.head())
    
    # df = pd.read_csv(file_path)
    # df.columns = ['id', 'entity', 'score', 'text']
    # print(df.head())
    
    # # I may need to make two splits, augmented and non
    # print(df['id'].count())
    # print(df['id'].nunique())
    
    # # Remove @s
    # df['test'] = df['text'].str.replace(r'@\S+', '', regex=True)
    # for i in range(5):
    #     print(df['text'][i])
    #     print(df['test'][i])
    #     print('---')
    
    


if __name__ == "__main__":
    # We can pass each csv
    file = ROOT / 'data' / 'testing.csv'
    
    preprocess(file)
    print(file)