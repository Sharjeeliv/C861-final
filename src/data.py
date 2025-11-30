
from typing import List
from pathlib import Path
import csv
import re

import emoji
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from .utils import time_execution

# ********************************
# VARIABLE AND SETUP
# ********************************
# NLTK PATH SETUPS
WIN_PATH =  Path("D:\\nltk_data")
NLTK_PATH = WIN_PATH if WIN_PATH.exists() else None
if NLTK_PATH: nltk.data.path.append(NLTK_PATH)

# NLTK DATA INSTALLATION
nltk.download('punkt_tab', download_dir=NLTK_PATH)
nltk.download('stopwords', download_dir=NLTK_PATH)
nltk.download('wordnet', download_dir=NLTK_PATH)
nltk.download('averaged_perceptron_tagger_eng', download_dir=NLTK_PATH)

# CONST 
STOP_WORDS = set(stopwords.words('english'))
ROOT = Path.cwd()

# REGEXES
RE_URL = re.compile(r"(https?://\S+|www\.\S+|(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}/\S*)", re.MULTILINE)
RE_RM_BAD_CHARS = re.compile(r"[^A-Za-z0-9\s\'\._!?-]")
RE_RM_EXTRA_SPC = re.compile(r'\s+([?.!,\'"](?:\s|$))')
 
 
# ********************************
# HELPER FUNCTIONS
# ********************************  
def _clean_text(text: str) -> str:
    # Text Cleaning and Normalization
    print("Data cleaning...")
    text = re.sub(r'\s+', ' ', text)            # 1. Normalize whitespaces
    text = re.sub(RE_URL, '', text)             # 2. Remove URLs and links
    # Punctuation Correction
    text = re.sub(r'\.{2,}', '... ', text)      # 3. Fix ellipsies
    text = re.sub(r'\’', '\'', text)            # 4. Fix apostrophes
    # Special Character Handling
    text = emoji.demojize(text)                 # 5. Emoji to text-form
    text = re.sub(RE_RM_BAD_CHARS, " ", text)   # 6. Remove special characters
    # Spacing Renormalization
    text = re.sub(r'\s+', ' ', text)            # 7. Renormalize spacing 
    text = re.sub(RE_RM_EXTRA_SPC, r'\1', text) # 8. Remove exta punc. space
    return text.strip()


def _get_pos(tag: str):
    # Convert NLTK POS tag (Treebank format) to
    # single-char POS tag needed by WordNetLemmatizer
    if tag.startswith('J'):     return wordnet.ADJ
    elif tag.startswith('V'):   return wordnet.VERB
    elif tag.startswith('R'):   return wordnet.ADV
    # Default to Noun if tag is not recognized or is a Noun
    else:                       return wordnet.NOUN
        

def _lemmatize(tokens: List[str]) -> List[str]:
    lemmatizer = WordNetLemmatizer()
    tokens_tagged = nltk.pos_tag(tokens)
    # Use pos (for better perf.)
    tokens_res = []
    for token, tag in tokens_tagged:
        tag = _get_pos(tag)
        lemma = lemmatizer.lemmatize(token, pos=tag)
        tokens_res.append(lemma)
    return tokens_res


def _rm_stopwords(tokens: List[str]) -> List[str]:
    res_tokens = []
    for token in tokens:
        if not token.isalpha(): continue
        if token in STOP_WORDS: continue
        res_tokens.append(token)
    return res_tokens
    

def _preprocess_text(text):
    # Multi-step text preprocessing
    # Pending control mechanism
    print("Data preprocessing...")
    tokens = word_tokenize(text.lower())    # 1. Lowercase and tokenize text
    tokens = _rm_stopwords(tokens)          # 2. Remove stop words
    tokens = _lemmatize(tokens)             # 3. POS and lemmatize (get base form)
    res_txt = ' '.join(tokens)              # 4. Rejoin tokens into text
    return res_txt


def _load_data_raw(path: Path | str):
    # Raw tweets are extremely malformed, we thus
    # process it manually using heuristics ot extract
    # the text data and then extensively processing it
    print("Data loading...")
    N_COL, rows = 4, []
    
    # File opening and reading
    f = open(path, "r", encoding="utf-8")
    raw_text = f.read()
    raw_rows = raw_text.split("\n")
    
    # Malformed text handling, iterate all raw rows
    for row in raw_rows:
        # Heuristic-based splitting
        text = row.strip()
        parts = text.split(',', N_COL-1)
        if not text: continue
        # Handle malform text by appending to prev row
        if not text[0].isdigit() or len(parts) < N_COL:
            rows[-1][-1] += " " + text
            continue
        # Append only VALID rows as new rows
        id_, game, sentiment, text = parts
        rows.append([int(id_.strip()), game.strip(), sentiment.strip(), text.strip()])
    
    # CLosing and conversion to df
    f.close()
    print(f"Rows loaded: {len(rows)}")
    df = pd.DataFrame(rows, columns=['id', 'entity', 'sentiment', 'text'])
    return df
    

# ********************************
# UTILITY FUNCTIONS
# ********************************  
def _save_data_csv(file_path: Path, suffix: str, df: pd.DataFrame):
    # Setup path
    out_dir = file_path.parent / "processed"
    if not out_dir.exists(): Path.mkdir(out_dir)
    path = out_dir / f"{file_path.stem}_{suffix}.csv"
    # Save data
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Saved to {path}")
    
    
def _load_data_csv(file_path: Path, suffix: str) -> pd.DataFrame | None:
    path = file_path.parent / "processed" / f"{file_path.stem}_{suffix}.csv"
    if not path.exists(): return 
    df = pd.read_csv(path, quoting=csv.QUOTE_MINIMAL)
    return df


def _print_samples(df: pd.DataFrame, n=10):
    print("Sample cleaned texts:")
    for i, text in enumerate(df['text'].sample(n, random_state=1)):
        print(f"Tweet {i}:\n{text}\n")

# ********************************
# MAIN INTERFACE FUNCTION
# ********************************
@time_execution
def preprocess(path: Path | str, run_all = False):
    print("Step: Preprocessing...")
    # Guard clause
    path = path if isinstance(path, Path) else Path(path)
    if not path.exists(): raise Exception(f"Path does not exit! {path}")
    
    clean_sfx, preproc_sfx = 'cleaned', 'processed'
    if run_all: print("Rerunning all steps!")
    
    # load cached data
    df = _load_data_csv(path, preproc_sfx)
    if not run_all and df is not None: return df
    
    # Load partially cached
    df = _load_data_csv(path, clean_sfx)
    if run_all or df is None: 
        # If unavailable, recompute
        df = _load_data_raw(path)
        df['text'] = df['text'].apply(_clean_text)
        _save_data_csv(path, clean_sfx, df)
    # Apply missing preprocessing steps
    df['text'] = df['text'].apply(_preprocess_text)
    _save_data_csv(path, preproc_sfx, df)
    return df
    
    
# ********************************
# MAIN AND TESTING FUNCTIONS
# ********************************
if __name__ == "__main__":

    file = ROOT / 'data' / 'training.csv'
    df = preprocess(file)
    _print_samples(df, 10)