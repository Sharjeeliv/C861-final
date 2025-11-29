import pandas as pd
import numpy as np
from pathlib import Path
import csv
import re
import emoji
import nltk


NTLK_PATH = "D:\\nltk_data"
nltk.data.path.append(NTLK_PATH)
nltk.download('punkt_tab', download_dir=NTLK_PATH)
nltk.download('stopwords', download_dir=NTLK_PATH)
nltk.download('wordnet', download_dir=NTLK_PATH)
nltk.download('averaged_perceptron_tagger_eng', download_dir=NTLK_PATH)
 
ROOT = Path.cwd()

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
    
from nltk import pos_tag


def clean_tweets(text: str) -> str:
    
    
    # Remove extra whitespace, unfirom spacing
    text = re.sub(r'\s+', ' ', text)
    # Remove links and urls
    RE_URL = re.compile(r"(https?://\S+|www\.\S+|(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}/\S*)", re.MULTILINE)
    text = re.sub(RE_URL, '', text)
    # Fix ellipsies
    text = re.sub(r'\.{2,}', '... ', text)
    text = re.sub(r'\’', '\'', text)
    
    
    # Handle emojis
    text = emoji.demojize(text) # delimiters=("","")
    # remove special characters
    # text = re.sub(r'[^A-Za-z0-9\s\'\.]', ' ', text)
    text = re.sub(r"[^A-Za-z0-9\s\'\._!?-]", " ", text)
    
    # Renormalize whitespace again
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([?.!,\'"](?:\s|$))', r'\1', text)  # remove space before punctuation
    
    return text.strip()


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):
    """
    Converts the NLTK POS tag (Treebank format) to the 
    single-character POS tag required by WordNetLemmatizer.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # Default to Noun if tag is not recognized or is a Noun
        return wordnet.NOUN

def process2(text: str) -> str:
    # Initialize necessary components
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    ## 1. Tokenization and Lowercasing
    tokens = word_tokenize(text.lower())
    
    ## 2. Stop Word Removal
    # This step is done here to keep the POS tagging accurate on only meaningful words
    tokens_no_stop = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    ## 3. POS Tagging
    # Gets the Treebank POS tag for each token (e.g., ('running', 'VBG'))
    tagged_tokens = nltk.pos_tag(tokens_no_stop)
    
    ## 4. Lemmatization with POS Tag
    processed_tokens = []
    for token, tag in tagged_tokens:
        # Convert Treebank tag to WordNet tag
        wntag = get_wordnet_pos(tag)
        
        # Lemmatize using the appropriate POS tag
        lemma = lemmatizer.lemmatize(token, pos=wntag)
        processed_tokens.append(lemma)
        
    ## 5. Join Tokens
    cleaned_text = ' '.join(processed_tokens)
    
    return cleaned_text

# def process2(text: str)-> str: 
    
#     tokens = word_tokenize(text.lower())
    
#     stop_words = set(stopwords.words('english'))
#     tokens = [token for token in tokens if token not in stop_words]
    
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]
#     cleaned_text = ' '.join(tokens)
    
#     return cleaned_text
    
    # Step 4: Tokenize
    

    
    # Step 5: Lowercase and remove stopwords
    
   
    
    # # Step 6: Lematize
    # 
    # lemmatizer = WordNetLemmatizer()
    # def process_tokens(tokens):
    #     processed = []
    #     for token in tokens:
    #         token = token.lower()
    #         if token not in stop_words:
    #             lemma = lemmatizer.lemmatize(token)
    #             processed.append(lemma)
    #     return processed
    
    # # Step 7: Join tokens back to text
    # df['text'] = df['tokens'].apply(process_tokens).apply(lambda tokens: ' '.join(tokens))


def preprocess(file_path: Path):
    # Extract data
    
    # Manual processing of csv:
    # First three commas correctly separate id, entity, score
    # We need to process text properly
    # Step 1 raw read
    rows = []
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        text = f.read()
        # All valid rows start with a number (id)
        raw_rows = text.split("\n")
        for row in raw_rows:
            tmp = row.strip()
            parts = row.strip().split(',', 3)
            if not tmp: continue
            if not tmp[0].isdigit() or len(parts) < 4:
                rows[-1][-1] += " " + tmp
                # print(f"Appending to previous row: {tmp}")
                continue
        
            id_, game, sentiment, text = parts
            rows.append([int(id_.strip()), game.strip(), sentiment.strip(), text.strip()])
    
    print(f"Total rows processed: {len(rows)}")
    df = pd.DataFrame(rows, columns=['id', 'entity', 'sentiment', 'text'])
        
    # for r in rows[:5]: print(r[3], '\n\n--')
    
    # Step 2: Clean 
    df['text'] = df['text'].apply(clean_tweets)
    
    # Step 3: Save cleaned data
    cleaned_file = file_path.parent / f"{file_path.stem}_cleaned.csv"
    df.to_csv(cleaned_file, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Cleaned data saved to: {cleaned_file}")
    
    # Processing step 2:
    df['text'] = df['text'].apply(process2)
    
    
    
    
    print("Sample cleaned texts:")
    for txt in df['text'].sample(10, random_state=1):
        print(f"- {txt}\n")
  
    


if __name__ == "__main__":
    # We can pass each csv
    file = ROOT / 'data' / 'training.csv'
    
    preprocess(file)
    
    
    print(file)