import pandas as pd
import re
import spacy
import string

from typing import List, Dict


def clean_text(text: str) -> str:
    """Cleans text by removing mentions, URLs, special characters, and punctuation."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'@\w+', '', text)  
    text = re.sub(r'http\S+|www.\S+', '', text)  
    text = text.translate(str.maketrans('', '', string.punctuation))  
    text = text.strip().lower()
    return text

def anonymize_text(text: str, nlp, entity_types: List[str] = ["PERSON", "GPE", "LOC"]) -> str:
    """Replaces named entities (e.g., names, locations) with generic placeholders."""
    if not text:
        return ""
    
    doc = nlp(text)
    anonymized_text = text
    for ent in doc.ents:
        if ent.label_ in entity_types:
            anonymized_text = anonymized_text.replace(ent.text, f'[{ent.label_}]')
    return anonymized_text


def process_tv_show_data(data_df: pd.DataFrame, character_name: str, context_size: int = 7) -> pd.DataFrame:
    """
    Process TV show data with a context window of 4
    Handles cases where indices might not be continuous
    """
    contexted = []
    
    try:
        character_indices = data_df[data_df.speaker == character_name].index.tolist()
    except:
        # In case column name is different
        character_indices = data_df[data_df.author == character_name].index.tolist()
    
    
    for i in character_indices:
        row = []
        
        
        try:
            row.append(data_df.loc[i, "transcript"])
        except KeyError:
            # Try alternative column name
            row.append(data_df.loc[i, "quote"])
            
       
        if i < context_size:
            continue
            
        
        context_complete = True
        prev = i - 1
        context_count = 0
        
        
        while context_count < context_size - 1 and prev >= 0:
            try:
                
                if prev in data_df.index:
                    try:
                        row.append(data_df.loc[prev, "transcript"])
                    except KeyError:
                        # Try alternative column name
                        row.append(data_df.loc[prev, "quote"])
                    prev -= 1
                    context_count += 1
                else:
                    # Skip missing indices
                    prev -= 1
                    context_complete = False
            except:
                context_complete = False
                break
        
       
        if context_count == context_size - 1 and context_complete:
            contexted.append(row)
    
   
    columns = ['response'] + [f'context/{i}' for i in range(context_size - 1)]
    

    if contexted:
        result_df = pd.DataFrame.from_records(contexted, columns=columns)
        return result_df
    else:
      
        return pd.DataFrame(columns=columns)



def process_social_media_data(df: pd.DataFrame) -> pd.DataFrame:
    """Formats social media chat data for training."""
    
    nlp = spacy.load("en_core_web_sm")
    df["context/0"] = df["context/0"].apply(lambda x: anonymize_text(clean_text(x), nlp))
    df["response"] = df["response"].apply(lambda x: anonymize_text(clean_text(x), nlp))
    return df[["context/0", "response"]]

def prepare_dataset_for_alpaca(df: pd.DataFrame, context_columns: List[str], response_column: str) -> pd.DataFrame:
    """Converts dataset into Alpaca format for fine-tuning."""
    
    alpaca_format = []
    for _, row in df.iterrows():
        input_text = "\n".join([f"Message {i+1}: {row[col]}" for i, col in enumerate(context_columns) if pd.notna(row[col])])
        entry = {
            "instruction": "Respond to this message as if you were in a conversation.",
            "input": input_text,
            "output": row[response_column]
        }
        alpaca_format.append(entry)
    return pd.DataFrame(alpaca_format)

def main():
    """Loads and processes datasets before saving in training format."""
    
    try:
        tv_show_data = pd.read_csv("")
        social_media_data = pd.read_csv("")
        
        processed_tv_show_data = process_tv_show_data(tv_show_data, context_size=7)
        processed_social_media_data = process_social_media_data(social_media_data)
        
        alpaca_tv_show = prepare_dataset_for_alpaca(processed_tv_show_data, [f'context/{i}' for i in range(6)], 'response')
        alpaca_social_media = prepare_dataset_for_alpaca(processed_social_media_data, ["context/0"], "response")
        
        final_dataset = pd.concat([alpaca_tv_show, alpaca_social_media], ignore_index=True)
        final_dataset.to_csv("datasets/final_dataset.csv", index=False)
        
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
