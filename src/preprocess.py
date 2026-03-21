import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import re

CONFIG = {
    'data_dir': r'C:\Users\Lenovo\Desktop\PD ML SET 2\data\archive (7)\PubMed_20k_RCT',
    'output_dir': r'C:\Users\Lenovo\Desktop\PD ML SET 2\data',
    'random_state': 42,
    'total_samples': 20000
}

def load_data():
    dfs = []
    for split in ['train.csv', 'dev.csv', 'test.csv']:
        path = os.path.join(CONFIG['data_dir'], split)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df_combined)} lines from dataset.")
    return df_combined

def reconstruct_abstracts(df):
    print("Reconstructing full abstracts...")
    df = df.sort_values(by=['abstract_id', 'line_number'])
    reconstructed = df.groupby('abstract_id')['abstract_text'].apply(lambda x: ' '.join(x)).reset_index()
    print(f"Reconstructed {len(reconstructed)} full abstracts.")
    return reconstructed

def map_labels(df):
    print("Mapping labels using keyword heuristics...")
    
    treatment_kw = ['intervention', 'trial', 'therapy', 'drug', 'dose', 'randomised', 'placebo', 'administered', 'efficacy']
    diagnosis_kw = ['screening', 'diagnostic', 'sensitivity', 'specificity', 'test', 'accuracy', 'imaging', 'biopsy', 'detection']
    prevention_kw = ['vaccine', 'prevention', 'prophylaxis', 'risk reduction', 'protective', 'immunisation', 'incidence']
    
    labels = []
    all_kws = treatment_kw + diagnosis_kw + prevention_kw + ['diagnosis', 'treatment', 'prevention']
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, all_kws)) + r')\b', flags=re.IGNORECASE)
    cleaned_texts = []
    
    for text in df['abstract_text']:
        text_lower = text.lower()
        t_count = sum(text_lower.count(kw) for kw in treatment_kw)
        d_count = sum(text_lower.count(kw) for kw in diagnosis_kw)
        p_count = sum(text_lower.count(kw) for kw in prevention_kw)
        
        counts = {'Treatment': t_count, 'Diagnosis': d_count, 'Prevention': p_count}
        max_count = max(counts.values())
        
        if max_count == 0:
            labels.append('Treatment')
        else:
            top_classes = [k for k, v in counts.items() if v == max_count]
            if len(top_classes) > 1:
                labels.append('Treatment')
            else:
                labels.append(top_classes[0])
                
        clean_text = pattern.sub('', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        cleaned_texts.append(clean_text)
                
    df['label'] = labels
    df['abstract_text'] = cleaned_texts
    return df

def balance_dataset(df, target_size=20000):
    print("Class distribution BEFORE balancing:")
    print(df['label'].value_counts())
    
    classes = df['label'].unique()
    samples_per_class = target_size // len(classes)
    remainder = target_size % len(classes)
    
    balanced_dfs = []
    for i, c in enumerate(classes):
        n_samples = samples_per_class + (1 if i < remainder else 0)
        class_df = df[df['label'] == c]
        
        replace = n_samples > len(class_df)
        sampled = class_df.sample(n=n_samples, replace=replace, random_state=CONFIG['random_state'])
        balanced_dfs.append(sampled)
        
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=CONFIG['random_state']).reset_index(drop=True)
    
    print("\nClass distribution AFTER balancing:")
    print(balanced_df['label'].value_counts())
    
    return balanced_df

def main():
    df_lines = load_data()
    df_abstracts = reconstruct_abstracts(df_lines)
    df_labeled = map_labels(df_abstracts)
    df_balanced = balance_dataset(df_labeled, target_size=CONFIG['total_samples'])
    
    output_path = os.path.join(CONFIG['output_dir'], 'processed_abstracts.csv')
    df_balanced.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    main()
