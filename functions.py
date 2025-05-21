from ctransformers import AutoModelForCausalLM
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import re
import time

def read_model(model_path = ''):
    llm = AutoModelForCausalLM.from_pretrained(
        "./model/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        model_type="llama" 
    )
    return llm


def simple_qa(model, question = "Explain GGUF files"):
    answer = model(f"Q: {question}. A:")
    return answer

def read_dataset(df_path = './data/df_final.csv'):
    df = pd.read_csv(f'{df_path}')
    return df

def create_prompt(instruction, context=None):
    if context:
        return f"""<|system|>
                    You are a helpful AI assistant that analyzes data. Use the provided context to answer questions.</s>
                    <|user|>
                    Context: {context}
                    Question: {instruction}</s>
                    <|assistant|>
                """
    else:
        return f"""<|system|>
                    You are a helpful AI assistant that analyzes data.</s>
                    <|user|>
                    {instruction}</s>
                    <|assistant|>
                """
    
def ask_llm(model, prompt, max_tokens=400):
    response = model(
        prompt,
        max_new_tokens=max_tokens,
        temperature=0.2,
        repetition_penalty=1.1
    )
    return response

def get_dataframe_summary(df):
    summary = f"""
    DataFrame shape: {len(df)} rows, {len(df.columns)} columns
    Columns: {', '.join(df.columns)}
    First row sample: {df.iloc[0].to_dict()}
    """
    
    prompt = create_prompt(
        "Please summarize the key characteristics of this dataset.",
        context=summary
    )
    
    if len(prompt.split()) > 400: 
        return "Data too large to process. Please try with a smaller subset."
    
    return ask_llm(prompt)

def get_efficient_dataframe_summary(df):
    stats = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": {col: df[col].isna().sum() for col in df.columns},
    }
    
    context = f"""
    Dataset Statistics:
    - Shape: {stats['shape']}
    - Columns: {', '.join(stats['columns'][:5])}{'...' if len(stats['columns']) > 5 else ''}
    - Dtypes: {', '.join([f'{k}:{v}' for k,v in list(stats['dtypes'].items())[:3]])}{'...' if len(stats['dtypes']) > 3 else ''}
    """
    
    prompt = create_prompt(
        "Based on these statistics, what can you tell me about this dataset?.",
        context=context
    )
    
    return ask_llm(prompt)

def analyze_column(df, column_name):
    if column_name not in df.columns:
        return f"Column {column_name} not found in DataFrame."
    
    col_summary = f"""
    Column '{column_name}' statistics:
    - Data type: {df[column_name].dtype}
    - Unique values: {df[column_name].unique()}
    - Missing values: {df[column_name].isna().sum()}
    - Max value: {df[column_name].max()}
    - Min value: {df[column_name].min()}
    """
    
    prompt = create_prompt(
        f"Analyze the '{column_name}' column and provide insights about its distribution.",
        context=col_summary
    )
    
    return ask_llm(prompt)


def draw_piechart(df, feature_name):
    feature_counts = df[feature_name].value_counts()
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.pie(feature_counts, 
        labels=feature_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        explode=[0.05] * len(feature_counts))

    ax.set_title(f'Distribution of {feature_name}', pad=20)
    ax.axis('equal')
    image_path = f'./images/{feature_name}.png'
    fig.savefig(image_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return image_path

def load_pie_chart(image_path):
    image = Image.open(image_path)
    return image