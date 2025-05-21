from ctransformers import AutoModelForCausalLM
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import json
import re
import time

bot_name = "ElldoraBot"

def read_model(model_path_or_repo_id = None):
    llm = AutoModelForCausalLM.from_pretrained(
        model_path_or_repo_id='./model/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
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
You are a helpful AI assistant named {bot_name} that analyzes data. Use the provided context to answer questions.</s>
<|user|>
Context: {context}
Request: {instruction}</s>
<|assistant|>"""
    else:
        return f"""<|system|>
You are a helpful AI assistant named {bot_name} that analyzes data. Provide clear and direct responses.</s>
<|user|>
{instruction}</s>
<|assistant|>"""
    
def ask_llm(model, prompt, max_tokens=400):
    response = model(
        prompt,
        max_new_tokens=max_tokens,
        temperature=0.7,
        repetition_penalty=1.1
    )
    return response

def get_dataframe_column_name(model, df, question):
    column_name = ""
    
    while column_name not in df.columns:
        
        context = f"""
        Available columns names in the dataset:
        {', '.join(df.columns)}
        
        Question to analyze: {question}
        """
        
        prompt = create_prompt(
            instruction="""Based on the question and the available columns, identify the most relevant column name.
            Rules:
            1. Return ONLY the exact column name from the provided list
            2. If the question mentions multiple columns, return the main column being analyzed
            3. If you're unsure, return the column that best matches the question's intent
            4. Do not add any explanations or additional text
            5. The response should be a single column name exactly as it appears in the list
            6. The response must be one word 
            """,
            context=context
        )
        
        column_name = ask_llm(model, prompt).strip()
        
        column_name = column_name.strip('"\'')
        column_name = column_name.strip()
        
        if ' ' in column_name:
            for col in df.columns:
                
                if col.lower() == column_name.lower():
                    column_name = col
                    break
    
    if column_name not in df.columns:
        column_name = df.columns[0]
    
    return column_name

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
    
    return prompt

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

def analyze_numeric_column(df, column_name):
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
    
    return prompt

def analyze_categorical_column(df, column_name):
    if column_name not in df.columns:
        return f"Column {column_name} not found in DataFrame."
    
    col_summary = f"""
    Column '{column_name}' statistics:
    - Data type: {df[column_name].dtype}
    - Unique values: {df[column_name].unique()}
    - Missing values: {df[column_name].isna().sum()}
    
    """
    
    prompt = create_prompt(
        f"Analyze the '{column_name}' column and provide insights about its distribution.",
        context=col_summary
    )
    
    return prompt


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

def get_question_keyword(question):

    qa_keyword_dict = {
        "What is the distribution of education levels among individuals?": "Educations",
        "What are the age ranges in the dataset?": "Age",
        "What percentage of people are single vs. married?": "Marriage",
        "How many are currently employed vs. have left the organization?": "isactive",
        "What is the gender distribution?": "Sex",
    }
    return qa_keyword_dict.get(question, "UNDEFINED")

def load_config(file_path, source):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config[source]