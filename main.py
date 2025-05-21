from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import logging
import csv
import pandas as pd
from functions import *


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = '7743361358:AAFNspdDcpx1fsgxypR5rIMmrqnMSFYh3To'

MODEL = read_model('')
dataset_file_path = './data/df_final.csv'

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"User {update.effective_user.id} started the bot")
    prompt = create_prompt(
        instruction="""Introduce yourself as an AI assistant that helps users analyze and understand data. 
        Mention your key capabilities:
        - Analyzing CSV files
        - Providing data summaries
        - Creating visualizations
        - Answering questions about data
        Keep it brief and friendly.""",
        context="Initial greeting"
    )
    answer = ask_llm(MODEL, prompt=prompt)
    await update.message.reply_text(f'{answer}')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"User {update.effective_user.id} requested help")
    prompt = create_prompt(
        instruction="""Help! I need somebody!
""",
context="Help the user with instructions"
    )
    answer = ask_llm(MODEL, prompt=prompt)

    await update.message.reply_text(answer)

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"User {update.effective_user.id} sent message: {update.message.text}")
    await update.message.reply_text(update.message.text)

async def read_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"User {update.effective_user.id} requested to read CSV file")
    if not context.args:
        await update.message.reply_text("Please provide a file path. Example: /read_csv ./data/df_final.csv")
        return
    file_path = context.args[0]
    try:
        df = read_dataset(file_path)
        t = type(df)
        prompt = get_dataframe_summary(df)
        answer = ask_llm(MODEL, prompt=prompt)
        await update.message.reply_text(answer)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        await update.message.reply_text("Error reading CSV file.")

async def question_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"User {update.effective_user.id} requested to read CSV file")
    if not context.args:
        await update.message.reply_text("Please ask a question, e.g. What is the distribution of education levels among individuals?")
        return
    
    question = " ".join(context.args)
    file_path = dataset_file_path
    try:
        df = pd.read_csv(file_path)
        col_name = get_question_keyword(question)
        
        prompt = analyze_categorical_column(df, column_name=col_name)
        answer = ask_llm(MODEL, prompt)        

        try:
            image_path = draw_piechart(df, col_name)
            await update.message.reply_photo(photo=open(image_path, 'rb'))
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            await update.message.reply_text("Could not create visualization for this data.")
        
        await update.message.reply_text(answer)
        
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        await update.message.reply_text("Error reading CSV file.")

def main():
    logger.info("Starting bot...")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("readcsv", read_csv))
    app.add_handler(CommandHandler("qa", question_answer))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    app.run_polling()
    logger.info("Bot stopped.")

if __name__ == '__main__':
    main()