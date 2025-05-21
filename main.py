from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import logging
import csv
import pandas as pd

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = 'YOUR_BOT_TOKEN'  # Replace with your actual token

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"User {update.effective_user.id} started the bot")
    await update.message.reply_text('Hello! I am your bot. How can I help you today?')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"User {update.effective_user.id} requested help")
    await update.message.reply_text('Help! I need somebody! Help! Not just anybody!')

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
        df = pd.read_csv(file_path)
        headers = df.columns.tolist()
        message = "Column names:\n" + ", ".join(headers)
        await update.message.reply_text(message)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        await update.message.reply_text("Error reading CSV file.")

def main():
    logger.info("Starting bot...")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("readcsv", read_csv))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    app.run_polling()
    logger.info("Bot stopped.")

if __name__ == '__main__':
    main()