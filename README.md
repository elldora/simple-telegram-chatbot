# Simple Telegram Chatbot

A Python-based Telegram chatbot that helps users analyze and understand data through natural language interactions. The bot can read CSV files, provide data summaries, create visualizations, and answer questions about the data.

## Features

- 📊 Data Analysis: Read and analyze CSV files
- 📈 Visualizations: Generate charts and graphs from data
- 💬 Natural Language Interface: Ask questions about your data in plain English
- 🤖 AI-Powered Responses: Uses LLM (Large Language Model) for intelligent responses
- 📝 Data Summaries: Get quick overviews of your datasets

## Prerequisites

- Python 3.7 or higher
- Telegram Bot Token (obtain from [@BotFather](https://t.me/botfather))
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/simple-telegram-chatbot.git
cd simple-telegram-chatbot
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv chatbot-env
source chatbot-env/bin/activate  # On Windows: chatbot-env\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a Telegram bot and get your token from [@BotFather](https://t.me/botfather)
2. Update the `TOKEN` variable in `main.py` with your bot token

## Usage

1. Start the bot:
```bash
python main.py
```

2. Available Commands:
- `/start` - Initialize the bot and get a welcome message
- `/help` - Get help and instructions
- `/readcsv <file_path>` - Read and analyze a CSV file
- `/qa <question>` - Ask questions about your data

## Project Structure

```
simple-telegram-chatbot/
├── main.py              # Main bot implementation
├── functions.py         # Helper functions
├── requirements.txt     # Project dependencies
├── data/               # Data directory
├── model/              # Model files
└── images/             # Generated visualizations
```

## Dependencies

- python-telegram-bot: Telegram Bot API wrapper
- pyTelegramBotAPI: Alternative Telegram Bot API wrapper
- llama-cpp-python: LLM integration
- ctransformers: Transformer model support
- ipywidgets: Interactive widgets for Jupyter
- seaborn: Statistical data visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)