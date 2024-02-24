#!/bin/bash
# A script to start up a web service with a scraper and a fastapi server

# Install the required packages
echo "Installing required packages..."
pip install -r requirements.txt && echo "Packages installed successfully." || { echo "Package installation failed. Exiting script."; exit 1; }

# Download the nltk data
echo "Downloading nltk data..."
python -c "import nltk; nltk.download('punkt')" && echo "nltk data downloaded successfully." || { echo "nltk data download failed. Exiting script."; exit 1; }

# Check if the nltk data has been downloaded successfully
if [ ! -f /opt/render/nltk_data/tokenizers/punkt/PY3/english.pickle ] && [ ! -f /home/jass/nltk_data/tokenizers/punkt/PY3/english.pickle ]; then
  echo "nltk data download failed. Exiting script."
  exit 1
fi

cd psgtechdotedu
mkdir data
cd ..

# Run the scraper if the JSON file does not exist
if [ ! -f psgtechdotedu/data/psgtech.json ]; then
  # Change directory to psgtechdotedu
  cd psgtechdotedu
  echo "Running the scraper..."
  scrapy crawl psgtech -O data/psgtech.json && echo "Scraper ran successfully." || { echo "Scraper failed. Exiting script."; exit 1; }
fi

