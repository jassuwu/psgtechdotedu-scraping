#!/bin/bash
# A script to start up a web service with a scraper and a fastapi server

# Install the required packages
pip install -r requirements.txt

# Download the nltk data
python -c "import nltk; nltk.download('punkt')"

# Run the scraper if the JSON file does not exist
if [ ! -f psgtechdotedu/data/psgtech.json ]; then
  # Change directory to psgtechdotedu
  cd psgtechdotedu
  scrapy crawl psgtech -O data/psgtech.json
fi

