Please note that this is the Backend and scraper repository. For information about the frontend, please refer to the [corresponding repository](https://github.com/jassuwu/search-psgtech).

# PSGoogle: An Institutional Search Engine

## Overview

PSGoogle is an institutional search engine designed specifically for PSG College of Technology. The goal of this project is to make the vast amount of information available on [psgtech.edu](https://psgtech.edu) easily accessible and searchable. This is achieved by implementing various concepts and techniques in Information Retrieval and Web Search.

## Why PSGoogle?

The official website of PSG College of Technology, [psgtech.edu](https://psgtech.edu), holds a wealth of information about the institution. However, the absence of a search bar makes finding specific information a hassle. PSGoogle aims to solve this problem by providing a user-friendly search interface.

## Data Storage Structure

The scraped dataset for this project consists of over 550 documents, with more than 8500 unique tokens. This is the rough storage structure (TS interface is used to show the format only):

```ts
interface PageDocument {
  url: string;
  title: string;
  processedText: string;
  links: string[];
}
```

## Implementation

The implementation of PSGoogle involves several steps:

1. **Scraping**: The web pages of psgtech.edu are scraped using Scrapy, a popular library for web scraping.

2. **Extracting and Processing**: The metadata of the web pages, such as title, URL, and keywords, are extracted using Scrapy and stored in a JSON format.

3. **Indexing**: The web pages and the metadata are indexed using a Term-Document matrix structure with TF-IDF weighting. The inverted index is constructed using the `TfidfVectorizer` from the sklearn.feature_extraction.text library and stored as a JSON file.

4. **Ranking**: The web pages are ranked based on the user query using a vector space model with cosine similarity.

The backend of PSGoogle is implemented using `FastAPI` and hosted on Render.

## Getting Started

To dive into PSGoogle, follow these steps:

### **Environment Setup**

Make sure you have Python installed on your machine. For running this project locally, we recommend setting up a virtual environment.

1. **Clone the Repository**: Clone this repository to your local machine using `git clone <repository_url>`.

2. **Run the Initialization Script**: Execute the `init.sh` script by running `./init.sh` in your command line. This script automates the setup process, including installing required packages, downloading necessary nltk data, and running the scraper if the JSON file does not exist.

3. **Start the Server**: Start the server by running `uvicorn main.py:app --reload` in your command line to start the server in debug mode. The server will start running on `http://localhost:8000`.

4. **Search for Documents**: You can now search for documents by sending a GET request to `http://localhost:8000/search?q=<your_query>` where `<your_query>` is the query you want to search for. The server will return a list of the top 50 relevant documents for your query.

5. **Explore the Swagger Documentation**: The FastAPI server provides Swagger documentation at the `/docs` route, which provides a user-friendly interface for testing the API endpoints.

## Contributions

PSGoogle is hardly feature complete, unfortunately. So, if you have any bugs to report or feature to work on, feel free to open an issue or submit a pull request.
