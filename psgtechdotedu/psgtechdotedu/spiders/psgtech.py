import re
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from urllib.parse import urlparse

from .constants import STOP_WORDS

class PsgtechSpider(CrawlSpider):
    name = 'psgtech'
    allowed_domains = ['psgtech.edu']
    start_urls = ['https://psgtech.edu']
    stemmer = PorterStemmer()

    # Define the rules for the spider
    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        page_url = response.url
        page_title = response.css('title::text').get()
        page_text = ' '.join(response.css('p::text, h1::text, h2::text, h3::text, h4::text, h5::text, h6::text, li::text, a::text, div::text').getall())

        punctuation_removed_title = re.sub(r'[^\w\s]', '', page_title)
        clean_title = re.sub(r'\s+',' ',punctuation_removed_title.strip())
        processed_title = [self.stemmer.stem(token.lower()) for token in word_tokenize(clean_title) if token.lower() not in STOP_WORDS]

        punctuation_removed_text = re.sub(r'[^\w\s]', '', page_text)
        clean_text = re.sub(r'\s+',' ',punctuation_removed_text.strip())
        processed_text = [self.stemmer.stem(token.lower()) for token in word_tokenize(clean_text) if token.lower() not in STOP_WORDS]

        parsed_url = urlparse(page_url)
        url_parts = re.split(r'[/.\-_]', parsed_url.netloc + parsed_url.path)
        processed_url = [self.stemmer.stem(part.lower()) for part in url_parts if part and part.lower() not in STOP_WORDS]

        processed_text += processed_title
        processed_text += processed_url

        yield {
            'url': page_url,
            'title': processed_title,
            'text': processed_text,
        }
