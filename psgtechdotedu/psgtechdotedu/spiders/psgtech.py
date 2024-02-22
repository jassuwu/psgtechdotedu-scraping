# import json
# import scrapy
# from pathlib import Path

# from .constants import STOP_URLS

# class PsgtechSpider(scrapy.Spider):
#     name = "psgtech"
#     allowed_domains = ["psgtech.edu"]
#     start_urls = ["https://psgtech.edu"]
#     prev_url_set = set()
#     prev_joint_url_set = set()

#     def parse(self, response):

#         # Parse the URLS
#         urls = response.css("a::attr(href)").getall()

#         # Remove the stop urls
#         urls = self.get_next_urls(response, urls)

#         # Stringify and dump
#         json_dump = json.dumps(urls)
#         Path('homepage.json').write_text(json_dump)
        
#         self.log(f"Saved the file.")

#     def get_next_urls(self, response, url_list):
#         processed_url_list = []
#         for url in url_list:
#             joint_url = response.urljoin(url)
#             if url not in STOP_URLS and url not in self.prev_url_set and joint_url not in self.prev_joint_url_set:
#                 self.prev_url_set.add(url)
#                 self.prev_joint_url_set.add(joint_url)
#                 processed_url_list.append(joint_url)
#         return processed_url_list

from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

class PsgtechSpider(CrawlSpider):
    name = 'psgtech'
    allowed_domains = ['psgtech.edu']
    start_urls = ['https://psgtech.edu']

    # Define the rules for the spider
    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        # Extract the URL of the current page
        page_url = response.url

        # Extract the title of the current page
        page_title = response.css('title::text').get()

        # Extract all text within <p>, <h1>, <h2>, <h3>, <h4>, <h5>, <h6>, <li>, <a>, and <div> tags
        # Might have to remove the <a> from here, it affects the results
        page_text = ' '.join(response.css('p::text, h1::text, h2::text, h3::text, h4::text, h5::text, h6::text, li::text, div::text').getall())

        # Return the data as an item
        yield {
            'url': page_url,
            'title': page_title,
            'text': page_text,
        }
