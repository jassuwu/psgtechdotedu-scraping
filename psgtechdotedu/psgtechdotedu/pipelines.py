# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import json
from collections import defaultdict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfPipeline(object):
    def __init__(self):
        self.documents = []
        self.urls = []
        self.links = []  # Use a list to store the links
        self.inverted_index = defaultdict(list)

    def process_item(self, item, spider):
        joint_document = ' '.join(item['text'])
        self.documents.append(joint_document)
        self.urls.append(item['url'])
        for link in item['links']:  # Add each link to the list
            self.links.append((item['url'], link))
        item['text'] = joint_document
        return item

    def close_spider(self, spider):
        vectorizer = TfidfVectorizer()
        td_matrix = vectorizer.fit_transform(self.documents)
        feature_names = vectorizer.get_feature_names_out()

        # Build the inverted index with TF-IDF weights
        for doc_id, doc in enumerate(td_matrix.toarray()):
            for term_id, weight in enumerate(doc):
                if weight > 0:
                    self.inverted_index[feature_names[term_id]].append((self.urls[doc_id], weight))
        # Save the inverted index to a file
        with open('data/inverted_index.json', 'w') as f:
            json.dump(self.inverted_index, f)

        # Save the links to a separate CSV file
        links_df = pd.DataFrame(self.links, columns=['from', 'to'])
        links_df.to_csv('data/links.csv')