# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfPipeline(object):
    def __init__(self):
        self.documents = []
        self.urls = []
        self.links = []  # Use a list to store the links

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

        df = pd.DataFrame(td_matrix.T.toarray(), index=vectorizer.get_feature_names_out(), columns=self.urls)
        df = df.sort_index(axis=1)
        df.to_csv('data/td_matrix.csv')

        # Save the links to a separate CSV file
        links_df = pd.DataFrame(self.links, columns=['from', 'to'])
        links_df.to_csv('data/links.csv')
