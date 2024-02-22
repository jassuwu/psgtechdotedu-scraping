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

    def process_item(self, item, spider):
        # Add the document text to the list of documents
        joint_document = ' '.join(item['text'])
        self.documents.append(joint_document)
        # Add the url to the list of urls
        self.urls.append(item['url'])
        item['text'] = joint_document
        return item

    def close_spider(self, spider):
        # Calculate the TD matrix
        vectorizer = TfidfVectorizer()
        td_matrix = vectorizer.fit_transform(self.documents)

        # Convert the TD matrix to a DataFrame
        df = pd.DataFrame(td_matrix.T.toarray(), index=vectorizer.get_feature_names_out(), columns=self.urls)

        # Sort the DataFrame columns in alphabetical order
        df = df.sort_index(axis=1)

        # Write the DataFrame to a CSV file
        df.to_csv('td_matrix.csv')

