import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from elasticsearch import Elasticsearch
from loguru import logger
import numpy as np
import os 


class Projector:
    def __init__(self, model_name="openai/clip-vit-base-patch16", es_host='es-test.aws.primehub.io', 
                 es_port=9200, es_scheme='http', es_http_auth=('elastic', 'K'), 
                 index_name='image-text-search-v1', folder_path='./'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.current_image = None

        # Connect to Elasticsearch
        self.es = Elasticsearch([{'host': es_host, 'port': es_port, 'scheme': es_scheme}],
                           http_auth=es_http_auth)

        # Define the index settings and mappings
        self.index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "smartcn"
                    },
                    "content_vector": {
                        "type": "dense_vector",
                        "dims": 512  # Replace with the actual number of dimensions
                    }
                }
            }
        }
        self.index_name = index_name
        self.folder_path = folder_path

    def display_most_similar_image(self, text):
        inputs = self.tokenizer([text], truncation=True, return_tensors="pt").to(self.device)
        vector = self.model.get_text_features(**inputs).cpu().detach().numpy()

        search_query = {
            "query": {
                "knn": {
                    "field": "content_vector",
                    "query_vector": vector.tolist()[0],
                    "num_candidates": 1
                }
            }
        }

        response = self.es.search(index=self.index_name, body=search_query)
        image_path = response['hits']['hits'][0]['_source']['content']

        if self.current_image:
            self.current_image.close()
        logger.error(f"Image path: {image_path}")
        self.current_image = Image.open(image_path)
        self.current_image.show()