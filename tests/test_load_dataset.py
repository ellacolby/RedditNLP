""" UnitTest """
import unittest
import logging

import tweetnlp
from tweetnlp.text_classification.dataset import DEFAULT_DATASETS_TEXT_CLASSIFICATION
from tweetnlp.ner.dataset import DEFAULT_DATASETS_NER

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


class Test(unittest.TestCase):
    """ Test """

    def test_classification(self):
        for k in DEFAULT_DATASETS_TEXT_CLASSIFICATION.keys():
            print(k)
            tweetnlp.load_dataset(k)
            tweetnlp.load_dataset_text_classification(k)
            if k == 'sentiment':
                for l in ['arabic', 'english', 'french', 'german', 'hindi', 'italian', 'portuguese', 'spanish']:
                    tweetnlp.load_dataset(k, multilingual=True, task_language=l)
                    tweetnlp.load_dataset_text_classification(k, multilingual=True, task_language=l)
            elif k == 'topic_classification':
                tweetnlp.load_dataset(k, multi_label=True)
                tweetnlp.load_dataset_text_classification(k, multi_label=True)

    def test_ner(self):
        for k in DEFAULT_DATASETS_NER.keys():
            print(k)
            tweetnlp.load_dataset(k)
            tweetnlp.load_dataset_ner(k)


if __name__ == "__main__":
    unittest.main()
