import hashlib
import argparse
import os
import json
from  nltk import sent_tokenize
import csv

summary_files = [
    'test_chen18_org.json',
    'test_gehrmann18_org.json',
    'test_see17_org.json'
]

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_path', default=None, help='folder path which contains the summary files')
    parser.add_argument('--article_path', default=None, help='path of the whole article file')
    parser.add_argument('--url_path', default=None, help='folder path of the url file')
    args = parser.parse_args()

    url_hex_list, summary_hex_list = [], []
    articles = []
    article_source = []

    with open(args.url_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        urls = [str(_).strip() for _ in lines]
    for _ in urls:
        url_hex_list.append(hashhex(_))
    
    with open(args.article_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        article_source = list(lines)


    for i, file_name in enumerate(summary_files):
        dir_name = os.path.join(args.summary_path, file_name.split('.')[0])
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        summaries = []
        with open(os.path.join(args.summary_path, summary_files[i]), 'r', encoding='utf-8') as f:
            summary_content = json.load(f)
        for _ in summary_content.keys():
            try:
                temp_index = url_hex_list.index(_)
            except ValueError:
                print(url_hex_list[:10])
                raise KeyboardInterrupt

            if len(articles) != len(summary_content):
                articles.append(sent_tokenize(article_source[temp_index]))
            summaries.append([])
            for j in range(10):
                if summary_content[_]['sents'].get(str(j)) == None:
                    break
                summaries[-1].append(summary_content[_]['sents'][str(j)]['text'])
        with open(os.path.join(dir_name, 'articles.tsv'), 'w', encoding='utf-8') as f:
            f_tsv = csv.writer(f, delimiter='\t')
            for _ in articles:
                f_tsv.writerow(_)
        with open(os.path.join(dir_name, 'summaries.tsv'), 'w', encoding='utf-8') as f:
            f_tsv = csv.writer(f, delimiter='\t')
            for _ in summaries:
                f_tsv.writerow(_)
    
    
    