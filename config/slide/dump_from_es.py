import os
from pathlib import Path

import jsonlines
import pandas as pda
import spacy
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from nltk.tokenize import NLTKWordTokenizer
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')
toker = NLTKWordTokenizer()

client = Elasticsearch(['localhost:9210', 'localhost:9211', 'localhost:9212'],
                       # sniff before doing anything
                       sniff_on_start=True,
                       # refresh nodes after a node fails to respond
                       sniff_on_connection_fail=True,
                       # and also every 60 seconds
                       sniffer_timeout=60,
                       timeout=600)


# client = Elasticsearch(timeout=600)


def dump_idiom(word, dump_file, index):
    body = {
        "_source": ["text"],
        "highlight": {
            "fields": {
                "text": {
                    "require_field_match": False,
                    "fragment_size": 400,
                    "number_of_fragments": 1,
                    "no_match_size": 20
                }
            }
        }
    }

    if "one's" in word.split() or "someone's" in word.split() or "anyone's" in word.split():
        query = {
            "bool": {
                "should": [
                ]
            }
        }

        if "one's" in word.split():
            key = "one's"
        elif "someone's" in word.split():
            key = "someone's"
        elif "anyone's" in word.split():
            key = "anyone's"

        for po in ['my', 'your', 'his', 'her', 'our', 'their']:
            query['bool']['should'].append({
                "match_phrase": {
                    "text": word.replace(key, po)
                }
            })
    elif "someone" in word.split() or "anyone" in word.split():
        query = {
            "bool": {
                "should": [
                ]
            }
        }

        if "someone" in word.split():
            key = "someone"
        elif "anyone" in word.split():
            key = "anyone"

        doc = nlp(word)
        idx = [t.i for t in doc if t.text in ['someone', 'anyone']][0]
        if doc[idx - 1].tag_.startswith('V'):
            for po in ['me', 'you', 'him', 'her', 'us', 'them']:
                query['bool']['should'].append({
                    "match_phrase": {
                        "text": word.replace(key, po)
                    }
                })
        else:
            for po in ['I', 'you', 'he', 'she', 'we', 'they']:
                query['bool']['should'].append({
                    "match_phrase": {
                        "text": word.replace(key, po)
                    }
                })
    elif word in ["an offer one can't refuse",
                  "as best one can",
                  "as far as one knows",
                  "bite off more than one can chew",
                  "know where one stands",
                  "last thing one needs",
                  "let one go"]:
        query = {
            "bool": {
                "should": [
                ]
            }
        }

        doc = nlp(word)
        idx = [t.i for t in doc if t.text in ['one']][0]
        if doc[idx - 1].tag_.startswith('V'):
            for po in ['me', 'you', 'him', 'her', 'us', 'them']:
                query['bool']['should'].append({
                    "match_phrase": {
                        "text": word.replace('one', po)
                    }
                })
        else:
            for po in ['I', 'you', 'he', 'she', 'we', 'they']:
                query['bool']['should'].append({
                    "match_phrase": {
                        "text": word.replace('one', po)
                    }
                })
    else:
        new_word = ' '.join(toker.tokenize(word))
        if new_word != word:
            query = {
                "bool": {
                    "should": [{
                        "match_phrase": {
                            "text": word
                        }
                    },
                        {
                            "match_phrase": {
                                "text": new_word
                            }
                        }
                    ]
                }
            }
        else:
            query = {
                "match_phrase": {
                    "text": word
                }
            }

    body['query'] = query

    records = []
    for i, hit in enumerate(tqdm(scan(client, query=body, index=index,
                                      scroll='10m', size=10000,
                                      request_timeout=600), leave=False, desc='hitting')):
        t = hit['highlight']
        tmp_context = t['text'][0]
        left = tmp_context.index('<em>')
        right = tmp_context.rindex('</em>')

        ground_truth = tmp_context[left:right].replace('<em>', '').replace('</em>', '')
        if ',' not in word and ',' in ground_truth:
            continue

        if index == 'bnc':
            tmp_context_d = get_contexts(hit['_id'])
            tmp_context_d[hit['_id']] = tmp_context
        else:
            tmp_context_d = {
                hit['_id']: tmp_context
            }

        d = {
            '_id': hit['_id'],
            'idiom': word,
            "groundTruth": [ground_truth],
            "content": tmp_context_d,
        }
        records.append(d)

    with jsonlines.open(dump_file, 'w') as f:
        [f.write(d) for d in tqdm(records, leave=False, desc='dumping')]


def build_values(book_name, line_num):
    if line_num == 0:
        values = [
            # f"{book_name}_{line_num}",
            f"{book_name}_{line_num + 1}",
            f"{book_name}_{line_num + 2}",
        ]
    else:
        values = [
            f"{book_name}_{line_num - 1}",
            # f"{book_name}_{line_num}",
            f"{book_name}_{line_num + 1}",
        ]

    return values


def get_contexts(hit_id):
    book_name, line_num = hit_id.rsplit('_', 1)
    line_num = int(line_num)

    values = build_values(book_name, line_num)

    body = {
        "_source": ["text"],
        "query": {
            "ids": {
                "values": list(values)
            }
        }
    }

    tmp_context_d = {}
    for hit in tqdm(scan(client, query=body, index=index,
                         scroll='10m', size=10000,
                         request_timeout=600),
                    leave=False, desc='windowing', total=len(values)):
        idx = hit['_id']
        t = hit['_source']
        tmp_context_d[idx] = t['text']
    return tmp_context_d


if __name__ == '__main__':
    import sys
    from more_itertools import chunked

    index = sys.argv[1]

    df_sentiment = pda.read_csv("idiomLexicon.tsv", sep="\t")

    idiom_definitions = {}
    for _, idiom, explanation in chunked(open('../Paraphrases_Idiom/idioms_dataset_2432').read().split('\n'), 3):
        idiom_definitions[idiom] = explanation

    idioms = set(df_sentiment.Idiom.tolist()).union(set(idiom_definitions.keys()))

    pbar = tqdm(list(idioms)[::-1])
    for word in pbar:
        if '/' in word:
            print(word)
            continue

        if "'" in word or "one" in word:
            if os.path.isfile(f'{index}_dumped/{word}.jsonl'):
                Path(f'{index}_dumped/{word}.jsonl').unlink()

        pbar.set_description(word)
        dump_file = f'{index}_dumped/{word}.jsonl'
        dump_file_tmp = f'{dump_file}.tmp'
        if not os.path.isfile(dump_file) and not os.path.isfile(dump_file_tmp):
            Path(dump_file_tmp).touch()
            dump_idiom(word, dump_file, index)
            Path(dump_file_tmp).unlink()
