import csv
import random
import argparse
import spacy
import pickle
import math
from logging_ import init_logger
import utils
import os
import re
import json

random.seed(42)

logger = init_logger()

nlp = spacy.load("en_core_web_md")
# texts = ["This is a text", "These are lots of texts", "..."]
# docs = list(nlp.pipe(texts, disable=["tagger", "parser"]))



def text_equal(str1, str2):
    # str1 = [_.strip() for _ in str1.split()]
    # str2 = [_.strip() for _ in str2.split()]
    # for _ in str1:
    #     if (_ != '') and (_ in str2):
    #         return True
    # return False
    if str1.strip() == str2.strip():
        return True
    return False

def read_tsv_data(data_path):
    '''
    read tsv data
    output a a-list, which contains b-lists (each b-list is a row), each b-list contains a list of texts 
    '''
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        results = []
        for _ in reader:
            results.append(_)
    return results

def matching_highlights(articles, summaries, highlight_positions):
    '''
    transform raw tsv file contents to a list of dict
    '''
    results = []
    for i, _ in enumerate(highlight_positions):
        tmp_highlights, tmp_positions = [[] for ii in range(len(_))], []
        for j, __ in enumerate(_):
            tmptmp_positions = [int(___) for ___ in __.split(',') if ___ != '']
            tmp_positions.append(tmptmp_positions)
            for k in tmptmp_positions:
                tmp_highlights[j].append(articles[i][k])
        try:
            assert len(summaries[i]) == len(tmp_highlights)
        except AssertionError:
            continue
        results.append({
            'summary':summaries[i], 
            'highlight':tmp_highlights, 
            'article':articles[i], 
            'highlight_position':tmp_positions
            })
    return results

def text_processing(data, mode = ['ner'], types = ['summary', 'highlight', 'article']):
    '''
    using spacy package to do ner/tagging/parsing on texts data

    Args:
        data: output of function ``matching_highlights''
        mode: a list, which could contain "tagger", "parser", "ner"
        types: a list, which could contain 'summary', 'highlight', 'article'
    '''
    disable_mode = [_ for _ in ["tagger", "parser", "ner"] if _ not in mode]
    results = []

    ## merge texts into a single list ["...", '...', ...]
    ## merging is for quick processing of spacy
    all_summaries, all_highlights, summary_lengths = [], [], []
    if 'article' in types:
        all_articles, article_lengths = [], [] 
    for _ in data:
        summary_lengths.append(len(_['summary']))
        all_summaries += _['summary']
        if 'article' in types:
            article_lengths.append(len(_['article']))
            all_articles += _['article']
        for __ in _['highlight']:
            all_highlights += __

    ## process the texts using spacy
    all_summaries_ner = list(nlp.pipe(all_summaries, disable=disable_mode))
    all_highlights_ner = list(nlp.pipe(all_highlights, disable=disable_mode))
    if 'article' in types:
        all_articles_ner = list(nlp.pipe(all_articles, disable=disable_mode))
    
    ## break lists of texts into aligned format
    summary_index, highlight_index, article_index = 0, 0, 0
    for i, _ in enumerate(data):
        assert len(_['highlight_position']) == summary_lengths[i]
        tmp_highlight = []
        for __ in _['highlight_position']:
            tmp_highlight.append(all_highlights_ner[highlight_index : highlight_index + len(__)])
            highlight_index += len(__)
        if 'article' in types:
            results.append({
                'summary': all_summaries_ner[summary_index : summary_index + summary_lengths[i]],
                'highlight': tmp_highlight,
                'article': all_articles_ner[article_index : article_index + article_lengths[i]],
                'highlight_position': _['highlight_position']
            })
            summary_index += summary_lengths[i]
            article_index += article_lengths[i]
        else:
            results.append({
                'summary': all_summaries_ner[summary_index : summary_index + summary_lengths[i]],
                'highlight': tmp_highlight,
                'highlight_position': _['highlight_position']
            })
            summary_index += summary_lengths[i]
    return results
        
def writing_ner_data(data, writing_path, mode = ['ner'], types = ['summary', 'highlight', 'article']):
    tagged_data = text_processing(data, mode=mode, types=types)
    logger.info('Writing data to file {}'.format(writing_path))
    with open(writing_path, 'wb') as f:
        tagged_data_pkl = pickle.dumps(tagged_data)
        f.write(tagged_data_pkl)
    logger.info('Successfully saved data to file {}'.format(writing_path))
    
def load_ner_data(file_path):
    logger.info('Loading data {}'.format(file_path))
    with open(file_path, 'rb') as f:
        tagged_data = pickle.loads(f.read())
    logger.info('Successfully loaded data {}, which contains {} samples'.format(file_path, len(tagged_data)))
    logger.info(tagged_data[0]['summary'])
    logger.info(tagged_data[0]['highlight'])
    return tagged_data

def get_entity(doc, offset = 0, number_type = ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']):
    '''
    Args:
        doc: Doc instance, from spacy package
        number_type: the list of Named Entity types that would be included in the <number> group
    Output:
        a dict:
            'text': str format of the entity, 
            'type': entity type/pronoun type
            'position': [start, end], span of the entity/pronoun
            'group': 'name' or 'number' or 'pronoun'
    '''
    entity_cache = []
    results = {
        'name': [],
        'number': [],
        'pronoun': []
    }
    for i in range(len(doc)):
        if doc[i].ent_iob_ is 'O' and doc[i].text in utils.all_third_pronoun:
            tmp_types = []
            for pp, which_pronoun in enumerate(utils.pronoun_lists):
                if doc[i].text in which_pronoun:
                    tmp_types.append(pp)
            if len(tmp_types) > 1:
                tmp_type = tmp_types[math.floor(random.random()*len(tmp_types))]
            else:
                tmp_type = tmp_types[0]
            results['pronoun'].append({
                'text': doc[i].text,
                'type': tmp_type,
                'position': [i, i + 1], 
                'offset_position': [offset + i, offset + i + 1], 
                'group': 'pronoun'
            })
        if doc[i].ent_iob_ == 'B':
            if len(entity_cache) != 0:
                entity_cache[0]['position'][1] = i
                entity_cache[0]['offset_position'][1] = offset + i
                results[entity_cache[0]['group']].append(entity_cache[0])
                entity_cache.clear()
            entity_cache.append({
                'text': doc[i].text, 
                'type': doc[i].ent_type_, 
                'position': [i, i], 
                'offset_position': [offset + i, offset + i]
            })
            if doc[i].ent_type_ in number_type:
                entity_cache[0]['group'] = 'number'
            else:
                entity_cache[0]['group'] = 'name'
        elif doc[i].ent_iob_ == 'I':
            entity_cache[0]['text'] += ' '+doc[i].text
        elif doc[i].ent_iob_ == 'O':
            if len(entity_cache) != 0:
                entity_cache[0]['position'][1] = i
                entity_cache[0]['offset_position'][1] = offset + i
                results[entity_cache[0]['group']].append(entity_cache[0])
                entity_cache.clear()
    return results


def del_and_insert(text, insert_text, position_span):
    '''
    to delete the position_span in text, and the insert the insert_text in the original position

    Args:
        text: a list of tokens
        insert_text: a str, which may contain several tokens separated by space
        position_span: a list of two elements, [start, end]
    Output: 
        the processed text: a list of tokens
    '''
    del text[position_span[0]: position_span[1]]
    for i, _ in enumerate(insert_text.split()):
        text.insert(position_span[0] + i, _)
    return text

def adjust_position(original_position, x):
    '''

    Args:
        original_position: [a, b]
        x: a list, two options: 
                    1. for swapping name entities inside the sentence only (e.g. [[5, 9], [1, 3]]). 
                        Note that x[1] should be larger than x[0]; 
                    2. for swapping number entities within the whole vocabulary (e.g. [[1, 3], 4])
    '''
    if isinstance(x[1], list):
        if original_position[1] <= x[1][0] or original_position[0] >= x[0][1]:
            return original_position
        elif original_position[0] >= x[1][1] and original_position[1] <= x[0][0]:
            change = (x[0][1] - x[0][0]) - (x[1][1] - x[1][0])
            return [(_ + change) for _ in original_position]
        else:
            raise ValueError('the input position for adjusting is incorrect. The span is {} while the swapping spans are {}.'.format(original_position, x))
    elif isinstance(x[1], int):
        if original_position[1] <= x[0][0]: 
            return original_position
        elif original_position[0] >= x[0][1]:
            return [(_ + (x[1] - (x[0][1] - x[0][0]))) for _ in original_position]
        else:
            raise ValueError('the input position for adjusting is incorrect. The span is {} while the swapping spans are {}.'.format(original_position, x))
    else:
        raise ValueError('the input x[1] = {} is neither a list nor an int'.format(x[1]))
    
def classifying_overlap(entity_text, hypothesis_instances, entity_type, allowed_gender = None, hypothesis_text = None, output_overlapping = False): 
    '''
    to maintain a threshold for entity-swapping, because an entity in the premise may not be included in the hypothesis

    Args:
        entity_text: a str
        hypothesis_instances: a list of output of get_entity['name']
        entity_type: name or number or pronoun
    Output: 
        True or False
    '''
    signal = False
    overlapping_tokens = []
    non_overlapping_entity = []
    if entity_type == 'name' or 'number': 
        for _ in entity_text.split():
            for __ in hypothesis_instances:
                if _ in __['text']: 
                    signal = True
                    overlapping_tokens.append(_)
    elif entity_type == 'pronoun': 
        if (entity_text in utils.male_pronoun and allowed_gender[1] == 1) \
            or (entity_text in utils.female_pronoun and allowed_gender[0] == 1):
            signal = True
            overlapping_tokens.append(entity_text)
    else:
        raise ValueError('The input entity_type ({}) for classifying_overlap functions cannot be accepted.'.format(entity_type))
    if entity_type == 'name':
        for _ in hypothesis_instances:
            tmp_signal = False
            for __ in entity_text.split():
                if __ in _['text']:
                    tmp_signal = True
            if not tmp_signal:
                non_overlapping_entity.append(_)
    if output_overlapping:
        return signal, overlapping_tokens, non_overlapping_entity
    return signal

def swap_name_entity(sentence_text, sent_instances, ops = None, prob = 0.5, hypothesis_instances = None):
    '''
    to swap two name entities inside a sentence
    ## new change on 2020/May/01:
            not swap any more. 
            Current: to change a entity in gold summary with a random entity from highlight text

    Args:
        sentence_text: a list of tokens
        sent_instances: a list, output of get_entity['name']
        ops: a list of previous operations to adjust positions
        prob: probability to do operation, ranging from 0.0 to 1.0
    Output: 
        sentence_text: a list of tokens
        signal: bool, whether or not that operation is done
        operations: produce operations to adjust positions
    '''

    # if 'Minatel' in sentence_text: 
    #     logger.info(sentence_text)
    #     logger.info(sent_instances)
    
    if ops:
        for _ in ops:
            for __ in sent_instances:
                __['offset_position'] = adjust_position(__['offset_position'], _)

    signal = False
    operations = []
    i = 0
    while i < math.floor(len(sent_instances)**0.5):
        if random.random() <= prob:
            try:
                swap_pair_indexes = random.sample(list(range(len(sent_instances))), 2)
            except ValueError:
                return sentence_text, signal, operations

            swap_pair = [sent_instances[swap_pair_indexes[0]], sent_instances[swap_pair_indexes[1]]]

            if (swap_pair[0]['text'] == swap_pair[1]['text']) or \
                (
                    (max(swap_pair[0]['offset_position'][0], swap_pair[1]['offset_position'][0]) - min(swap_pair[0]['offset_position'][1], swap_pair[1]['offset_position'][1]) == 1) and \
                    (sentence_text[min(swap_pair[0]['offset_position'][1], swap_pair[1]['offset_position'][1])].strip() in {',', 'and', 'or'})
                ): 
                del sent_instances[swap_pair_indexes[0]]
                continue

            if swap_pair[0]['offset_position'][0] < swap_pair[1]['offset_position'][0]:
                swap_pair[0], swap_pair[1] = swap_pair[1], swap_pair[0]

            if not (classifying_overlap(sent_instances[swap_pair_indexes[0]]['text'], hypothesis_instances, 'name') \
                    or classifying_overlap(sent_instances[swap_pair_indexes[1]]['text'], hypothesis_instances, 'name')): 
                del sent_instances[swap_pair_indexes[0]]
                tmp_signal = False
                for i, _ in enumerate(sent_instances):
                    if classifying_overlap(_['text'], hypothesis_instances, 'name'): 
                        tmp_signal = True
                if tmp_signal:
                    continue
                else:
                    return sentence_text, signal, operations

            signal = True
            i += 1
            # swap_pair = [sent_instances[swap_pair_indexes[0]], sent_instances[swap_pair_indexes[1]]]
            # if swap_pair[0]['offset_position'][0] < swap_pair[1]['offset_position'][0]:
            #     swap_pair[0], swap_pair[1] = swap_pair[1], swap_pair[0]
            if random.random() <= prob:
                sentence_text = del_and_insert(sentence_text, swap_pair[1]['text'], swap_pair[0]['offset_position'])
                sentence_text = del_and_insert(sentence_text, swap_pair[0]['text'], swap_pair[1]['offset_position'])
                operations.append([swap_pair[0]['offset_position'], swap_pair[1]['offset_position']])
            elif random.random() <= prob:
                sentence_text = del_and_insert(sentence_text, swap_pair[1]['text'], swap_pair[0]['offset_position'])
                operations.append([swap_pair[0]['offset_position'], (swap_pair[1]['offset_position'][1]-swap_pair[1]['offset_position'][1][0])])
            else:
                sentence_text = del_and_insert(sentence_text, swap_pair[0]['text'], swap_pair[1]['offset_position'])
                operations.append([swap_pair[1]['offset_position'], (swap_pair[0]['offset_position'][1]-swap_pair[0]['offset_position'][1][0])])

            del sent_instances[swap_pair_indexes[0]]
            if swap_pair_indexes[0] < swap_pair_indexes[1]: 
                del sent_instances[swap_pair_indexes[1] - 1]
            else:
                del sent_instances[swap_pair_indexes[1]]

            for j in range(len(sent_instances)):
                try:
                    sent_instances[j]['offset_position'] = adjust_position(sent_instances[j]['offset_position'], operations[-1])
                except ValueError:
                    logger.info(sentence_text)
                    logger.info(sent_instances)
                    logger.info(sent_instances[j])
                    logger.info(swap_pair)
                    raise KeyboardInterrupt
                    # if sent_instances[j]['text'] not in ' '.join(sentence_text):
                    #     continue
                    # else:
                    #     raise ValueError('This exception is in the sentence_text. ')

    return sentence_text, signal, operations

def swap_number_entity(sentence_text, sent_instances, number_vocab = None, ops = None, must_swap = False, prob = 0.65, hypothesis_instances = None):
    '''
    to change a number entity into a different one with the same entity_type

    Args:
        sentence_text: a list of tokens
        sent_instances: a list, output of get_entity['number']
        number_vocab: a dict. The whole dict is categoried into several lists, by ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
        ops: a list of previous operations to adjust positions
        must_swap: bool, whether or not to ignore prob
        prob: probability to do operation, ranging from 0.0 to 1.0
    Output: 
        sentence_text: a list of tokens
        signal: bool, whether or not that operation is done
        operations: a list of operations produced from this function, to adjust positions
    '''

    for i, _ in enumerate(sent_instances):
        if not classifying_overlap(_['text'], hypothesis_instances, 'number'): 
            del sent_instances[i]

    if ops:
        for _ in ops:
            for __ in sent_instances:
                __['offset_position'] = adjust_position(__['offset_position'], _)
    
    operations = []
    signal = False
    type_instances = dict()
    for key in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
        type_instances[key] = []
    for _ in sent_instances:
        type_instances[_['type']].append(_)
    for i, _ in enumerate(sent_instances):
        if i == 0 and must_swap:
            temp_insert = _
            while temp_insert['text'] == _['text']:
                temp_insert = random.sample(number_vocab[_['type']], 1)[0]
            signal = True
            sentence_text = del_and_insert(sentence_text, temp_insert['text'], _['offset_position'])
            operations.append([_['offset_position'], temp_insert['offset_position'][1] - temp_insert['offset_position'][0]])
            for j in range(i + 1, len(sent_instances)):
                sent_instances[j]['offset_position'] = adjust_position(sent_instances[j]['offset_position'], operations[-1])
        else:
            if random.random() <= prob:
                temp_insert = _
                while temp_insert['text'] == _['text']:
                    temp_insert = random.sample(number_vocab[_['type']], 1)[0]
                signal = True
                sentence_text = del_and_insert(sentence_text, temp_insert['text'], _['offset_position'])
                operations.append([_['offset_position'], temp_insert['offset_position'][1] - temp_insert['offset_position'][0]])
                for j in range(i + 1, len(sent_instances)):
                    sent_instances[j]['offset_position'] = adjust_position(sent_instances[j]['offset_position'], operations[-1])
    return sentence_text, signal, operations
    
def swap_pronoun(sentence_text, sent_instances, ops = None, must_swap = False, prob = 0.65, hypothesis_instances = None):
    '''
    to change a pronoun (with gender) into a different gender within the same type

    Args:
        sentence_text: a list of tokens
        sent_instances: a list, output of get_entity['number']
        ops: a list of previous operations to adjust positions
        must_swap: bool, whether or not to ignore prob
        prob: probability to do operation, ranging from 0.0 to 1.0
    Output: 
        sentence_text: a list of tokens
        signal: bool, whether or not that operation is done
        operations: a list of operations produced from this function, to adjust positions
    '''

    if len(hypothesis_instances) == 0:
        return sentence_text, False, []

    allowed_gender = [0, 0] ## [female, male]
    for _ in hypothesis_instances: 
        if _['text'] in utils.male_pronoun:
            allowed_gender[1] =  1
        elif _['text'] in utils.female_pronoun:
            allowed_gender[0] = 1
    if not (allowed_gender[0] == 1 and allowed_gender[1] == 1):
        for i, _ in enumerate(sent_instances):
            if not classifying_overlap(_['text'], hypothesis_instances, 'pronoun', allowed_gender = allowed_gender): 
                del sent_instances[i]

    if ops:
        for _ in ops:
            for __ in sent_instances:
                __['offset_position'] = adjust_position(__['offset_position'], _)
    operations = []
    signal = False
    for i, _ in enumerate(sent_instances):
        if i == 0 and must_swap: 
            for __ in utils.pronoun_lists[_['type']]:
                if __ != _['text']:
                    temp_insert_text = __
                    break
            signal = True
            sentence_text = del_and_insert(sentence_text, temp_insert_text, _['offset_position'])
        else:
            if random.random() <= prob: 
                for __ in utils.pronoun_lists[_['type']]:
                    if __ != _['text']:
                        temp_insert_text = __
                        break
                signal = True
                sentence_text = del_and_insert(sentence_text, temp_insert_text, _['offset_position'])
    return sentence_text, signal, operations
    

def pseudo_data_algorithm(premise, hypothesis = None, number_vocab = None, use_article = False, article = None, swapping_policy = ['name', 'number', 'pronoun']):
    '''

    Args:
        premise: a highlight, format: a list of Doc (from spacy)
        ## hypothesis: a summary sentence (format: Doc), normally gold
        hypothesis: a list of summary sentence (format: a list of spacy.tokens.doc.Doc), normally gold
        use_article: bool, whether to use article under  the condition that only highlight is not enough
        article: a whole article, format: a list of Doc (from spacy)
        swapping_policy: a list, may include 'name', 'number', 'pronoun', 'predicate', 'noise'
    Output: 
        {
        'text': sentence_text, (a list of tokens)
        'ref': hypothesis text, (a list of tokens)
        'signal': [name_signal, number_signal, pronoun_signal], 
        'label': label (1 or 0)
        }
    '''
    entities = {
        'name': [],
        'number': [],
        'pronoun': []
    }
    sentence_text = []
    sentence_text_original = []
    if isinstance(premise, list):
        offset = 0
        for _ in premise:
            # sentence_text += ' '.join(re.split(r"([-\s])", _.text)).split()
            sentence_text += [token.text for token in _]
            sentence_text_original += _.text.split()
            temp_entities = get_entity(_, offset=offset)
            for __ in temp_entities.keys():
                entities[__] += temp_entities[__]
            offset = len(sentence_text)
    elif isinstance(premise, spacy.tokens.doc.Doc):
        # sentence_text = ' '.join(re.split(r"([-\s])", premise.text)).split()
        sentence_text = [token.text for token in premise]
        sentence_text_original = premise.text.split()
        temp_entities = get_entity(premise)
        for _ in temp_entities.keys():
            entities[_] += temp_entities[_]
    else:
        raise TypeError('The input type = {} is wrong. It should be either List or spacy.tokens.doc.Doc.'.format(type(premise))) 

    ## processing hypothesis instance
    hyp_entities = {
        'name': [],
        'number': [],
        'pronoun': []
    }
    # # difference between hyp_text and ref_text:
    #     hyp_text: builded from spacy tokenization
    #     ref_text: builded from simple split() command
    hyp_text = []
    ref_text = []
    if isinstance(hypothesis, list): 
        offset = 0
        for _ in hypothesis: 
            hyp_text += [token.text for token in _]
            ref_text += _.text.split()
            temp_entities = get_entity(_, offset=offset)
            for __ in temp_entities.keys():
                hyp_entities[__] += temp_entities[__]
            offset = len(hyp_text)
    else:
        hyp_text = [token.text for token in hypothesis]
        ref_text = hypothesis.text.split()
        temp_entities_ = get_entity(hypothesis)
        for _ in temp_entities_.keys():
                hyp_entities[_] += temp_entities_[_]

    operations = []
    signal = False

    sentence_text, name_signal, name_ops = swap_name_entity(sentence_text, 
                                                            entities['name'], 
                                                            ops=operations, 
                                                            hypothesis_instances=hyp_entities['name'])
    operations += name_ops
    signal = signal or name_signal
    sentence_text, number_signal, number_ops = swap_number_entity(sentence_text, 
                                                                    entities['number'], 
                                                                    number_vocab=number_vocab, 
                                                                    ops=operations, 
                                                                    must_swap=not signal, 
                                                                    hypothesis_instances=hyp_entities['number'])
    signal = signal or number_signal
    operations += number_ops
    sentence_text, pronoun_signal, pronoun_ops = swap_pronoun(sentence_text, 
                                                                entities['pronoun'], 
                                                                ops=operations, 
                                                                must_swap=not signal, 
                                                                hypothesis_instances=hyp_entities['pronoun'])
    signal = signal or pronoun_signal
    label = 0
    if signal: 
        label = 1
    
    
    return {
        'text': sentence_text, 
        'original': sentence_text_original, 
        'ref': ref_text, 
        'signal': [name_signal, number_signal, pronoun_signal], 
        'label': label
    }

def pseudo_data_algorithm_only_mask(highlight_doc, summary_doc, prob=0.5): 
    result = {
        'label': 0, # whether or not [MASK] token is inserted in the summary sentence
        'text':[], # a list of strs
        'masked_entity': [], 
        'count': 0 # how many times to make changes
    }
    label = 0

    # for processing highlight_doc
    highlight_entities = {
        'name': [],
        'number': [],
        'pronoun': []
    }
    highlight_text = []
    offset = 0
    for _ in highlight_doc:
        # highlight_text += ' '.join(re.split(r"([-\s])", _.text)).split()
        highlight_text += [token.text for token in _]
        temp_entities = get_entity(_, offset=offset)
        for __ in temp_entities.keys():
            highlight_entities[__] += temp_entities[__]
        offset = len(highlight_text)

    # processing summary_doc
    summary_entities = {
        'name': [],
        'number': [],
        'pronoun': []
    }
    summary_text = [token.text for token in summary_doc]
    temp_entities = get_entity(summary_doc)
    for _ in temp_entities.keys():
        summary_entities[_] += temp_entities[_]

    allowed_gender = [0, 0] ## [female, male]
    for _ in highlight_entities['pronoun']: 
        if _['text'] in utils.male_pronoun:
            allowed_gender[1] =  1
        elif _['text'] in utils.female_pronoun:
            allowed_gender[0] = 1

    # entity_types = ['name', 'number', 'pronoun']
    entity_types = ['name', 'number']

    ops = []

    for entity_type in entity_types: 
        for i, _ in enumerate(summary_entities[entity_type]):
            try:
                assert len(_['text'].split()) == _['offset_position'][1] - _['offset_position'][0]
            except AssertionError:
                print(_)
                print(ops)
            overlapping_signal, overlapping_tokens = classifying_overlap(_['text'], highlight_entities[entity_type], entity_type, allowed_gender=allowed_gender, output_overlapping=True)
            if overlapping_signal: 
                if random.random() <= prob:
                    # summary_text[_['offset_position'][0]:_['offset_position'][1]] = ['[MASK]' for ii in range(_['offset_position'][1]-_['offset_position'][0])]
                    del summary_text[_['offset_position'][0]:_['offset_position'][1]]
                    for j in range(_['offset_position'][0], _['offset_position'][0] + len(overlapping_tokens)):
                        summary_text.insert(j, '[MASK]')
                    result['masked_entity'].append({
                                                    'text': overlapping_tokens, 
                                                    'position': [_['offset_position'][0], _['offset_position'][0] + len(overlapping_tokens)]
                                                    })
                    if len(overlapping_tokens) != _['offset_position'][1] - _['offset_position'][0]: 
                        ops.append([[_['offset_position'][0], _['offset_position'][1]], len(overlapping_tokens)])
                        for _entity_type in entity_types: 
                            for __ in summary_entities[_entity_type]: 
                                if __['offset_position'][0] != ops[-1][0][0]:
                                    try:
                                        __['offset_position'] = adjust_position(__['offset_position'], ops[-1])
                                    except ValueError:
                                        print(__)
                                        print(ops)
                                        print(summary_text)
                                        print(summary_entities)
                                        print(result['masked_entity'])
                                        raise KeyboardInterrupt
                        for __ in result['masked_entity']: 
                            if __['position'][0] != ops[-1][0][0]:
                                __['position'] = adjust_position(__['position'], ops[-1])

                        _['offset_position'][1] = _['offset_position'][1] - ((_['offset_position'][1] - _['offset_position'][0]) - len(overlapping_tokens))

                    label = 1
                    result['count'] += 1
    
    # ops.sort(key = lambda x : x[0][0], reverse=True)
    # if ops:
    #     for op in ops:
    #         for __ in result['masked_entity']:
    #             if __['position'][0] >= op[0][1] and (op[0][1] - op[0][0]) > op[1]:
    #             if __['text'] == ['Palestinian'] and __['position'][0] < 3:
    #                     print('#'*50)
    #                     print(summary_text)
    #                     print(summary_entities)
    #                     print(__)
    
    # for __ in result['masked_entity']:
    #     if __['text'] == ['Palestinian'] and __['position'][0] < 3:
    #         print('#'*50)
    #         print(summary_text)
    #         print(summary_entities)
    #         print(__)

    result['text'] = ['[CLS]'] + highlight_text + ['[SEP]'] + summary_text
    result['label'] = label
    return result

def pseudo_data_algorithm_new(highlight_doc, summary_doc, name_vocab=None, number_vocab=None, prob={'name':0.5, 'number':0.5, 'pronoun':0.5}):
    
    tmp_data_statistic = {
        'name': 0,
        'number': 0,
        'pronoun': 0
    }

    result = {
        'text': [],
        'gold_entity':[],
    }

    highlight_all_entities, summary_all_entities = [], []

    # preprocessing highlight_doc
    highlight_entities = {
        'name': [],
        'number': [],
        'pronoun': []
    }
    highlight_text = []
    offset = 0
    for _ in highlight_doc:
        # highlight_text += ' '.join(re.split(r"([-\s])", _.text)).split()
        highlight_text += [token.text for token in _]
        tmp_entities = get_entity(_, offset=offset)
        for __ in tmp_entities.keys():
            highlight_entities[__] += tmp_entities[__]
            for entity in tmp_entities[__]:
                highlight_all_entities.append(entity)
        offset = len(highlight_text)

    # preprocessing summary_doc
    summary_entities = {
        'name': [],
        'number': [],
        'pronoun': []
    }
    summary_text = [token.text for token in summary_doc]
    tmp_entities = get_entity(summary_doc)
    for _ in tmp_entities.keys():
        summary_entities[_] += tmp_entities[_]
        for entity in tmp_entities[_]:
            summary_all_entities.append(entity)
    # sort entities in reverse order
    summary_all_entities.sort(key = lambda x : x['offset_position'][0], reverse=True)

    allowed_gender = [0, 0] ## [female, male]
    for _ in highlight_entities['pronoun']: 
        if _['text'] in utils.male_pronoun:
            allowed_gender[1] =  1
        elif _['text'] in utils.female_pronoun:
            allowed_gender[0] = 1
    
    # entity_types = ['name', 'number', 'pronoun']
    entity_types = ['name', 'number', 'pronoun']

    for _ in summary_all_entities:
        if _['group'] not in entity_types:
            continue
        overlapping_signal, overlapping_tokens, non_overlapping_entity = \
            classifying_overlap(
                _['text'], 
                highlight_entities[_['group']], 
                _['group'], 
                allowed_gender=allowed_gender, 
                output_overlapping=True
                )
        if overlapping_signal and random.random() <= prob[_['group']]:
            # name
            if _['group'] == 'name':
                # 50% chance to do swapping with entityies [inside] the sentence
                if random.random() <= 0.5:
                    if len(non_overlapping_entity) == 0:
                        continue
                    replace_entity = random.sample(non_overlapping_entity, 1)[0]
                    replace_entity = replace_entity.get('text').split()
                # 50% chance to do swapping with entityies [outside] the sentence
                else:
                    temp_type = 'PERSON' if _['type'] == 'PERSON' else 'OTHER'
                    replace_entity = _
                    while replace_entity['text'] == _['text']:
                        replace_entity = random.sample(name_vocab[temp_type], 1)[0]
                    replace_entity = replace_entity.get('text').split()
            # number
            elif _['group'] == 'number':
                replace_entity = random.sample(number_vocab[_['type']], 1)[0]
                replace_entity = replace_entity.get('text').split()
            # pronoun
            elif _['group'] == 'pronoun':
                for __ in utils.pronoun_lists[_['type']]:
                    if __ != _['text']:
                        replace_entity = [__]
                        break
            del summary_text[_['position'][0]:_['position'][1]]
            tmp_data_statistic[_['group']] += 1
            summary_text.insert(_['position'][0], replace_entity)
            result['gold_entity'].insert(0, overlapping_tokens)
    result['text'] = [highlight_text, summary_text]
    return result, tmp_data_statistic

def main_func(tagged_data, use_single_summ_sentence = False, mode = 'classifier', only_mask=False):
    '''
    the main function, which iteratively call pseudo_data_algorithm function

    Args:
        tagged_data: the output of load_ner_data
        mode: ``classifier'' or ``lev_transformer'', to control whether the output is for a classifier or a lev-transformer. 
    
    Output:
        a list of dict:
            {
            'text': sentence_text, (a list of tokens)
            'original': sentence_text_original, 
            'ref': hypothesis text, (a list of tokens)
            'signal': [name_signal, number_signal, pronoun_signal], (True or False)
            'label': label (1 or 0)
            }
    '''
    #####################
    if mode == 'lev_transformer': 
        use_single_summ_sentence = True
        src_samples, tgt_samples = [], []

    # data_statistic = [0, 0, 0, 0, 0, 0]  # [zero signal, one signal, two signal, three signal, #positive, #negative]
    data_statistics = {
        'name': [0 for i in range(15)],
        'number': [0 for i in range(15)],
        'pronoun': [0 for i in range(15)]
    }
    

    name_vocab, number_vocab = dict(), dict()
    pseudo_highlights = []
    only_mask_data = []

    result_data, result_data_statistics = [], []

    ## build name vocab
    for _ in utils.name_type:
        name_vocab[_] = []
    for i, _ in enumerate(tagged_data):
        tmp_highlight_ent = []
        # tmp_summary_ent = [get_entity(zz) for zz in _['summary']]
        for j, __ in enumerate(_['highlight']):
            tmptmp_highlight_ent = []
            for sent in __:
                tmptmp_highlight_ent += get_entity(sent)['name']
            tmp_highlight_ent.append(tmptmp_highlight_ent)
        for j, __ in enumerate(tmp_highlight_ent):
            for ___ in __: 
                if ___['group'] == 'name':
                    if ___['type'] == 'PERSON':
                        name_vocab['PERSON'].append(___)
                    else:
                        name_vocab['OTHER'].append(___)

    ## build number vocab
    for _ in utils.number_type:
        number_vocab[_] = []
    for i, _ in enumerate(tagged_data):
        tmp_highlight_ent = []
        # tmp_summary_ent = [get_entity(zz) for zz in _['summary']]
        for j, __ in enumerate(_['highlight']):
            tmptmp_highlight_ent = []
            for sent in __:
                tmptmp_highlight_ent += get_entity(sent)['number']
            tmp_highlight_ent.append(tmptmp_highlight_ent)
        for j, __ in enumerate(tmp_highlight_ent):
            for ___ in __: 
                if ___['group'] == 'number':
                    number_vocab[___['type']].append(___)
    # for _ in number_vocab.keys():
    #     number_vocab[_] = list(set(number_vocab[_]))

    result_data_prototype, result_data_statistics_prototype = [], []
    # whether a sample has been changed
    swapping_signals = []

    for i, _ in enumerate(tagged_data): 
        for j, __ in enumerate(_['highlight']):
            if only_mask:
                temp_sample = pseudo_data_algorithm_only_mask(__, _['summary'][j])
                # to generate high-quality pseudo data
                if temp_sample['count'] >= 2: 
                    only_mask_data.append(temp_sample)
            
            else:
                temp_sample, temp_data_statistic = pseudo_data_algorithm_new(__, _['summary'][j], name_vocab=name_vocab, number_vocab=number_vocab)
                all_count = 0
                for value in temp_data_statistic.values():
                    all_count += value
                swapping_signals.append(
                                        (1 if all_count != 0 else 0)
                                    )
                result_data_prototype.append(temp_sample)
                result_data_statistics_prototype.append(temp_data_statistic)
    
    if not only_mask:
        probab =  swapping_signals.count(0)/swapping_signals.count(1)

        for i, _ in enumerate(result_data_prototype):
            # ---
            # # 50% is positive samples
            # if (swapping_signals[i] == 0 and random.random() <= probab) or (swapping_signals[i] == 1):
            # ---
            # -----
            # all negative samples
            if swapping_signals[i] == 1:
                result_data.append(result_data_prototype[i])
                result_data_statistics.append(result_data_statistics_prototype[i])
                for entity_type_key in result_data_statistics_prototype[i].keys():
                    data_statistics[entity_type_key][result_data_statistics_prototype[i][entity_type_key]] += 1



            # else: 
            #     if use_single_summ_sentence:
            #         # 1.
            #         # make change to gold summary
            #         # temp_sample = pseudo_data_algorithm(__, hypothesis=_['summary'][j], number_vocab=number_vocab)
            #         # 2.
            #         # make change to highlight sentence (extrtactive sentence)
            #         temp_sample = pseudo_data_algorithm(_['summary'][j], hypothesis=__, number_vocab=number_vocab)
            #     else:
            #         # 1. 
            #         # make change to gold summary
            #         # temp_sample = pseudo_data_algorithm(__, hypothesis=_['summary'], number_vocab=number_vocab)
            #         # 3.
            #         # make change to highlight sentence (extrtactive sentence)
            #         temp_sample = pseudo_data_algorithm(_['summary'][j], hypothesis=_['highlight'], number_vocab=number_vocab)

            #     temp_count = 0
            #     for signal in temp_sample['signal']:
            #         if signal:
            #             temp_count += 1
            #     data_statistic[temp_count] += 1

            #     if mode == 'classifier': 
            #         ## control portions of positive and negative samples
            #         if temp_count >= 2: 
            #             pseudo_highlights.append(temp_sample)
            #             data_statistic[-1] += 1  # negative++
            #             if random.random() <= 0.6:   
            #                 ## 60% of the two-signal samples are also saved as positive
            #                 _temp_sample = temp_sample
            #                 _temp_sample['label'] = 0
            #                 _temp_sample['text'] = temp_sample['original']
            #                 pseudo_highlights.append(_temp_sample)
            #                 data_statistic[-2] += 1 # positive++
            #         else:
            #             if random.random() <= 0.1: 
            #                 # 10% of the few-signal data are positive
            #                 _temp_sample = temp_sample
            #                 _temp_sample['label'] = 0
            #                 _temp_sample['text'] = temp_sample['original']
            #                 pseudo_highlights.append(_temp_sample)
            #                 # positive++
            #                 data_statistic[-2] += 1 
            #     elif mode == 'lev_transformer': 
            #         if temp_count >= 2: 
            #             # pseudo_highlights.append(temp_sample)
            #             src_samples.append('[CLS] ' + ' '.join(temp_sample['ref']) + ' [SEP] ' + ' '.join(temp_sample['text']))
            #             tgt_samples.append('[CLS] ' + ' '.join(temp_sample['ref']) + ' [SEP] ' + ' '.join(temp_sample['original']))
            #             data_statistic[-1] += 1
            #             # if random.random() <= 0.2:   
            #             #     ## 20% of the two-signal samples are also saved as positive
            #             #     _temp_sample = temp_sample
            #             #     _temp_sample['label'] = 0
            #             #     _temp_sample['text'] = temp_sample['original']
            #             #     pseudo_highlights.append(_temp_sample)
            #             #     data_statistic[-2] += 1 # positive++
            #         elif temp_count >= 1 and random.random() <= 0.15: 
            #                 # 15% of the one-signal data are positive
            #                 # pseudo_highlights.append(temp_sample)
            #                 src_samples.append('[CLS] ' + ' '.join(temp_sample['ref']) + ' [SEP] ' + ' '.join(temp_sample['text']))
            #                 tgt_samples.append('[CLS] ' + ' '.join(temp_sample['ref']) + ' [SEP] ' + ' '.join(temp_sample['original']))
            #                 data_statistic[-1] += 1
            #         if random.random() <= 0.1: 
            #             # 10% of all data are positive
            #             _temp_sample = temp_sample
            #             _temp_sample['label'] = 0
            #             _temp_sample['text'] = temp_sample['original']
            #             # pseudo_highlights.append(_temp_sample)
            #             src_samples.append('[CLS] ' + ' '.join(_temp_sample['ref']) + ' [SEP] ' + ' '.join(_temp_sample['text']))
            #             tgt_samples.append('[CLS] ' + ' '.join(_temp_sample['ref']) + ' [SEP] ' + ' '.join(_temp_sample['original']))
            #             # positive++
            #             data_statistic[-2] += 1 
    if only_mask: 
        return only_mask_data
    else:
        return result_data, data_statistics, result_data_statistics

    # if mode == 'classifier': 
    #     return pseudo_highlights, data_statistic
    # return src_samples, tgt_samples, data_statistic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-article_path', required=True, help='article tsv file path')
    # parser.add_argument('-summ_path', required=True, help='summaries tsv file path')
    # parser.add_argument('-highlight_path', required=True, help='highlight tsv file path')
    parser.add_argument('-file_path', default='/data/senyang/summ_consistence/spacy_ner_data/', help='folder path of the input files')
    parser.add_argument('-mode', required=True, help='train, val or test')
    parser.add_argument('-chunk_size', default=20000, type=int, help='output file max size')
    parser.add_argument('-lev_true', action='store_true', default=False, help='whether for lev_transformer data')
    parser.add_argument('-only_mask', action='store_true', default=False, help='whether only use [MASK] instead of swapping')
    args = parser.parse_args()

    chunk_size = args.chunk_size

    ###  codes for processing raw data and saving them into pkl files
    # articles, summaries, highlight_positions = read_tsv_data(args.file_path+'articles.tsv'), read_tsv_data(args.file_path+'summaries.tsv'), read_tsv_data(args.file_path+'highlight.tsv')
    # data = matching_highlights(articles, summaries, highlight_positions)
    # if len(data) <= chunk_size:
    #     writing_ner_data(data, args.file_path+'spacy_ner_data.pkl', types=['summary', 'highlight', 'article'])
    # else:
    #     chunk_num = math.ceil(len(data)/chunk_size)
    #     for i in range(chunk_num):
    #         tmp_data = data[i*chunk_size: (i+1)*chunk_size]
    #         writing_ner_data(tmp_data, args.file_path+'spacy_ner_data_{}.pkl'.format(i), types=['summary', 'highlight', 'article'])

    # [zero signal, one signal, two signal, three signal, #positive, #negative]
    pseudo_data_statistic = [0, 0, 0, 0, 0, 0]

    if args.only_mask: 
        header = ['text', 'entity', 'label', 'count']
        only_mask_data = []
        if args.mode == 'train': 
            for i in range(100):
                file_path = args.file_path + args.mode + '/spacy_ner_data_{}.pkl'.format(i)
                if not os.path.exists(file_path):
                    break
                tagged_data = load_ner_data(file_path)
                logger.info('Building pseudo data using {} to {} samples'.format(chunk_size*i, chunk_size*(i+1)))
                tmp_only_mask_data = main_func(tagged_data, mode='lev_transformer', only_mask=args.only_mask)
                only_mask_data += tmp_only_mask_data
                # for p, _ in enumerate( data_statistic):
                #     pseudo_data_statistic[p] += _
                logger.info('Builded the {}_th chunk. '.format(i))
            logger.info('Start writing csv file. ')
            with open(args.file_path + args.mode + '/only_mask.csv', 'w', newline='') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(header)
                for _ in only_mask_data: 
                    f_csv.writerow([str(_['text']), str(_['masked_entity']), str(_['label']), str(_['count'])])
                
        elif args.mode in {'val', 'test'}: 
            tagged_data = load_ner_data(args.file_path + args.mode +'/spacy_ner_data.pkl')
            logger.info('Building Peuso data.')
            only_mask_data = main_func(tagged_data, mode='lev_transformer', only_mask=args.only_mask)
            logger.info('Start writing csv file. ')
            with open(args.file_path + args.mode + '/only_mask.csv', 'w', newline='') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(header)
                for _ in only_mask_data: 
                    f_csv.writerow([str(_['text']), str(_['masked_entity']), str(_['label']), str(_['count'])])
    else:
        header = ['text', 'gold_entity', 'statistic']
        result_data, result_data_statistics = [], []
        data_count = {
        'name': [0 for i in range(15)],
        'number': [0 for i in range(15)],
        'pronoun': [0 for i in range(15)]
        }
        if args.mode == 'train': 
            for i in range(100):
                file_path = args.file_path + args.mode + '/spacy_ner_data_{}.pkl'.format(i)
                if not os.path.exists(file_path):
                    break
                tagged_data = load_ner_data(file_path)
                logger.info('Building pseudo data using {} to {} samples'.format(chunk_size*i, chunk_size*(i+1)))
                tmp_result_data, tmp_data_count, tmp_data_statistics = main_func(tagged_data, mode='lev_transformer', only_mask=args.only_mask)
                result_data += tmp_result_data
                result_data_statistics += tmp_data_statistics
                for entity_type_key in tmp_data_count.keys():
                    for count_index in range(len(tmp_data_count[entity_type_key])):
                        data_count[entity_type_key][count_index] += tmp_data_count[entity_type_key][count_index]
                # for p, _ in enumerate( data_statistic):
                #     pseudo_data_statistic[p] += _
                logger.info('Builded the {}_th chunk. '.format(i))
                
        elif args.mode in {'val', 'test'}: 
            tagged_data = load_ner_data(args.file_path + args.mode +'/spacy_ner_data.pkl')
            logger.info('Building Peuso data.')
            result_data, data_count, result_data_statistics = main_func(tagged_data, mode='lev_transformer', only_mask=args.only_mask)

        logger.info('Start writing csv file. ')
        logger.info('Totally {} samples. '.format(len(result_data)))
        with open(args.file_path + args.mode + '/pseudo_data.csv', 'w', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(header)
            for i, _ in enumerate(result_data): 
                f_csv.writerow([str(_['text']), str(_['gold_entity']), str(result_data_statistics[i])])
        with open(args.file_path + args.mode + '/pseudo_data_count.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(data_count))
    # else: 
    #     if not args.lev_true: 
    #         if args.mode == 'train': 
    #             for i in range(100):
    #                 file_path = args.file_path + args.mode + '/spacy_ner_data_{}.pkl'.format(i)
    #                 if not os.path.exists(file_path):
    #                     break
    #                 tagged_data = load_ner_data(file_path)
    #                 logger.info('Building pseudo data using {} to {} samples'.format(chunk_size*i, chunk_size*(i+1)))
    #                 pseudo_data, data_statistic = main_func(tagged_data)
    #                 for p, _ in enumerate( data_statistic):
    #                     pseudo_data_statistic[p] += _
    #                 with open(args.file_path + args.mode +'/pseudo_data/pseudo_data_{}.pkl'.format(i), 'wb') as f:
    #                     pickle.dump(pseudo_data, f, protocol=3)
    #                 logger.info('Builded {} samples. Successfully saved pseudo_data_{}.pkl'.format(len(pseudo_data), i))
    #         elif args.mode in {'val', 'test'}: 
    #             tagged_data = load_ner_data(args.file_path + args.mode +'/spacy_ner_data.pkl')
    #             logger.info('Building Peuso data.')
    #             pseudo_data, data_statistic = main_func(tagged_data)
    #             for p, _ in enumerate( data_statistic):
    #                 pseudo_data_statistic[p] += _
    #             with open(args.file_path + args.mode +'/pseudo_data.pkl', 'wb') as f:
    #                 pickle.dump(pseudo_data, f, protocol=3)
    #             logger.info('Builded {} samples. Successfully saved pseudo_data.pkl.'.format(len(pseudo_data)))
    #     else: 
    #         src, tgt = [], []
    #         if args.mode == 'train': 
    #             for i in range(100):
    #                 file_path = args.file_path + args.mode + '/spacy_ner_data_{}.pkl'.format(i)
    #                 if not os.path.exists(file_path):
    #                     break
    #                 tagged_data = load_ner_data(file_path)
    #                 logger.info('Building pseudo data using {} to {} samples'.format(chunk_size*i, chunk_size*(i+1)))
    #                 tmp_src, tmp_tgt, data_statistic = main_func(tagged_data, mode='lev_transformer')
    #                 src += tmp_src
    #                 tgt += tmp_tgt
    #                 for p, _ in enumerate( data_statistic):
    #                     pseudo_data_statistic[p] += _
    #                 logger.info('Builded the {}_th chunk. '.format(i))
    #             logger.info('Start writing source file. ')
    #             with open(args.file_path + args.mode +'.src-tgt.src', 'w', encoding='utf-8') as f:
    #                 for _ in src: 
    #                     f.write(_ + '\n')
    #             logger.info('Start writing target file. ')
    #             with open(args.file_path + args.mode +'.src-tgt.tgt', 'w', encoding='utf-8') as f:
    #                 for _ in tgt: 
    #                     f.write(_ + '\n')
                    
    #         elif args.mode in {'val', 'test'}: 
    #             tagged_data = load_ner_data(args.file_path + args.mode +'/spacy_ner_data.pkl')
    #             logger.info('Building Peuso data.')
    #             src, tgt, data_statistic = main_func(tagged_data, mode='lev_transformer')
    #             for p, _ in enumerate( data_statistic):
    #                 pseudo_data_statistic[p] += _
    #             logger.info('Start writing source file. ')
    #             with open(args.file_path + args.mode +'.src-tgt.src', 'w', encoding='utf-8') as f:
    #                 for _ in src: 
    #                     f.write(_ + '\n')
    #             logger.info('Start writing target file. ')
    #             with open(args.file_path + args.mode +'.src-tgt.tgt', 'w', encoding='utf-8') as f:
    #                 for _ in tgt: 
    #                     f.write(_ + '\n')
        
    #     logger.info('Whole PseudoDataset statistics: {}'.format(pseudo_data_statistic))

