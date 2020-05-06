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
    
def classifying_overlap(entity_text, hypothesis_instances, entity_type, allowed_gender = None, hypothesis_text = None): 
    '''
    to maintain a threshold for entity-swapping, because an entity in the premise may not be included in the hypothesis

    Args:
        entity_text: a str
        hypothesis_instances: a list of output of get_entity['name']
        entity_type: name or number or pronoun
    Output: 
        True or False
    '''
    if entity_type == 'name' or 'number': 
        for _ in entity_text.split():
            for __ in hypothesis_instances:
                if _ in __['text']: 
                    return True
    elif entity_type == 'pronoun': 
        if (entity_text in utils.male_pronoun and allowed_gender[1] == 1) \
            or (entity_text in utils.female_pronoun and allowed_gender[0] == 1):
            return True
    else:
        raise ValueError('The input entity_type ({}) for classifying_overlap functions cannot be accepted.'.format(entity_type))
    return False

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

    # for debugging
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
        
def main_func(tagged_data, use_single_summ_sentence = False, mode = 'classifier'):
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

    data_statistic = [0, 0, 0, 0, 0, 0]  # [zero signal, one signal, two signal, three signal, #positive, #negative]

    number_vocab = dict()
    pseudo_highlights = []

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
    for i, _ in enumerate(tagged_data): 
        for j, __ in enumerate(_['highlight']):
            if use_single_summ_sentence:
                # 1.
                # make change to gold summary
                # temp_sample = pseudo_data_algorithm(__, hypothesis=_['summary'][j], number_vocab=number_vocab)
                # 2.
                # make change to highlight sentence (extrtactive sentence)
                temp_sample = pseudo_data_algorithm(_['summary'][j], hypothesis=__, number_vocab=number_vocab)
            else:
                # 1. 
                # make change to gold summary
                # temp_sample = pseudo_data_algorithm(__, hypothesis=_['summary'], number_vocab=number_vocab)
                # 3.
                # make change to highlight sentence (extrtactive sentence)
                temp_sample = pseudo_data_algorithm(_['summary'][j], hypothesis=_['highlight'], number_vocab=number_vocab)

            temp_count = 0
            for signal in temp_sample['signal']:
                if signal:
                    temp_count += 1
            data_statistic[temp_count] += 1

            if mode == 'classifier': 
                ## control portions of positive and negative samples
                if temp_count >= 2: 
                    pseudo_highlights.append(temp_sample)
                    data_statistic[-1] += 1  # negative++
                    if random.random() <= 0.6:   
                        ## 60% of the two-signal samples are also saved as positive
                        _temp_sample = temp_sample
                        _temp_sample['label'] = 0
                        _temp_sample['text'] = temp_sample['original']
                        pseudo_highlights.append(_temp_sample)
                        data_statistic[-2] += 1 # positive++
                else:
                    if random.random() <= 0.1: 
                        # 10% of the few-signal data are positive
                        _temp_sample = temp_sample
                        _temp_sample['label'] = 0
                        _temp_sample['text'] = temp_sample['original']
                        pseudo_highlights.append(_temp_sample)
                        # positive++
                        data_statistic[-2] += 1 
            elif mode == 'lev_transformer': 
                if temp_count >= 2: 
                    # pseudo_highlights.append(temp_sample)
                    src_samples.append('[CLS] ' + ' '.join(temp_sample['ref']) + ' [SEP] ' + ' '.join(temp_sample['text']))
                    tgt_samples.append('[CLS] ' + ' '.join(temp_sample['ref']) + ' [SEP] ' + ' '.join(temp_sample['original']))
                    data_statistic[-1] += 1
                    # if random.random() <= 0.2:   
                    #     ## 20% of the two-signal samples are also saved as positive
                    #     _temp_sample = temp_sample
                    #     _temp_sample['label'] = 0
                    #     _temp_sample['text'] = temp_sample['original']
                    #     pseudo_highlights.append(_temp_sample)
                    #     data_statistic[-2] += 1 # positive++
                elif temp_count >= 1 and random.random() <= 0.15: 
                        # 15% of the one-signal data are positive
                        # pseudo_highlights.append(temp_sample)
                        src_samples.append('[CLS] ' + ' '.join(temp_sample['ref']) + ' [SEP] ' + ' '.join(temp_sample['text']))
                        tgt_samples.append('[CLS] ' + ' '.join(temp_sample['ref']) + ' [SEP] ' + ' '.join(temp_sample['original']))
                        data_statistic[-1] += 1
                if random.random() <= 0.1: 
                    # 10% of all data are positive
                    _temp_sample = temp_sample
                    _temp_sample['label'] = 0
                    _temp_sample['text'] = temp_sample['original']
                    # pseudo_highlights.append(_temp_sample)
                    src_samples.append('[CLS] ' + ' '.join(_temp_sample['ref']) + ' [SEP] ' + ' '.join(_temp_sample['text']))
                    tgt_samples.append('[CLS] ' + ' '.join(_temp_sample['ref']) + ' [SEP] ' + ' '.join(_temp_sample['original']))
                    # positive++
                    data_statistic[-2] += 1 
            
    if mode == 'classifier': 
        return pseudo_highlights, data_statistic
    return src_samples, tgt_samples, data_statistic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-article_path', required=True, help='article tsv file path')
    # parser.add_argument('-summ_path', required=True, help='summaries tsv file path')
    # parser.add_argument('-highlight_path', required=True, help='highlight tsv file path')
    parser.add_argument('-file_path', default='/data/senyang/summ_consistence/spacy_ner_data/', help='folder path of the input files')
    parser.add_argument('-save_path', default='/data/senyang/summ_consistence/spacy_ner_data_mask/', help='folder path of the output files')
    parser.add_argument('-mode', required=True, help='train, val or test')
    parser.add_argument('-chunk_size', default=20000, type=int, help='output file max size')
    parser.add_argument('-lev_true', action='store_true', default=False, help='whether for lev_transformer data')
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

    if not args.lev_true: 
        if args.mode == 'train': 
            for i in range(100):
                file_path = args.file_path + args.mode + '/spacy_ner_data_{}.pkl'.format(i)
                if not os.path.exists(file_path):
                    break
                tagged_data = load_ner_data(file_path)
                logger.info('Building pseudo data using {} to {} samples'.format(chunk_size*i, chunk_size*(i+1)))
                pseudo_data, data_statistic = main_func(tagged_data)
                for p, _ in enumerate( data_statistic):
                    pseudo_data_statistic[p] += _
                with open(args.save_path + args.mode +'/pseudo_data/pseudo_data_{}.pkl'.format(i), 'wb') as f:
                    pickle.dump(pseudo_data, f, protocol=3)
                logger.info('Builded {} samples. Successfully saved pseudo_data_{}.pkl'.format(len(pseudo_data), i))
        elif args.mode in {'val', 'test'}: 
            tagged_data = load_ner_data(args.file_path + args.mode +'/spacy_ner_data.pkl')
            logger.info('Building Peuso data.')
            pseudo_data, data_statistic = main_func(tagged_data)
            for p, _ in enumerate( data_statistic):
                pseudo_data_statistic[p] += _
            with open(args.save_path + args.mode +'/pseudo_data.pkl', 'wb') as f:
                pickle.dump(pseudo_data, f, protocol=3)
            logger.info('Builded {} samples. Successfully saved pseudo_data.pkl.'.format(len(pseudo_data)))
    else: 
        src, tgt = [], []
        if args.mode == 'train': 
            for i in range(100):
                file_path = args.file_path + args.mode + '/spacy_ner_data_{}.pkl'.format(i)
                if not os.path.exists(file_path):
                    break
                tagged_data = load_ner_data(file_path)
                logger.info('Building pseudo data using {} to {} samples'.format(chunk_size*i, chunk_size*(i+1)))
                tmp_src, tmp_tgt, data_statistic = main_func(tagged_data, mode='lev_transformer')
                src += tmp_src
                tgt += tmp_tgt
                for p, _ in enumerate( data_statistic):
                    pseudo_data_statistic[p] += _
                logger.info('Builded the {}_th chunk. '.format(i))
            logger.info('Start writing source file. ')
            with open(args.save_path + args.mode +'.src-tgt.src', 'w', encoding='utf-8') as f:
                for _ in src: 
                    f.write(_ + '\n')
            logger.info('Start writing target file. ')
            with open(args.save_path + args.mode +'.src-tgt.tgt', 'w', encoding='utf-8') as f:
                for _ in tgt: 
                    f.write(_ + '\n')
                
        elif args.mode in {'val', 'test'}: 
            tagged_data = load_ner_data(args.file_path + args.mode +'/spacy_ner_data.pkl')
            logger.info('Building Peuso data.')
            src, tgt, data_statistic = main_func(tagged_data, mode='lev_transformer')
            for p, _ in enumerate( data_statistic):
                pseudo_data_statistic[p] += _
            logger.info('Start writing source file. ')
            with open(args.save_path + args.mode +'.src-tgt.src', 'w', encoding='utf-8') as f:
                for _ in src: 
                    f.write(_ + '\n')
            logger.info('Start writing target file. ')
            with open(args.save_path + args.mode +'.src-tgt.tgt', 'w', encoding='utf-8') as f:
                for _ in tgt: 
                    f.write(_ + '\n')
    
    logger.info('Whole PseudoDataset statistics: {}'.format(pseudo_data_statistic))

