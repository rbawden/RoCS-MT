#!/usr/bin/python
from read_files import *
from sacrebleu.metrics import BLEU, CHRF, TER
from comet import download_model, load_from_checkpoint
from read_files import read_annots_file
import pickle
import torch
import os
comet_model_path = download_model("Unbabel/wmt22-comet-da")
#comet_model_path = '../../../../../../../../linkhome/rech/genini01/ulv12mq/.cache/huggingface/hub/models--Unbabel--wmt22-comet-da/snapshots/371e9839ca4e213dde891b066cf3080f75ec7e72/checkpoints/model.ckpt'
comet_model = load_from_checkpoint(comet_model_path)
comet_qe_model_path = download_model("Unbabel/wmt22-cometkiwi-da")
#comet_qe_model_path = '../../../../../../../../linkhome/rech/genini01/ulv12mq/.cache/huggingface/hub/models--Unbabel--wmt22-cometkiwi-da/snapshots/b3a8aea5a5fc22db68a554b92b3d96eb6ea75cc9/checkpoints/model.ckpt'
comet_qe_model = load_from_checkpoint(comet_qe_model_path)
bleu = BLEU()

THRESHOLD = 50

def read_file(filename, ref=False):
    sents = []
    if ref:
        num_refs = 1
        # get number of references
        with open(filename) as fp:
            for line in fp:
                new_num_refs = len(line.strip().split('\t'))
                if new_num_refs > num_refs:
                    num_refs = new_num_refs
        for r in range(num_refs):
            sents.append([])
    
    with open(filename) as fp:
        for line in fp:
            prep_line = line.split('\t')
            if not ref:
                assert len(prep_line) == 1, 'There should be one source sentence (and no more)'
                sents.append(prep_line[0].strip())
            else:                
                for r in range(num_refs):
                    if len(prep_line) < r or prep_line[r].strip() == '':
                        sents[r].append(prep_line[0].strip())
                    else:
                        # fill up sentences with copied if not the maximum number of refs
                        sents[r].append(prep_line[r].strip())
    return sents


def calc_bleu(sys_sents, ref_sents):
    print(len(sys_sents), [len(ref_sents[i]) for i in range(len(ref_sents))])
    assert len(sys_sents) == len(sys_sents)
    
    return bleu.corpus_score(sys_sents, ref_sents)

def calc_comet_qe(src_sents, sys_sents):
    print(len(src_sents), len(sys_sents))
    assert len(src_sents) == len(sys_sents)
    
    data = [{"src": src_sent, "mt": sys_sent} \
                for src_sent, sys_sent in zip(src_sents, sys_sents)]
    if torch.cuda.is_available():
        return comet_qe_model.predict(data, batch_size=32).scores
    else:
        return comet_qe_model.predict(data, batch_size=32, gpus=0).scores


def calc_comet(src_sents, sys_sents, ref_sents):

    print(len(src_sents), len(sys_sents), len(ref_sents))
    assert len(src_sents) == len(sys_sents) == len(ref_sents)
    data = [{"src": src_sent, "mt": sys_sent, "ref": ref_sent} \
                for src_sent, sys_sent, ref_sent in zip(src_sents, sys_sents, ref_sents)]
    if torch.cuda.is_available():
        return comet_model.predict(data, batch_size=32).scores
    else:
        return comet_model.predict(data, batch_size=32, gpus=0).scores


def partition_sents(src_sents, sys_sents, ref_sents, annots):
    phen2data = {}
    assert len(src_sents) == len(sys_sents) == len(annots)
    assert all([len(x) == len(src_sents) for x in ref_sents])
    for i, (src_sent, sys_sent, sent_annot) in enumerate(zip(src_sents, sys_sents, annots)):
        phens = set([y.strip() for x in sent_annot for y in x['manual'].split(',')  if x['manual'] != ''])
        for phen in phens:
            if phen not in phen2data:
                phen2data[phen] = {'idx': [], 'src': [], 'sys': [], 'ref': [[] for _ in range(len(ref_sents))]}
            phen2data[phen]['src'].append(src_sent)
            phen2data[phen]['sys'].append(sys_sent)
            phen2data[phen]['idx'].append(i)
            for r in range(len(ref_sents)):
                phen2data[phen]['ref'][r].append(ref_sents[r][i])

    return phen2data


def mean(list_values):
    print(list_values)
    return sum(list_values)/len(list_values)

    
def calculate_all_qe(src_sents, sys_sents, annots=None, cache_file=None, system_name='System'):
    '''
    src_sents is a list of sets of src_sents (like ref_sents for calculate_all_refbased). Each list
    should have the same number of sentences.
    '''
    subset2scores = {}
    if cache_file is not None and os.path.exists(cache_file):
        subset2scores = pickle.load(open(cache_file, 'rb'))
    dict_init(subset2scores, 'all', {})
    
    if 'comet' not in subset2scores['all']:
        subset2scores['all']['#sents'] = len(sys_sents)
        # calculate comet scores for each of the sets of src sentences
        for r in range(len(src_sents)):
            comet_scores = calc_comet_qe(src_sents[r], sys_sents)
            subset2scores['all']['comet-individual-' + str(r)] = comet_scores
            ave_comet_score = sum(comet_scores)/len(comet_scores)
            subset2scores['all']['comet-ave-' + str(r)] = ave_comet_score
            
    # calculate comet-qe by partitioned data (for each phenomenon if annots provided)
    if annots is not None:
        phen2data = partition_sents(src_sents[0], sys_sents, src_sents, annots)
        for phen in phen2data:
            if phen not in subset2scores:
                subset2scores[phen] = {}
                subset2scores[phen]['#sents'] = len(phen2data[phen]['src'])
                # for each of the sets of src_sents
                for s in range(len(src_sents)):
                    # get the individual comet scores
                    subset2scores[phen]['comet-individual-' + str(s)] = []
                    for sent_idx in subset2scores[phen]['idx']:
                        subset2scores[phen]['comet-individual-' + str(s)].append(subset2scores['all']['comet-individual-' + str(s)])
                    # calculate the average
                    subset2scores[phen]['comet-ave-' + str(s)] = mean(subset2scores[phen]['comet-individual-' + str(s)])
                # for each sent index, get the best score out of each of the sets of src_sents
                subset2scores[phen]['comet-individual-best'] = []
                subset2scores[phen]['comet-individual-best-idx'] = []
                for s in range(len(phen2data[phen]['src'])):
                    all_scores = [(j, subset2scores[phen]['comet-individual-' + str(j)][s]) for j in range(len(src_sents))]
                    idx, val = sorted(all_scores, lambda key=x: x[1], reverse=True)[0]
                    subset2scores[phen]['comet-individual-best'].append(val)
                    subset2scores[phen]['comet-individual-best-idx'].append(idx)
                # get the average of best
                subset2scores[phen]['comet-ave-best'] = mean(subset2scores[phen]['comet-individual-best'])
                
    if cache_file is not None:
        pickle.dump(subset2scores, open(cache_file, 'wb'))
    return subset2scores


def dict_init(dico, new_key, new_value={}):
    if new_key not in dico:
        dico[new_key] = new_value
    return 


def calculate_all_refbased(src_sents, sys_sents, ref_sents, annots=None, comet_too=True,
                          cache_file=None, system_name='System'):
    subset2scores = {}
    if cache_file is not None and os.path.exists(cache_file):
        subset2scores = pickle.load(open(cache_file, 'rb'))
       
    dict_init(subset2scores, 'all', {})
    # bleu scores on all sentences
    if 'bleu' not in subset2scores['all']:
        subset2scores['all']['#sents'] = len(sys_sents)
        bleu_score = calc_bleu(sys_sents, ref_sents)
        subset2scores['all']['bleu'] = bleu_score.score
        
    # comet scores on all sentences
    if comet_too and 'comet' not in subset2scores['all']:
       # for each set of reference sentences
        for s in range(len(ref_sents)):
            comet_scores = calc_comet(src_sents, sys_sents, ref_sents[s])
            # scores for each sentence individually
            subset2scores['all']['comet-individual-' + str(s)] = comet_scores
            # average of those scores
            ave_comet_score = sum(comet_scores)/len(comet_scores)
            subset2scores['all']['comet-ave-' + str(s)] = ave_comet_score
        # average of all sets of reference sentences
        subset2scores['all']['comet-individual-ave'] = []
        for s in range(len(ref_sents[0])):
            all_ref_scores = [subset2scores['all']['comet-individual-' + str(r)][s] for r in range(len(ref_sents))]
            subset2scores['all']['comet-individual-ave'].append(mean(all_ref_scores))
        # average of this average
        subset2scores['all']['comet-ave'] = mean(subset2scores['all']['comet-individual-ave'])
            
    # calculate bleu and comet by partitioned data (for each phenomenon if annots provided)
    if annots is not None:
        phen2data = partition_sents(src_sents, sys_sents, ref_sents, annots)
        # individually for each phenomenon
        for phen in phen2data:
            # calculate bleu on partition
            bleu_score = calc_bleu(phen2data[phen]['sys'], phen2data[phen]['ref']).score
            if phen not in subset2scores:
                subset2scores[phen] = {}
                if 'bleu' not in subset2scores[phen]:
                    subset2scores[phen]['bleu'] = bleu_score
                    subset2scores[phen]['#sents'] = len(phen2data[phen]['src'])
                # get comet scores from individual scores calculated above (for the most common)
                if 'comet' not in subset2scores[phen] and comet_too and subset2scores[phen]['#sents'] > THRESHOLD:
                    # get scores (individual and average for each set of references)
                    for r in range(len(ref_sents)):
                        # get the individual comet scores for this phenomenon
                        subset2scores[phen]['comet-individual-' + str(r)] = []
                        for sent_idx in phen2data[phen]['idx']:
                            sent_score = subset2scores['all']['comet-individual-' + str(r)][sent_idx]
                            subset2scores[phen]['comet-individual-' + str(r)].append(sent_score)
                        # calculate the average for this set of reference
                        subset2scores[phen]['comet-ave-' + str(r)] = mean(subset2scores[phen]['comet-individual-' + str(r)])
                    # get the average between the scores for all references
                    subset2scores[phen]['comet-individual-ave'] = []
                    # for each of the reference sentences
                    for s in range(len(phen2data[phen]['ref'])):
                        all_scores = [subset2scores[phen]['comet-individual-' + str(r)][s] for r in range(len(ref_sents))]
                        subset2scores[phen]['comet-individual-ave'].append(mean(all_scores))
                    # get the average over all sets of references
                    subset2scores[phen]['comet-ave'] = mean(subset2scores[phen]['comet-individual-ave'])
    if cache_file is not None:
        pickle.dump(subset2scores, open(cache_file, 'wb'))
    return subset2scores


def prep_v(value_to_prep, round_val=1):
    return r'\gradient{' + str(round(value_to_prep, round_val)) + '}'

def print_row(subset2scores, system_name='System'):
    phens = [x for x in sorted(subset2scores.keys()) if x != 'all' and subset2scores[x]['#sents'] > THRESHOLD]
    # print headers
    #print('System' + ' & ' * int(len(phens) > 0) + ' & '.join([x.replace('_', '\_') for x in phens]) + r' & all \\')
    # print values for this system
    for metric in subset2scores['all']:
        if metric not in ['bleu', 'comet']:
            continue
        prec = 1
        if metric == 'comet':
            prec = 3
        prep_system_name = re.sub('\.en-..\.txt', '', system_name.replace('_', '\_'))
        print('*' + metric + '* ' + prep_system_name + ' & '+ ' & '.join([prep_v(subset2scores[phen][metric],  prec) for phen in phens] +
                                                                         [prep_v(subset2scores['all'][metric], prec)]) + r' \\')

    # print number of sentences -> check that this is the same for each system
    print('#sents & ' + ' & '.join([str(subset2scores[phen]['#sents']) for phen in phens] +
                                   [str(subset2scores['all']['#sents'])]) + r' \\')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('src', help='source file (one sentence per line)')
    parser.add_argument('sys', help='system file (one sentence per line)')
    parser.add_argument('ref', help='reference file (one sentence per line - tab separated if several references)')
    parser.add_argument('-a', '--annots', help='Annotation file', default=None)
    parser.add_argument('-c', '--cache', help='Cache file', default=None)
    args = parser.parse_args()

    src_sents = read_file(args.src)
    sys_sents = read_file(args.sys)
    ref_sents = read_file(args.ref, ref=True)
    sent_annots = None
    if args.annots is not None:
        sent_annots = read_annots_file(args.annots)

    subset2scores = calculate_all_refbased(src_sents, sys_sents, ref_sents, sent_annots, comet_too=False, cache_file=args.cache)
    print_row(subset2scores)
    
