#!/usr/bin/python
from read_files import *
from sacrebleu.metrics import BLEU, CHRF, TER
from comet import download_model, load_from_checkpoint
from read_files import read_annots_file
import pickle
import torch
import os
#comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model_path = '../../../../../../../../linkhome/rech/genini01/ulv12mq/.cache/huggingface/hub/models--Unbabel--wmt22-comet-da/snapshots/371e9839ca4e213dde891b066cf3080f75ec7e72/checkpoints/model.ckpt'
comet_model = load_from_checkpoint(comet_model_path)
#comet_qe_model_path = download_model("Unbabel/wmt22-cometkiwi-da")
comet_qe_model_path = '../../../../../../../../linkhome/rech/genini01/ulv12mq/.cache/huggingface/hub/models--Unbabel--wmt22-cometkiwi-da/snapshots/b3a8aea5a5fc22db68a554b92b3d96eb6ea75cc9/checkpoints/model.ckpt'
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
    #print(len(sys_sents), [len(ref_sents[i]) for i in range(len(ref_sents))])
    assert len(sys_sents) == len(sys_sents)
    
    return bleu.corpus_score(sys_sents, ref_sents)

def calc_comet_qe(src_sents, sys_sents, ref_sents):
    #print(len(src_sents), len(sys_sents))
    assert len(src_sents) == len(sys_sents)
    
    data = [{"src": src_sent, "mt": sys_sent} \
                for src_sent, sys_sent in zip(src_sents, sys_sents)]
    if torch.cuda.is_available():
        return comet_qe_model.predict(data, batch_size=32).scores
    else:
        return comet_qe_model.predict(data, batch_size=32, gpus=0).scores


def calc_comet(src_sents, sys_sents, ref_sents):
    #print(len(src_sents), len(sys_sents), len(ref_sents))
    assert len(src_sents) == len(sys_sents) == len(ref_sents)
    data = [{"src": src_sent, "mt": sys_sent, "ref": ref_sent} \
                for src_sent, sys_sent, ref_sent in zip(src_sents, sys_sents, ref_sents)]
    if torch.cuda.is_available():
        return comet_model.predict(data, batch_size=32).scores
    else:
        return comet_model.predict(data, batch_size=32, gpus=0).scores

def calc_comet_several_refs(src_sents, sys_sents, ref_sents):
    assert len(src_sents) == len(sys_sents)
    assert all(len(r) == len(sys_sents) for r in ref_sents)
    scores = []
    for refset in ref_sents:
        data = [{"src": src_sent, "mt": sys_sent, "ref": ref_sent} \
                    for src_sent, sys_sent, ref_sent in zip(src_sents, sys_sents, refset)]
        if torch.cuda.is_available():
            ref_set_scores = comet_model.predict(data, batch_size=32).scores
        else:
            ref_set_scores = comet_model.predict(data, batch_size=32, gpus=0).scores
        scores.append(ref_set_scores)
    ave_scores = []
    # average for each sentence
    for s in range(len(ref_sents[0])):
        ave_scores.append(mean([scores[r][s] for r in range(len(ref_sents))]))
    return ave_scores

    
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
    #print(list_values)
    return sum(list_values)/len(list_values)

    
def calculate_all_comet(set_src_sents, sys_sents, ref_sents, comet_func, annots=None, cache_file=None, system_name='System'):
    '''
    src_sents is a list of sets of src_sents (like ref_sents for calculate_all_refbased). Each list
    should have the same number of sentences.
    '''
    if cache_file is not None and os.path.exists(cache_file):
        subset2scores = pickle.load(open(cache_file, 'rb'))
    else:
        subset2scores = {}
    dict_init(subset2scores, 'all', {})

    # over the entire test set, calculate comet scores for each of the sets of src sentences and averaged
    if 'comet-individual-0' not in subset2scores['all']:
        subset2scores['all']['#sents'] = len(sys_sents)
        for src_set_num in range(len(set_src_sents)):
            for ref_set_num in range(len(ref_sents)):
                list_comet_scores = comet_func(set_src_sents[src_set_num], sys_sents, ref_sents)
                # sentence-level scores for that src set and ref set
                subset2scores['all']['comet-individual-' + str(src_set_num) + '-' + str(ref_set_num)] = list_comet_scores
                # corpus average for that src_set and ref_set
                subset2scores['all']['comet-ave-' + str(src_set_num) + '-' + str(ref_set_num)] = mean(list_comet_scores)

        subset2scores['all']['comet-individual-best'] = []
        subset2scores['all']['comet-individual-best-idx'] = []
        subset2scores['all']['comet-individual-ave'] = []
        # get average and best for each sent
        for sent_idx in range(len(sys_sents)):
            # get all scores for this sentence for all combinations of src and refs sets
            all_scores = [((j, k), subset2scores['all']['comet-individual-' + str(j) + '-' + str(k)][sent_idx]) \
                              for j in range(len(set_src_sents)) for k in range(len(ref_sents))]
            # get best score for this sentence
            idx, val = sorted(all_scores, key=lambda x: x[1], reverse=True)[0]
            subset2scores['all']['comet-individual-best'].append(val)
            subset2scores['all']['comet-individual-best-idx'].append(idx)
            # get average score for this sentence
            subset2scores['all']['comet-individual-ave'].append(mean([x[1] for x in all_scores]))
        # get the corpus average over best individual sent scores
        subset2scores['all']['comet-ave-best'] = mean([subset2scores['all']['comet-individual-best'][s] for s in range(len(sys_sents))])
        # get theh corpus average over average indvidual sent scores
        subset2scores['all']['comet-ave-ave'] = mean([subset2scores['all']['comet-individual-ave'][s] for s in range(len(set_src_sents))])
            
    # calculate comet by partitioned data (for each phenomenon if annots provided)
    if annots is not None:
        phen2data = partition_sents(set_src_sents[0], sys_sents, set_src_sents, annots)
        for phen in phen2data:
            dict_init(subset2scores, phen, {})
            subset2scores[phen]['#sents'] = len(phen2data[phen]['src'])

            # get average and best scores for each individaul sentence
            for src_set_num in range(len(set_src_sents)):
                for ref_set_num in range(len(ref_sents)):
                    # get individual comet scores for this src and ref set for this particular phenomenon
                    subset2scores[phen]['comet-individual-' + str(src_set_num) + '-' + str(ref_set_num)] = []
                    for sent_idx in phen2data[phen]['idx']:
                        sent_score = subset2scores['all']['comet-individual-' + str(src_set_num) + '-' + str(ref_set_num)][sent_idx]
                        subset2scores[phen]['comet-individual-' + str(src_set_num) + '-' + str(ref_set_num)].append(sent_score)
                    # calculate the corpus average for this set of src_set and ref_set for this particular phenomenon
                    subset2scores[phen]['comet-ave-' + str(src_set_num) + '-' + str(ref_set_num)] = \
                      mean(subset2scores[phen]['comet-individual-' + str(src_set_num) + '-' + str(ref_set_num)])

            # for each sent, get the ave and best score over src and ref sets
            subset2scores[phen]['comet-individual-ave'] = []
            subset2scores[phen]['comet-individual-best'] = []
            subset2scores[phen]['comet-individual-best-idx'] = []
            for s, sent_idx in enumerate(phen2data[phen]['idx']):
                # get scores for each of the src andn ref sets
                all_scores = [((j,k), subset2scores[phen]['comet-individual-' + str(j) + '-' + str(k)][s]) for j in range(len(set_src_sents)) for k in range(len(ref_sents))]
                idx, val = sorted(all_scores, key=lambda x: x[1], reverse=True)[0]
                # get the best score for each sentence
                subset2scores[phen]['comet-individual-best'].append(val)
                subset2scores[phen]['comet-individual-best-idx'].append(idx)
                # get average score for this sent
                subset2scores[phen]['comet-individual-ave'] = [mean([x[1] for x in all_scores])]
                
            # get the average over all sentence-level average scores
            subset2scores[phen]['comet-ave-ave'] = mean(subset2scores[phen]['comet-individual-ave'])
            # get the average over all sentence-level best scores
            subset2scores[phen]['comet-ave-best'] = mean(subset2scores[phen]['comet-individual-best'])
                
    if cache_file is not None:
        pickle.dump(subset2scores, open(cache_file, 'wb'))
    return subset2scores


def dict_init(dico, new_key, new_value={}):
    if new_key not in dico:
        dico[new_key] = new_value
    return 

# only use for BLEU!
def calculate_all_bleu(src_sents, sys_sents, set_ref_sents, annots=None, comet_too=False,
                          cache_file=None, system_name='System'):
    
    if cache_file is not None and os.path.exists(cache_file):
        subset2scores = pickle.load(open(cache_file, 'rb'))
    else:
        subset2scores = {}
       
    dict_init(subset2scores, 'all', {})
    # bleu score calculated for all sentences
    if 'bleu' not in subset2scores['all']:
        subset2scores['all']['#sents'] = len(sys_sents)
        subset2scores['all']['bleu'] = calc_bleu(sys_sents, set_ref_sents).score

    # individual comet scores (and their average) for all sentences
    if comet_too and 'comet-individual-0' not in subset2scores['all']:
        # for each set of reference sentences
        for num_ref_set in range(len(set_ref_sents)):
            comet_scores = calc_comet(src_sents, sys_sents, set_ref_sents[num_ref_set])
            subset2scores['all']['comet-individual-' + str(num_ref_set)] = comet_scores
            subset2scores['all']['comet-ave-' + str(num_ref_set)] = mean(comet_scores)
        
            
        # for each sentence calculate average of all scores for all sets of refs
        subset2scores['all']['comet-individual-ave'] = []
        for sent_idx in range(len(sys_sents)):
            all_scores = [subset2scores['all']['comet-individual-' + str(num_ref_set)][sent_idx] for num_ref_set in range(len(set_ref_sents))]
            subset2scores['all']['comet-individual-ave'].append(mean(all_scores))
        # average of these scores
        subset2scores['all']['comet-ave'] = mean(subset2scores['all']['comet-individual-ave'])
            
    # calculate bleu and comet by partitioned data (for each phenomenon if annots provided)
    if annots is not None:
        phen2data = partition_sents(src_sents, sys_sents, set_ref_sents, annots)
        # individually for each phenomenon
        for phen in phen2data:
            # calculate bleu for this phenomenon
            bleu_score = calc_bleu(phen2data[phen]['sys'], phen2data[phen]['ref']).score
            dict_init(subset2scores, phen, {})
            if 'bleu' not in subset2scores[phen]:
                subset2scores[phen]['bleu'] = bleu_score
                subset2scores[phen]['#sents'] = len(phen2data[phen]['src'])

            #print(phen)
            # get comet scores from individual scores calculated above (for the most common)
            if 'comet' not in subset2scores[phen] and comet_too and subset2scores[phen]['#sents'] > THRESHOLD:
                # get each set of references, calculate individual scores and an average score
                for num_ref_set in range(len(set_ref_sents)):
                    subset2scores[phen]['comet-individual-' + str(num_ref_set)] = []
                    for s, sent_idx in enumerate(phen2data[phen]['idx']):
                        #print(sent_idx, len(subset2scores['all']['comet-individual-' + str(num_ref_set)]))
                        sent_score = subset2scores['all']['comet-individual-' + str(num_ref_set)][sent_idx]
                        subset2scores[phen]['comet-individual-' + str(num_ref_set)].append(sent_score)
                    # calculate the average for this set of reference
                    subset2scores[phen]['comet-ave-' + str(num_ref_set)] = mean(subset2scores[phen]['comet-individual-' + str(num_ref_set)])
                    
                # get the average between the scores for all references
                subset2scores[phen]['comet-individual-ave'] = []
                # for each of the reference sentences calculate the average over the ref sets
                for s, sent_idx in enumerate(phen2data[phen]['idx']):
                    all_scores = [subset2scores[phen]['comet-individual-' + str(num_ref_set)][s] for num_ref_set in range(len(set_ref_sents))]
                    subset2scores[phen]['comet-individual-ave'].append(mean(all_scores))
                # get the average over all sets of references
                subset2scores[phen]['comet-ave'] = mean(subset2scores[phen]['comet-individual-ave'])
    if cache_file is not None:
        pickle.dump(subset2scores, open(cache_file, 'wb'))
    return subset2scores


def prep_v(value_to_prep, round_val=1):
    if round_val == 1:
        return r'\gradient{' + f"{value_to_prep:.{round_val}f}" + '}'
    else:
        return r'\cometgradient{' + f"{value_to_prep:.{round_val}f}" + '}'

def print_row(subset2scores, metric, system_name='System'):
    phens = [x for x in sorted(subset2scores.keys()) if x != 'all' and subset2scores[x]['#sents'] > THRESHOLD]
    # print headers
    #print('System' + ' & ' * int(len(phens) > 0) + ' & '.join([x.replace('_', '\_') for x in phens]) + r' & all \\')
    prec = 1
    if 'comet' in metric:
        prec = 3
    prep_system_name = re.sub('\.en-..\.txt', '', system_name.replace('_', '\_'))
    print(prep_system_name + ' & '+ ' & '.join([prep_v(subset2scores[phen][metric],  prec) for phen in phens] +
                                                   [prep_v(subset2scores['all'][metric], prec)]) + r' \\')

    # print number of sentences -> check that this is the same for each system
    #print('#sents & ' + ' & '.join([str(subset2scores[phen]['#sents']) for phen in phens] + [str(subset2scores['all']['#sents'])]) + r' \\')

def print_row_diff(subset2scores1, subset2scores2, metric, system_name='System'):
    phens = [x for x in sorted(subset2scores2.keys()) if x != 'all' and subset2scores2[x]['#sents'] > THRESHOLD]
    # print headers
    #print('System' + ' & ' * int(len(phens) > 0) + ' & '.join([x.replace('_', '\_') for x in phens]) + r' & all \\')
    prec = 1
    if 'comet' in metric:
        prec = 3
    prep_system_name = re.sub('\.en-..\.txt', '', system_name.replace('_', '\_'))
    print(prep_system_name + ' & '+ ' & '.join([prep_v(subset2scores1[phen][metric] - subset2scores2[phen][metric],  prec) for phen in phens] +
                                               [prep_v(subset2scores1['all'][metric] - subset2scores2['all'][metric], prec)]) + r' \\')
    

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
    
