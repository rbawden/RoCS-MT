#!/usr/bin/python
from read_files import *
from sacrebleu.metrics import BLEU, CHRF, TER
from comet import download_model, load_from_checkpoint
from read_files import read_annots_file
import pickle
import torch
import os
comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)
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
    return bleu.corpus_score(sys_sents, ref_sents)


def calc_comet(src_sents, sys_sents, ref_sents):
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
                phen2data[phen] = {'src': [], 'sys': [], 'ref': [[] for _ in range(len(ref_sents))]}
            phen2data[phen]['src'].append(src_sent)
            phen2data[phen]['sys'].append(sys_sent)
            for r in range(len(ref_sents)):
                phen2data[phen]['ref'][r].append(ref_sents[r][i])

    return phen2data
            
    
def calculate_all(src_sents, sys_sents, ref_sents, annots=None, comet_too=True, cache_file=None, system_name='System'):
    if cache_file is not None and os.path.exists(cache_file):
        subset2scores = pickle.load(open(cache_file, 'rb'))
    else:
        subset2scores = {}
    # BLEU scores and COMET scores on entire set
    if 'all' not in subset2scores:
        subset2scores['all'] = {}
    if 'bleu' not in subset2scores['all']:
        bleu_score = calc_bleu(sys_sents, ref_sents)
        subset2scores['all']['bleu'] = bleu_score.score
        subset2scores['all']['#sents'] = len(sys_sents)
    if comet_too and 'comet' not in subset2scores['all']:
        comet_scores = calc_comet(src_sents, sys_sents, ref_sents)
        ave_comet_score = sum(comet_scores)/len(comet_scores)
        subset2scores['all']['comet'] = ave_comet_score

    # by partition if annots provided
    if annots is not None:
        phen2data = partition_sents(src_sents, sys_sents, ref_sents, annots)
        for i, phen in enumerate(phen2data):
            #print(phen, str(i) + '/' + str(len(phen2data)))
            bleu_score = calc_bleu(phen2data[phen]['sys'], phen2data[phen]['ref']).score
            if phen not in subset2scores:
                subset2scores[phen] = {}
                if 'bleu' not in subset2scores[phen]:
                    subset2scores[phen]['bleu'] = bleu_score
                    subset2scores[phen]['#sents'] = len(phen2data[phen]['src'])

                if 'comet' not in subset2scores[phen] and comet_too and subset2scores[phen]['#sents'] > THRESHOLD:
                    comet_scores_individual = []
                    ave_comet_scores = []
                    for ref_trans in phen2data[phen]['ref']:
                        comet_scores = calc_comet(phen2data[phen]['src'], phen2data[phen]['sys'], phen2data[phen]['ref'])
                        print(comet_scores)

                        comet_scores_individual.append(comet_scores)
                        ave_comet_scores.append(sum(comet_scores)/len(comet_scores))
                    subset2scores[phen]['comet'] = sum(ave_comet_scores)/len(ave_comet_scores)
                    subset2scores[phen]['comet-individual'] = comet_scores_individual
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
        if metric == 'comet':
            prec = 3
        else:
            prec = 1
        print(re.sub('\.en-..\.txt', '', system_name.replace('_', '\_')) + ' & '+ ' & '.join([prep_v(subset2scores[phen][metric],  prec) for phen in phens] +
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

    subset2scores = calculate_all(src_sents, sys_sents, ref_sents, sent_annots, comet_too=True, cache_file=args.cache)
    print_row(subset2scores)
    
