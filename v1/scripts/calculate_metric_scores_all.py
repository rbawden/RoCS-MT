#!/usr/bin/python
from calculate_metric_scores import *
import os

def get_files(hyp_file, lp):
  trg = lp.split('-')[-1]
  raw_src_sents = read_file('../src/RoCS-MT.src.raw-manseg.en')
  norm_src_sents = read_file('../src/RoCS-MT.src.norm-manseg.en')
  sent_annots = read_annots_file('../ref/RoCS-annotated.tsv')
  sys_sents = read_file(hyp_file)
  sys_sent_docs = read_file(re.sub('.txt', '.ids.txt', hyp_file))
  raw_sys_sents = [x for i, x in enumerate(sys_sents) if 'rm' in sys_sent_docs[i]]
  norm_sys_sents = [x for i, x in enumerate(sys_sents) if 'nm' in sys_sent_docs[i]]
  if not os.path.exists('../ref/RoCS-MT.ref.' + trg):
    ref_sents = None
  else:
    ref_sents = read_file('../ref/RoCS-MT.ref.' + trg, ref=True)

  return lp, trg, raw_src_sents, norm_src_sents, sent_annots, raw_sys_sents, norm_sys_sents, ref_sents


def process_hyp(lp, trg, raw_src_sents, norm_src_sents, sent_annots, raw_sys_sents,
                  norm_sys_sents, ref_sents, system_name='System', type_eval='ref-raw'):

  if type_eval == 'bleu-raw':
    subset2scores = calculate_all_refbased(raw_src_sents, raw_sys_sents, ref_sents, 
                                      sent_annots, cache_file=None, system_name=system_name, comet_too=False)
    #print('BLEU-RAW')
    print_row(subset2scores, 'bleu', system_name=system_name)

  elif type_eval == 'bleu-norm':
    subset2scores = calculate_all_refbased(norm_src_sents, norm_sys_sents, ref_sents,
                                      sent_annots, cache_file=None, system_name=system_name, comet_too=False)
    #print('BLEU-NORM')
    print_row(subset2scores, 'bleu', system_name=system_name)

  elif type_eval == 'bleu-diff':
    subset2scores_nm = calculate_all_refbased(norm_src_sents, norm_sys_sents, ref_sents,
                                      sent_annots, cache_file=None, system_name=system_name, comet_too=False)
    subset2scores_rm = calculate_all_refbased(raw_src_sents, raw_sys_sents, ref_sents,
                                      sent_annots, cache_file=None, system_name=system_name, comet_too=False)
    print_row_diff(subset2scores_nm, subset2scores_rm, 'bleu', system_name=system_name)
    
  elif type_eval == 'comet-raw':
    cache_file = 'cache_results_wmt22-comet-da-raw/' + lp + '.' + system_name + '.pickle'
    subset2scores = calculate_all([raw_src_sents, norm_src_sents], raw_sys_sents, ref_sents, calc_comet_several_refs,
                                             sent_annots, cache_file=cache_file, system_name=system_name)
    print('REF-BASED-RAW')
    print_row(subset2scores, 'comet-ave-best', system_name=system_name)

  elif type_eval == 'comet-norm':
    cache_file = 'cache_results_wmt22-comet-da-norm/' + lp + '.' + system_name + '.pickle'
    subset2scores = calculate_all([raw_src_sents, norm_src_sents], norm_sys_sents, ref_sents, calc_comet_several_refs,
                                             sent_annots, cache_file=cache_file, system_name=system_name)
    print('REF-BASED-NORM')
    print_row(subset2scores, 'comet-ave-best', system_name=system_name)

  elif type_eval == 'comet-diff':
    cache_file = 'cache_results_wmt22-comet-da-norm/' + lp + '.' + system_name + '.pickle'
    subset2scores_nm = calculate_all([raw_src_sents, norm_src_sents], norm_sys_sents, ref_sents, calc_comet_several_refs,
                                             sent_annots, cache_file=cache_file, system_name=system_name)
    cache_file = 'cache_results_wmt22-comet-da-raw/' + lp + '.' + system_name + '.pickle'
    subset2scores_rm = calculate_all([raw_src_sents, norm_src_sents], norm_sys_sents, ref_sents, calc_comet_several_refs,
                                             sent_annots, cache_file=cache_file, system_name=system_name)
    print_row_diff(subset2scores_nm, subset2scores_rm, 'comet-ave-best', system_name=system_name)

  elif type_eval == 'cometqe-raw':
    cache_file = 'cache_results_wmt22-cometkiwi-da-raw/' + lp + '.' + system_name + '.pickle'
    subset2scores = calculate_all([raw_src_sents, norm_src_sents], raw_sys_sents, ref_sents, calc_comet_qe,
                                       annots=sent_annots, cache_file=cache_file, system_name=system_name)
    print('QE-BASED-RAW')
    print_row(subset2scores, 'comet-ave-best-0', system_name=system_name)

  elif type_eval == 'cometqe-norm':
    cache_file = 'cache_results_wmt22-cometkiwi-da-norm/' + lp + '.' + system_name + '.pickle'
    subset2scores = calculate_all([raw_src_sents, norm_src_sents], norm_sys_sents, ref_sents, calc_comet_qe,
                                       annots=sent_annots, cache_file=cache_file, system_name=system_name)
    print('QE-BASE-NORM')
    print_row(subset2scores, 'comet-ave-best', system_name=system_name)

  elif type_eval == 'cometqe-diff':
    cache_file = 'cache_results_wmt22-cometkiwi-da-norm/' + lp + '.' + system_name + '.pickle'
    subset2scores_nm = calculate_all([raw_src_sents, norm_src_sents], norm_sys_sents, ref_sents, calc_comet_qe,
                                       annots=sent_annots, cache_file=cache_file, system_name=system_name)
    cache_file = 'cache_results_wmt22-cometkiwi-da-raw/' + lp + '.' + system_name + '.pickle'
    subset2scores_rm = calculate_all([raw_src_sents, norm_src_sents], raw_sys_sents, ref_sents, calc_comet_qe,
                                       annots=sent_annots, cache_file=cache_file, system_name=system_name)
    print_row_diff(subset2scores_nm, subset2scores_rm, 'comet-ave-best', system_name=system_name)
    

# all systems (ordered by logical order)
sys = ['GPT4-5shot', 'ONLINE-B', 'ONLINE-G',  'ONLINE-M', 'ONLINE-W', 'ONLINE-Y',
       'NLLB_MBR_BLEU', 'NLLB_Greedy', 'Lan-BridgeMT', 'ZengHuiMT', 'GTCOM_Peter',
       'AIRC', 'CUNI-DocTransformer', 'PROMT']

type_eval = 'cometqe-norm'
#type_eval = 'cometqe-raw'
#type_eval = 'comet-norm'
#type_eval = 'comet-raw'
#type_eval = 'bleu-norm'
#type_eval = 'bleu-raw'
#type_eval = 'bleu-diff'
#type_eval = 'comet-diff'
#type_eval = 'cometqe-diff'

lps = ['en-cs', 'en-de', 'en-he', 'en-ja', 'en-ru', 'en-uk', 'en-zh']
ref_lps = ['en-cs', 'en-de', 'en-ru', 'en-uk']
systems = [# two lines unconstrained
           'GPT4-5shot', 'ONLINE-A', 'ONLINE-B', 'ONLINE-G',  'ONLINE-M', 'ONLINE-W', 'ONLINE-Y',
           'NLLB_MBR_BLEU', 'NLLB_Greedy', 'Lan-BridgeMT', 'GTCOM_Peter', 'KYB', 'PROMT', 'Yishu', 'ZengHuiMT',
           # two lines constrained
           'AIRC', 'ANVITA', 'CUNI-Transformer', 'CUNI-DocTransformer', 'CUNI-GA',
           'HW-TSC', 'IOL_Research', 'NAIST-NICT', 'Samsung_Research_Philippines', 'SKIM', 'UvA-LTL']

# all language pairs
for lp in lps:
  print(r'\midrule')
  print(lp + r' \\')
  print(r'\midrule')
  for sys_name in systems:
    # only proceed with compatible systems/lps
    hyp_file = '../sys/' + lp + '/' + sys_name + '.' + lp + '.txt'
    if not os.path.exists(hyp_file):
      continue
    
    all_files = get_files(hyp_file, lp)
    # ignore for ref-based eval those language pairs with no reference file
    if 'qe' not in type_eval and lp not in ['en-cs', 'en-de', 'en-ru', 'en-uk']:
      print('skipping')
      continue
    process_hyp(*all_files, system_name=hyp_file.split('/')[-1], type_eval=type_eval)
  
