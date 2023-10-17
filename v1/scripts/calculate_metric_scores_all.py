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

def process_hyp(lp, trg, raw_src_sents, norm_src_sents, sent_annots, raw_sys_sents, norm_sys_sents, ref_sents, system_name='System', type_eval='ref-raw'):
  # debugging
  #raw_src_sents = raw_src_sents[:100]
  #norm_src_sents = norm_src_sents[:100]
  #sent_annots = sent_annots[:100]
  #sys_sents = sys_sents[:100]
  #ref_sents = [ref_sents[r][:100] for r in range(len(ref_sents))]
  #ref_sents = [ref_sents[0], [x[:-1] for x in ref_sents[0]]]

  if type_eval == 'ref-raw':
    cache_file = 'cache_results_wmt22-comet-da-raw/' + lp + '.' + system_name + '.pickle'
    subset2scores = calculate_all_refbased(raw_src_sents, raw_sys_sents, ref_sents, sent_annots,
                  cache_file=cache_file, system_name=system_name)
    print('REF-BASED-RAW')
    print_row(subset2scores, system_name=system_name)

  if type_eval == 'ref-norm':
    cache_file = 'cache_results_wmt22-comet-da-norm/' + lp + '.' + system_name + '.pickle'
    subset2scores = calculate_all_refbased(norm_src_sents, norm_sys_sents, ref_sents, sent_annots,
                  cache_file=cache_file, system_name=system_name)
    print('REF-BASED-NORM')
    print_row(subset2scores, system_name=system_name)

  if type_eval == 'qe-raw':
    cache_file = 'cache_results_wmt22-cometkiwi-da-raw/' + lp + '.' + system_name + '.pickle'
    calculate_all_qe([raw_src_sents, norm_src_sents], raw_sys_sents, annots=sent_annots, cache_file=cache_file,
                      system_name=system_name)
    print('QE-BASED-RAW')
    print_row(subset2scores, system_name=system_name)

  if type_eval == 'qe-norm':
    cache_file = 'cache_results_wmt22-cometkiwi-da-norm/' + lp + '.' + system_name + '.pickle'
    calculate_all_qe([raw_src_sents, norm_src_sents], norm_sys_sents, annots=sent_annots, cache_file=cache_file,
                      system_name=system_name)
    print('QE-BASE-NORM')
    print_row(subset2scores, system_name=system_name)

# all systems (ordered by logical order)
sys = ['GPT4-5shot', 'ONLINE-B', 'ONLINE-G',  'ONLINE-M', 'ONLINE-W', 'ONLINE-Y',
       'NLLB_MBR_BLEU', 'NLLB_Greedy', 'Lan-BridgeMT', 'ZengHuiMT', 'GTCOM_Peter',
       'AIRC', 'CUNI-DocTransformer', 'PROMT']

#type_eval = 'qe-norm'
#type_eval = 'qe-raw'
#type_eval = 'ref-norm'
type_eval = 'ref-raw'
  
# all language pairs
for lp in ['en-cs', 'en-de', 'en-ru', 'en-uk']:
  print(r'\midrule')
  print(lp + r' \\')
  print(r'\midrule')
  for sys_name in sys:
    # only proceed with compatible systems/lps
    hyp_file = '../sys/' + lp + '/' + sys_name + '.' + lp + '.txt'
    if not os.path.exists(hyp_file):
      continue
    
    all_files = get_files(hyp_file, lp)
    # ignore for ref-based eval those language pairs with no reference file
    if 'ref' in type_eval and all_files[0] not in ['en-cs', 'en-de', 'en-ru', 'en-uk']:
      continue
    print(hyp_file)
    process_hyp(*all_files, system_name=hyp_file.split('/')[-1], type_eval=type_eval)
  
