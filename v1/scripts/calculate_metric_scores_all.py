#!/usr/bin/python
from calculate_metric_scores import *
import os

def process_hyp(hyp_file, lp, system_name='System'):
  trg = lp.split('-')[-1]
  raw_src_sents = read_file('../src/RoCS-MT.src.raw-manseg.en')
  norm_src_sents = read_file('../src/RoCS-MT.src.norm-manseg.en')
  sent_annots = read_annots_file('../ref/RoCS-annotated.tsv')
  sys_sents = read_file(hyp_file)
  sys_sent_docs = read_file(re.sub('.txt', '.ids.txt', hyp_file))
  sys_sents = [x for i, x in enumerate(sys_sents) if 'rm' in sys_sent_docs[i]]
  if not os.path.exists('../ref/RoCS-MT.ref.' + trg):
    return
  ref_sents = read_file('../ref/RoCS-MT.ref.' + trg, ref=True)

  # debugging
  raw_src_sents = raw_src_sents[:100]
  norm_src_sents = norm_src_sents[:100]
  sent_annots = sent_annots[:100]
  sys_sents = sys_sents[:100]
  ref_sents = [ref_sents[r][:100] for r in range(len(ref_sents))]
  ref_sents = [ref_sents[0], [x[:-1] for x in ref_sents[0]]]
  
  cache_file = 'cache_results_wmt22-comet-da/' + lp + '.' + system_name + '.pickle'
  subset2scores = calculate_all_refbased(raw_src_sents, sys_sents, ref_sents, sent_annots,
                cache_file=cache_file, system_name=system_name)
  print_row(subset2scores, system_name=system_name)

  cache_file = 'cache_results_wmt22-cometkiwi-da/' + lp + '.' + system_name + '.pickle'
  calculate_all_qe([raw_src_sents, norm_src_sents], sys_sents, annots=sent_annots, cache_file=cache_file,
                     system_name=system_name)
  print_row(subset2scores, system_name=system_name)

  
sys = ['GPT4-5shot', 'ONLINE-B', 'ONLINE-M',  'ONLINE-G', 'ONLINE-W', 'ONLINE-Y',
       'NLLB_MBR_BLEU', 'NLLB_Greedy', 'Lan-BridgeMT', 'ZengHuiMT', 'GTCOM_Peter',
       'AIRC', 'CUNI-DocTransformer', 'PROMT']

for lp in ['en-cs', 'en-de', 'en-ru', 'en-uk']:
  print(r'\midrule')
  print(lp + r' \\')
  print(r'\midrule')
  for sys_name in sys:
    hyp_file = '../sys/' + lp + '/' + sys_name + '.' + lp + '.txt'
    if not os.path.exists(hyp_file):
      continue
    process_hyp(hyp_file, lp, system_name=hyp_file.split('/')[-1])
  
