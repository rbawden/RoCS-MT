#!/usr/bin/python
from calculate_metric_scores import *
import os

def process_hyp(hyp_file, lp, system_name='System'):
  trg = lp.split('-')[-1]
  src_sents = read_file('../src/RoCS-MT.src.raw-manseg.en')
  sent_annots = read_annots_file('../ref/RoCS-annotated.tsv')
  sys_sents = read_file(hyp_file)
  sys_sent_docs = read_file(re.sub('.txt', '.ids.txt', hyp_file))
  sys_sents = [x for i, x in enumerate(sys_sents) if 'rm' in sys_sent_docs[i]]
  if not os.path.exists('../ref/RoCS-MT.ref.' + trg):
    return
  ref_sents = read_file('../ref/RoCS-MT.ref.' + trg, ref=True)

  #import pdb; pdb.set_trace()
  
  cache_file = 'cache_results/' + lp + '.' + system_name + '.pickle'
  
  subset2scores = calculate_all(src_sents, sys_sents, ref_sents, sent_annots, comet_too=True,
                cache_file=cache_file, system_name=system_name)
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
  
