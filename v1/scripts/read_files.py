#!/usr/bin/python
import re
import csv

def clean_quotes(text):
    if '"' in text:
        text = re.sub('(^\"|\"$)', '', text).replace('""', '"')
    return text

def read_annots_file(annots_file):
    data = []
    headers = None
    last_sentid = None
    with open(annots_file) as fp:
        for line in fp:
            if headers is None:
                headers = line.strip('\n').split('\t')
            else:
                if line.split('\t')[0] == '':
                    continue
                sentid = int(line.split('\t')[1])
                if sentid != last_sentid:
                    data.append([])
                data[-1].append({headers[i]: clean_quotes(x) for i, x in enumerate(line.strip('\n').split('\t')[:len(headers)])})
                last_sentid = sentid
    return data

def read_sources(source_folder):
    subset2docid2sents = {}
    for filename in ['RoCS-MT.src.norm-manseg+docid.en.tsv', 'RoCS-MT.src.norm-spacyseg+docid.en.tsv',
                    'RoCS-MT.src.raw-manseg+docid.en.tsv', 'RoCS-MT.src.raw-spacyseg+docid.en.tsv']:
        with open(source_folder + '/' + filename) as fp:
            subset = ''.join([x[0] for x in re.match('RoCS-MT\.src.(.+?)\.en.tsv', filename).group(1).split('-')])
            subset2docid2sents[subset] = {}
            for line in fp:
                docid, sent = line.strip('\n').split('\t')
                docnum = int(docid.split('-')[-1])
                if docnum not in subset2docid2sents[subset]:
                    subset2docid2sents[subset][docnum] = []
                subset2docid2sents[subset][docnum].append(sent.strip())
    return subset2docid2sents

def read_sys_sents(sents_file, idx_file):
    subset2docid2sents = {}
    with open(sents_file) as sfp, open(idx_file) as ifp:
        for i, (sent, subset_docid) in enumerate(zip(sfp, ifp)):
            subset, docid = subset_docid.strip().split('-')
            docid = int(docid)
            if subset not in subset2docid2sents:
                subset2docid2sents[subset] = {}
            if docid not in subset2docid2sents[subset]:
                subset2docid2sents[subset][docid] = []
            subset2docid2sents[subset][docid].append(sent.strip())
    return subset2docid2sents

def read_refs(ref_file):
    ref_data = {}
    headers = None
    with open(ref_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        last_segid = -1
        for row in reader:
            if headers is None:
                headers = row
                continue
            if len(row) < 1:
                continue
            docid = int(row[1])
            assert int(row[0]) > last_segid, 'The segment ids must be in numerical order with no breaks: ' + filename
            if docid not in ref_data:
                ref_data[docid] = []
            ref_data[docid].append(row[8].strip())
            last_segid += 1
    return ref_data
