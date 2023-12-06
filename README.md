# RoCS-MT (Robust Challenge Set for Machine Translation)

<img align="right" width="220" src="https://github.com/rbawden/RoCS-MT/blob/d1913e9534035772d3cae54d460bda65494643ba/img/rocs-mt-logo.png">

RoCS-MT, a Robust Challenge Set for Machine Translation (MT), is designed to test MT systems’ ability to translate user-generated content (UGC) that displays non-standard characteristics, such as spelling errors, devowelling, acronymisation, etc. RoCS-MT is composed of English comments from Reddit, selected for their non-standard nature, which have been manually normalised and professionally translated into five languages: French, German, Czech, Ukrainian and Russian. The challenge set was included as a test suite at WMT 2023. This repository therefore also includes automatic translations from the submissions to the general MT task. 

## News

- 06/10/2023: Version 1 release: this initial release correpsonds to the first version of the challenge set at WMT 2023. Scripts for analysis and reproducing results will be made available shortly after the conference!

## Citation

Please cite the following article:

Rachel Bawden and Benoît Sagot. 2023. [RoCS-MT: Robustness Challenge Set for Machine Translation](https://aclanthology.org/2023.wmt-1.21/). In Proceedings of the Eighth Conference on Machine Translation, pages 198–216, Singapore. Association for Computational Linguistics.

```
@inproceedings{bawden-sagot-2023-rocs,
    title = "{R}o{CS}-{MT}: Robustness Challenge Set for Machine Translation",
    author = "Bawden, Rachel  and
      Sagot, Beno{\^\i}t",
    editor = "Koehn, Philipp  and
      Haddon, Barry  and
      Kocmi, Tom  and
      Monz, Christof",
    booktitle = "Proceedings of the Eighth Conference on Machine Translation",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.wmt-1.21",
    pages = "198--216"
}
```


## Description of files

Each version of the challenge set (only v1 published for now) contains the following files:

### Source files (raw + normalised versions)
The source texts are available in several versions, depending on the sentence segmentation (manual or with spaCy) and on whether the sentences are in their raw or normalised version:
- manual sentence segmentation, raw text: `src/RoCS-MT.src.raw-manseg.en`
- manual sentence segmentation, normalised text: `src/RoCS-MT.src.raw-manseg.en`
- spaCy sentence segmentation, raw text: `src/RoCS-MT.src.raw-spacyseg.en`
- spaCy sentence segmentation, normalised text: `src/RoCS-MT.src.norm-spacyseg.en`

`.tsv` versions of the files are available in the same directory, indicating the document ids of each sentence.

### Reference files
The manually segmented, normalised texts were translated by professional translators into French, German, Czech, Ukrainian and Russian. They can be found in `ref/en-{fr,de,cs,uk,ru}`. We additionally include manual annotations of normalisation phenomena in `ref/RoCS-annotated.tsv`.

N.B. Some postedition still needs to be done for some of the translators due to variants being included (for gender variation).

### MT system translations
The `sys/` folder contains system outputs from the 2023 general MT task. They correspond to the translations of the concatenation of the four versions of the source texts.

### Guidelines

We include the first version of the guidelines for both normalisation and translation in `guidelines`. Both guides, in particular the one for normalisation, is likely to evolve in future releases.
