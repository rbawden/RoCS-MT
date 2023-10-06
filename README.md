# RoCS-MT (Robust Challenge Set for Machine Translation)

## News

- 06/10/2023: Version 1 release

## Description of files

### Source files (raw + normalised versions)
The source texts are available in several versions, depending on the sentence segmentation (manual or with spaCy) and on whether the sentences are in their raw or normalised version:
- manual sentence segmentation, raw text: `src/RoCS-MT.src.raw-manseg.en`
- manual sentence segmentation, normalised text: `src/RoCS-MT.src.raw-manseg.en`
- spaCy sentence segmentation, raw text: `src/RoCS-MT.src.raw-spacyseg.en`
- spaCy sentence segmentation, normalised text: `src/RoCS-MT.src.norm-spacyseg.en`

`.tsv` versions of the files are available in the same directory, indicating the document ids of each sentence.

### Reference files
The manually segmented, normalised texts were translated by professional translators into French, German, Czech, Ukrainian and Russian. They can be found in `ref/en-{fr,de,cs,uk,ru}`.

N.B. Some postedition still needs to be done for some of the translators due to variants being included (for gender variation).

### MT system translations
The `sys/` folder contains system outputs from the 2023 general MT task. They correspond to the translations of the concatenation of the four versions of the source texts.

