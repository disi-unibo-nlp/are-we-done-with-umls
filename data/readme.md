
# For metamap_mapped:
## question - options - answer_idx - answer are string text
## concept - option1 - option2 - option3 - option4 are UMLS CUIs

medQA:
question - options - answer_idx - answer - concept - option1 - option2 - option3 - option4

medMCQA:

id - question - options - answer_idx - answer - concept - option1 - option2 - option3 - option4

pubmedQA:
pmids - question - options - answer - concept

# For spacy_mapped:
## cause most of the paper separeate each option to suffix with the question and transfer to a yes/no task, eg: 1. "How are you? Merda!" 2. "How are you? Brava!"
## here it keeps the same shape as those paper, so the rows are raw qa*option numbers
## sent - ans are string text
## qc - ac are UMLS CUIs

medQA:
sent - ans - qc - ac

medMCQA:

sent - ans - qc - ac

pubmedQA:
sent - ans - qc - ac


# UMLS
QA_cuis.txt
all test dataset's CUIs with there connected CUIs in full UMLS 2025AB

cui_detail of QA_cuis:
CUI, STR(string name), TUI(type), STY(semantic type), DEF_SAB(definition dictionary source), DEF(definition)

triplets of QA_cuis:
CUI1 | REL (relation type) | CUI2 | RELA (detail)

