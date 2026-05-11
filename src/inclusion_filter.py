import os
import shutil
from tqdm import tqdm
from docling.document_converter import DocumentConverter
from flashtext import KeywordProcessor

def extract_text_with_docling(pdf_path, converter):
    """
    docling extract PDF text
    """
    try:
        result = converter.convert(pdf_path)
        text = result.document.export_to_markdown()
        return text
    except Exception as e:
        print(f"Docling failed {pdf_path}: {e}")
        return ""

def scan_pdfs_for_umls_with_docling(folder_path, 
                                    pdf_filter_in_folder, 
                                    pdf_filter_out_folder, 
                                    pdf_filter_pass_folder,
                                    umls_keywords):
    """
    match UMLS keywords
    """
    
    keyword_processor = KeywordProcessor(case_sensitive=False)
    for word in umls_keywords:
        keyword_processor.add_keyword(word)

    print("Initializing Docling model...")
    converter = DocumentConverter()

    results = {}
    
    pdf_files = [os.path.join(root, file) for root, _, files in os.walk(folder_path) 
                 for file in files if file.lower().endswith('.pdf')]
    
    for pdf_path in tqdm(pdf_files, desc="Scanning PDFs"):
        file = os.path.basename(pdf_path)
        print(f"Scanning: {file} ...")

        text = extract_text_with_docling(pdf_path, converter)
        if text:
            found_keywords = keyword_processor.extract_keywords(text)
            if found_keywords:
                results[file] = list(set(found_keywords))
                shutil.move(pdf_path, os.path.join(pdf_filter_in_folder, file))
            else:
                results[file] = [] 
                shutil.move(pdf_path, os.path.join(pdf_filter_out_folder, file))
        else:
            results[file] = [] 
            shutil.move(pdf_path, os.path.join(pdf_filter_pass_folder, file))
                        

if __name__ == "__main__":
    data_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data"
    )
    
    pdf_folder = os.path.join(data_root, "pdfs")
    os.makedirs(pdf_folder, exist_ok=True)
    pdf_filter_out_folder = os.path.join(data_root, "filtered_out")
    os.makedirs(pdf_filter_out_folder, exist_ok=True)
    pdf_filter_in_folder = os.path.join(data_root, "filtered_in")
    os.makedirs(pdf_filter_in_folder, exist_ok=True)
    pdf_filter_pass_folder = os.path.join(data_root, "filtered_pass")
    os.makedirs(pdf_filter_pass_folder, exist_ok=True)
    
    umls_terms = ['UMLS', 'umls', 
                  'unified medical language system', 
                  'Unified Medical Language System',
                  'Unified medical language system',
                  'UMLS Metathesaurus', 
                  'UMLS Semantic Network']
    
    match_results = scan_pdfs_for_umls_with_docling(pdf_folder, 
                                                    pdf_filter_in_folder, 
                                                    pdf_filter_out_folder, 
                                                    pdf_filter_pass_folder,
                                                    umls_terms)
    
    print("\n--- Finished ---")