from sphinx.application import Sphinx
import nbformat
import re

def get_words_from_text(text):
    url_pattern = r'(https?|ftp)://[^\s/$.?#].[^\s]*'
    text_no_urls = re.sub(url_pattern, '', text)
    return re.findall(r'\b\w+\b', text_no_urls)

def validate_notebooks(app: Sphinx):
    notebook_path = 'docs/sdk/tutorials/data-preparation/index.ipynb'

    # Read the notebook
    with open(notebook_path, 'r') as notebook_file:
        lowercase_detected = False
        notebook_content = nbformat.read(notebook_file, as_version=4)
        keywords = ['Document', 'Documents', 'Project', 'Konfuzio']
        keywords_lower = [w.lower() for w in keywords]
        # Iterate through cells
        for cell in notebook_content.cells:
            if cell.cell_type == 'markdown':
                text = cell['source']
                words = get_words_from_text(''.join(text))                

                for word in words:
                    if word.lower() in keywords_lower:
                        if word[0].islower():
                            lowercase_detected = True
                            print(f"Error: '{word}' starts with a lowercase letter in a markdown cell.")
                            print("Source text:", text)
        if lowercase_detected:
            exit(1)

def setup(app: Sphinx):
    app.connect('builder-inited', validate_notebooks)
