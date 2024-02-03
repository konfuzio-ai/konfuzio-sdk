import os
import re

import nbformat
from sphinx.application import Sphinx


def get_words_from_text(text):
    """
    Extracts individual words from a given text while excluding URLs.

    Args:
        text (str): The input text containing words and possibly URLs.

    Returns:
        list: A list of individual words extracted from the text.

    This function utilizes regular expressions to first remove any URLs from the text,
    and then extracts individual words using word boundaries.

    Example:
        >>> text = "This is a sample sentence with a URL https://example.com and some punctuation."
        >>> words = get_words_from_text(text)
        >>> print(words)
        ['This', 'is', 'a', 'sample', 'sentence', 'with', 'and', 'some', 'punctuation']
    """
    url_pattern = r'(https?|ftp)://[^\s/$.?#].[^\s]*'
    text_no_urls = re.sub(url_pattern, '', text)
    return re.findall(r'\b\w+\b', text_no_urls)


def validate_notebooks(app: Sphinx):
    """
    Validates Jupyter notebooks by checking for lowercase starting words in markdown cells.

    Args:
        app (Sphinx): The Sphinx application.

    This function reads a Jupyter notebook, iterates through its markdown cells,
    and checks for lowercase starting words. If found, it prints an error message
    along with the source text.

    Example:
        >>> validate_notebooks(app)
    """
    base_path = 'docs/sdk/tutorials/'

    lowercase_detected = False

    for _, dirs, _ in os.walk(base_path):
        for directory in dirs:
            notebook_path = os.path.join(base_path, directory, 'index.ipynb')
            if not os.path.isfile(notebook_path):
                continue
            print('Validating: ', notebook_path)
            # Read the notebook
            with open(notebook_path, 'r') as notebook_file:
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
                                    print(
                                        f"Error in '{notebook_path}': '{word}' starts with a lowercase letter in a markdown cell."
                                    )
                                    print('Source text:', text)
    if lowercase_detected:
        exit(1)


def setup(app: Sphinx):
    app.connect('builder-inited', validate_notebooks)
