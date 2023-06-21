"""List all extra dependencies to be installed for Konfuzio SDK's AIs and dev mode."""

EXTRAS = {
    'dev': [
        'cloudpickle==2.2.1',
        'flake8',
        'pydocstyle',
        'pytest',
        'pre-commit',
        'parameterized',
        'Sphinx==4.4.0',
        'sphinx-reload==0.2.0',
        'sphinx-notfound-page==0.8',
        'm2r2==0.3.2',
        'sphinx-sitemap==2.2.0',
        'sphinx-rtd-theme==1.0.0',
        'sphinxcontrib-mermaid==0.8.1',
        'matplotlib==3.7.1',
    ],
    'ai': [
        'chardet==5.1.0',
        'pydantic==1.10.8',  # pydantic is used by spacy. We need to force a higher pydantic version to avoid
        # https://github.com/tiangolo/fastapi/issues/5048
        'torch>=1.8',
        'torchvision>=0.9',
        'transformers>=4.21.2',
        'tensorflow-cpu==2.12.0',
        'timm==0.6.7',
        'scipy==1.10.1',
        'spacy>=2.3.5',
    ],
}
