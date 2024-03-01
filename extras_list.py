"""List all extra dependencies to be installed for Konfuzio SDK's AIs and dev mode."""

EXTRAS = {
    'dev': [
        'coverage==7.3.2',
        'pytest>=7.1.2',
        'pre-commit>=2.20.0',
        'parameterized>=0.8.1',
        'Sphinx==4.4.0',
        'sphinx-toolbox==3.4.0',
        'sphinx-reload==0.2.0',
        'sphinx-notfound-page==0.8',
        'm2r2==0.3.2',
        'sphinx-sitemap==2.2.0',
        'sphinx-rtd-theme==1.0.0',
        'sphinxcontrib-mermaid==0.8.1',
        'sphinx-copybutton==0.5.2',
        'myst_nb==0.17.2',
        'ruff',
    ],
    "ai": [
        "accelerate>=0.19.0",
        "chardet==5.1.0",
        "datasets==2.14.6",
        "evaluate==0.4.1",
        "pydantic==1.10.8",  # pydantic is used by spacy. We need to force a higher pydantic version to avoid
        # https://github.com/tiangolo/fastapi/issues/5048
        'spacy>=2.3.5',
        'torch>=1.8.1',
        'torchvision>=0.9.1',
        'transformers==4.36.0',
        'tensorflow-cpu==2.12.0',
        'timm==0.6.7',
        'mlflow==2.9.2',
    ],
}
