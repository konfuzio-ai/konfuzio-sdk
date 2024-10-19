"""List all extra dependencies to be installed for Konfuzio SDK's AIs and dev mode."""

# Keep track with AI type needs which package in order to make bento builds as small as possible.
CATEGORIZATION_EXTRAS = [
    'torch',
    'torchvision',
    'transformers',
    'timm',
]

EXTRAS = {
    'dev': [
        'autodoc_pydantic==2.2.0',
        'coverage==7.3.2',
        'jupytext==1.16.4',
        'pytest',
        'pre-commit>=2.20.0',
        'parameterized>=0.8.1',
        'Sphinx',
        'sphinx-toolbox==3.4.0',
        'sphinx-reload==0.2.0',
        'sphinx-notfound-page==0.8',
        'm2r2==0.3.2',
        'nbval==0.10.0',
        'sphinx-sitemap==2.2.0',
        'sphinx-rtd-theme==1.0.0',
        'sphinxcontrib-mermaid==0.8.1',
        'sphinx-copybutton==0.5.2',
        'myst_nb',
        'ruff',
        'pytest-rerunfailures',
    ],
    'ai': list(
        set(
            [
                'accelerate',
                'chardet',
                'datasets',
                'evaluate',
                'spacy',
                'tensorflow-cpu',
                'mlflow',
            ]
            + CATEGORIZATION_EXTRAS
        )
    ),
}
