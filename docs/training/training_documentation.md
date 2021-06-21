.. meta::
   :description: Documentation of the features of the trainer package including the contents structure and examples of how to use it.

The functionalites of the Trainer module are not yet available in the SDK.

## LabelSectionModel Examples

![LabelSectionModel diagram](../_static/img/label_section_model.png)

A `LabelSectionModel` is a model that takes in a `Document` and predicts a `Label` per token and a `SectionLabel` for each line in the document.

### Training our first LabelSectionModel

A `LabelSectionModel` contains both a `LabelClassifier` and `SectionClassifier`. Both classifiers have set default modules and training is performed with a set of default hyperparameters. The `build` method returns metrics for each classifier, a dictionary of lists with the loss and accuracy values per batch for training/evaluation, which can be used e.g. visualization.

```python
from konfuzio.data import Project
from konfuzio.default_models import LabelSectionModel

# load the project
project = Project()

# create a default label section model from the project
model = LabelSectionModel(project)

# build (i.e. train) the label section model
label_classifier_metrics, section_classifier_metrics = model.build()

# save the trained label section model
model.save()
```

### Customizing the LabelSectionModel

We can also control the tokenization method and the hyperparameters of the `LabelSectionModel`. We change the tokenization from the default whitespace tokenization to BPE (byte-pair encoding) tokenization. The `LabelClassifier` now has a dropout of 0.25 and contains a 2-layer unidirectional LSTM module. The `SectionClassifier` now has a dropout of 0.5 and contains an NBOW module with a 64-dimensional embedding layer. Training both classifiers is still performed with the default training hyperparameters.

```python
from konfuzio.data import Project
from konfuzio.default_models import LabelSectionModel
from konfuzio.tokenizers import BPETokenizer

project = Project()

# specify a different tokenizer
tokenizer = BPETokenizer()

# configuration dict for the label classifier
label_classifier_config = {'dropout_rate': 0.25,
                           'text_module': {'name': 'lstm',
                                           'n_layers': 2,
                                           'bidirectional': False}}

# configuation dict for the section classifier
section_classifier_config = {'dropout_rate': 0.5,
                             'text_module': {'name': 'nbow',
                                             'emb_dim': 64,}}

# create label section model with chosen tokenizer and classifier configs
model = LabelSectionModel(project,
                          tokenizer=tokenizer,
                          label_classifier_config=label_classifier_config,
                          section_classifier_config=section_classifier_config)

# build the label section model with default training hyperparameters
model.build()

# save the trained label section model
model.save()
```

### Setting the LabelSectionModel training hyperparameters

We'll now use the default classifier hyperparameters but customize the training hyperparameters. An example of hyperparameters that can be changed: validation ratio, batch size, number of epochs, patience, optimizer, learning rate decay.

```python
from konfuzio.data import Project
from konfuzio.default_models import LabelSectionModel

# load the project
project = Project()

# create a default label section model from the project
model = LabelSectionModel(project)

label_training_config = {'valid_ratio': 0.15,  # what percentage of training data should be used to create validation data
                         'batch_size': 64,  # size of batches used for training
                         'seq_len': 100,  # number of sequential tokens to predict over
                         'n_epochs': 100,  # number of epochs to train for
                         'patience': 0,  # number of epochs without improvement in validation loss before we stop training
                         'optimizer': {'name': 'Adam', 'lr': 1e-5}}  # optimizer hyperparameters

section_training_config = {'valid_ratio': 0.2,
                           'batch_size': 128,
                           'max_length': 100,  # maximum tokens per line to consider
                           'n_epochs': 50,
                           'patience': 3,
                           'optimizer': {'name': 'RMSprop', 'lr': 1e-3, 'momentum': 0.9},
                           'lr_decay': 0.9}  # if validation loss does not improve, multiply learning rate by this value

# build model with training configs
model.build(label_training_config=label_training_config,
            section_training_config=section_training_config)

# save the trained label section model
model.save()
```

### Customizing the LabelSectionModel model and training hyperparameters

Customizing both the `LabelSectionModel` hyperparameters and the training hyperparameters. This example combines the two above examples.

```python
from konfuzio.data import Project
from konfuzio.default_models import LabelSectionModel
from konfuzio.tokenizers import BPETokenizer

# load the project
project = Project()

# specify a different tokenizer
tokenizer = BPETokenizer()

# configuration dict for the label classifier
label_classifier_config = {'dropout_rate': 0.25,
                           'text_module': {'name': 'lstm',
                                           'n_layers': 2,
                                           'bidirectional': False}}

# configuation dict for the section classifier
section_classifier_config = {'dropout_rate': 0.5,
                             'text_module': {'name': 'nbow',
                                             'emb_dim': 64,}}

# create label section model with chosen tokenizer and classifier configs
model = LabelSectionModel(project,
                          tokenizer=tokenizer,
                          label_classifier_config=label_classifier_config,
                          section_classifier_config=section_classifier_config)

label_training_config = {'valid_ratio': 0.15,  # what percentage of training data should be used to create validation data
                         'batch_size': 64,  # size of batches used for training
                         'seq_len': 100,  # number of sequential tokens to predict over
                         'n_epochs': 100,  # number of epochs to train for
                         'patience': 0,  # number of epochs without improvement in validation loss before we stop training
                         'optimizer': {'name': 'Adam', 'lr': 1e-5}}  # optimizer hyperparameters

section_training_config = {'valid_ratio': 0.2,
                           'batch_size': 128,
                           'max_length': 100,  # maximum tokens per line to consider
                           'n_epochs': 50,
                           'patience': 3,
                           'optimizer': {'name': 'RMSprop', 'lr': 1e-3, 'momentum': 0.9},
                           'lr_decay': 0.9}  # if validation loss does not improve, multiply learning rate by this value

# build model with training configs
model.build(label_training_config=label_training_config,
            section_training_config=section_training_config)

# save the trained label section model
model.save()
```

### Implementing a custom LabelSectionModel training loop

The `build` method of `LabelSectionModel` calls `self.build_label_classifier` and `self.build_section_classifier`. Both of which call a generic `self.fit_classifier` for both the label and section classifiers. If we want to customize the `fit_classifier` method - e.g. change the way the classifiers are trained based on some specific criteria, such as using custom loss function - then we can override the `fit_classifier` method. The custom `fit_classifier` method should take in the train, valid and test examples, the classifier, and any configuration arguments.

```python
from konfuzio.data import Project
from konfuzio.default_models import LabelSectionModel

class CustomerSpecificModel(LabelSectionModel):
    def fit_classifier(self, train_examples: DataLoader, valid_examples: DataLoader, test_examples: DataLoader, classifier: Classifier, **kwargs) -> Dict[str, List[float]]:
        # new code must take in train/valid/test examples, the classifier model, and any configuration arguments supplied as kwargs from the config dict
        # custom code goes here
        return metrics

# load the project
project = Project()

# can also use a custom tokenizer and model config here
model = CustomerSpecificModel(project)

# example custom configuration dicts
label_training_config = {'custom_loss_function_arg': 123}
section_training_config = {'custom_loss_function_arg': 999}

# build model with training configs
label_classifier_metrics, section_classifier_metrics = model.build(label_training_config,
                                                                   section_training_config)

# save the trained label section model
model.save()
```

### Implementing a custom LabelSectionModel classifier training loop

By default, both classifiers use the same generic `fit_classifier` function. If we want each to have their own custom `fit_classifier` function then we can do so by overwriting the `build_label_classifier`/`build_section_classifier` functions and implementing a custom `fit_label_classifier`/`fit_section_classifier` function within them. 

We can use existing functions to get the data iterators and then using our custom `fit_label_classifier`/`fit_section_classifier` functions in place of the generic `fit_classifier` function.

We could also customize the format of the training data by writing our own `get_label_classifier_iterators`/`get_section_classifier_iterators` functions. These must return a PyTorch `DataLoader` for the training, validation, and test sets, and must be compatible with the custom `fit_label_classifier`/`fit_section_classifier` functions.

Below is an example of how to implement our own `fit_label_classifier` function.

```python
from konfuzio.data import Project
from konfuzio.default_models import LabelSectionModel
from konfuzio.default_models.utils import get_label_classifier_iterators
from konfuzio.default_models.utils import get_section_classifier_iterators

class CustomerSpecificModel(LabelSectionModel):
    def fit_label_classifier(self, train_examples: DataLoader, valid_examples: DataLoader, test_examples: DataLoader, classifier: Classifier, **kwargs) -> Dict[str, List[float]]:
        # custom code goes here.
        return metrics

    def build_label_classifier(self, label_training_config: dict = {}) -> Dict[str, List[float]]:

        # get the iterators over examples for the label classifier
        examples = get_label_classifier_iterators(self.projects,
                                                  self.tokenizer,
                                                  self.text_vocab,
                                                  self.label_vocab,
                                                  **label_training_config)

        # unpack the examples
        train_examples, valid_examples, test_examples = examples

        # place label classifier on device
        self.label_classifier = self.label_classifier.to(self.device)

        # now uses our custom fit_label_classifier instead of generic fit_classifier
        label_classifier_metrics = self.fit_label_classifier(train_examples,
                                                             valid_examples,
                                                             test_examples,
                                                             self.label_classifier,
                                                             **label_training_config)

        # put label classifier back on cpu to free up GPU memory
        self.label_classifier = self.label_classifier.to('cpu')

        return label_classifier_metrics

# load the project
project = Project()

# define model with custom build_label_classifier function
model = CustomerSpecificModel(project)

# specify any custom training hyperparameters
label_training_config = {'custom_loss_function_arg': 123}

# build model with training configs
label_classifier_metrics, section_classifier_metrics = model.build(label_training_config,
                                                                   section_training_config)

# save the trained model
model.save()
```

## DocumentModel Examples

![DocumentModel Diagram](../_static/img/document_model.png)

A `DocumentModel` takes pages as input and predicts the "category" (the project ID) for that page. It can use both image features (from a .png image of the page) and text features (from the OCR text from the page).

### Training our first DocumentModel

A `DocumentModel` contains a `DocumentClassifier`. Similar to the `LabelSectionModel`, it has a set of default model hyperparameters and default training hyperparameters.

```python
from konfuzio.data import Project
from konfuzio.default_models import DocumentModel

# need to write `get_project` ourselves
projects = [get_project(project_id) for project_id in project_ids]

# create default document model from a list of projects
model = DocumentModel(projects)

# build (i.e. train) the document model
document_classifier_metrics = model.build()

# save the document model
model.save()
```

### Customizing the DocumentModel

Below we show how to implement a custom tokenizer, image preprocessing, image augmentation, and classifier. Image preprocessing is applied to images for training, evaluation, and inference. Image augmentation is only applied during training. We only need a `multimodal_module` when we use both text and image modules.

```python
from konfuzio.data import Project
from konfuzio.default_models import DocumentModel
from konfuzio.tokenizers import BPETokenizer

# need to write `get_project` ourselves
projects = [get_project(project_id) for project_id in project_ids]

# specify a custom tokenizer
tokenizer = BPETokenizer()

# specify how images should be preprocessed
image_preprocessing = {'target_size': (1000, 1000),
                       'grayscale': True}

# specify how images should be augmented during training
image_augmentation = {'rotate': 5}

# configuration dict for the document classifier
document_classifier_config = {'image_module': {'name': 'efficientnet_b0',
                                               'freeze': False},
                              'text_module': {'name': 'lstm',
                                              'n_layers': 2},
                              'multimodal_module': {'name': 'concatenate',
                                                    'hid_dim': 512}}

# create document model with chosen tokenizer and classifier configs
model = DocumentModel(projects,
                      tokenizer=tokenizer,
                      image_preprocessing=image_preprocessing,
                      image_augmentation=image_augmentation,
                      document_classifier_config=document_classifier_config)


# build (i.e. train) the document model
document_classifier_metrics = model.build()

# save the document model
model.save()
```

### Classifying documents using image features only

To use a `DocumentModel` that only uses the image of the document, simply do not include a `text_module` or `multimodal_module` in the classifier config. Passing a tokenizer when we have no `text_module` will throw an error, as there should be no text to tokenizer, so we make sure to pass `None` to the `tokenizer` argument like so:

```python
from konfuzio.data import Project
from konfuzio.default_models import DocumentModel

# need to write `get_project` ourselves
projects = [get_project(project_id) for project_id in project_ids]

# needs to ensure we do not use a tokenizer as we are not using a text_module
tokenizer = None

# specify how images should be preprocessed
image_preprocessing = {'target_size': (1000, 1000),
                       'grayscale': True}

# specify how images should be augmented during training
image_augmentation = {'rotate': 5}

# configuration dict for the document classifier
# no text_module AND no multimodal_module
document_classifier_config = {'image_module': {'name': 'efficientnet_b0',
                                               'freeze': False}}

# create document model with chosen tokenizer and classifier configs
model = DocumentModel(projects,
                      tokenizer=tokenizer,
                      image_preprocessing=image_preprocessing,
                      image_augmentation=image_augmentation,
                      document_classifier_config=document_classifier_config)

# build (i.e. train) the document model
document_classifier_metrics = model.build()

# save the document model
model.save()
```

### Classifying documents using text features only

To use a `DocumentModel` that only uses the image of the document, simply do not include an `image_module` or `multimodal_module` in the classifier config. Passing an `image_preprocessing` or `image_augmentation` argument when we have no `image_module` will throw an error so we need to ensure we pass `None` to the `image_preprocessing` and `image_augmentation` arguments like so:

```python
from konfuzio.data import Project
from konfuzio.default_models import DocumentModel
from konfuzio.tokenizers import BPETokenizer

# need to write `get_project` ourselves
projects = [get_project(project_id) for project_id in project_ids]

tokenizer = BPETokenizer()

# both should be None when not using an image_module
image_preprocessing = None
image_augmentation = None

# configuration dict for the document classifier
# no image_module AND no multimodal_module
document_classifier_config = {'text_module': {'name': 'lstm',
                                              'n_layers': 2}}

# create document model with chosen tokenizer and classifier configs
model = DocumentModel(projects,
                      tokenizer=tokenizer,
                      image_preprocessing=image_preprocessing,
                      image_augmentation=image_augmentation,
                      document_classifier_config=document_classifier_config)

# build (i.e. train) the document model
document_classifier_metrics = model.build()

# save the document model
model.save()
```

### Setting the DocumentModel training hyperparameters

Similar to the `LabelSectionModel` we can customize the training config which will work with ANY classifier/module combination:

```python
from konfuzio.data import Project
from konfuzio.default_models import DocumentModel

# need to write `get_project` ourselves
projects = [get_project(project_id) for project_id in project_ids]

# create a default document model
model = DocumentModel(projects)

# define the custom training hyperparameters
document_training_config = {'valid_ratio': 0.2,
                            'batch_size': 128,
                            'max_length': 100,  # maximum tokens per page to consider, will do nothing if no text_module used
                            'n_epochs': 50,
                            'patience': 3,
                            'optimizer': {'name': 'RMSprop', 'lr': 1e-3, 'momentum': 0.9},
                            'lr_decay': 0.9}

# build (i.e. train) the document model with custom training hyperparameters
document_classifier_metrics = model.build(document_training_config=document_training_config)

# save the document model
model.save()
```

### Implementing a custom DocumentModel training loop

We can also override the `fit_classifier` method to define our method of training the document classifier. We can do a similar thing with overwriting `build` to define our own custom data processing.

```python
from konfuzio.data import Project
from konfuzio.default_models import DocumentModel

class CustomerSpecificModel(DocumentModel):
    def fit_classifier(self, train_examples: DataLoader, valid_examples: DataLoader, test_examples: DataLoader, classifier: Classifier, **kwargs) -> Dict[str, float]:
        # new code must take in train/valid/test examples, the classifier model, and kwargs from the config dict
        # custom code goes here
        return metrics

# need to write `get_project` ourselves
projects = [get_project(project_id) for project_id in project_ids]

# can also use a custom tokenizer and model config here
model = CustomerSpecificModel(project)

# custom fit_classifier config
custom_document_training_config = {'custom_loss_function_arg': 123}

# train document model with custom fit_classifier function
custom_document_classifier_metrics = model.build(document_training_config=custom_document_training_config)

# save a trained model
model.save()
```

## Design Philosophy

- A "model" encapsulates everything we need for training on a labeled dataset and then performing extraction (inference) on some real data.
- Models contain "classifiers", one for each task the model is performing. The `LabelSectionModel` has a label classifier for POS tagging of tokens within a document and a section classifier for labeling sections of a document. The `DocumentModel` has a document classifier for predicting the category of each page within a document. Each classifier is a PyTorch `nn.Module`.
- Each classifier is made up of "modules". These are the main core of the classifier and are usually some kind of neural network architecture. Modules are split into three different categories: image modules (e.g. VGG and EfficientNet), text (e.g. NBOW, LSTM, and Transformer), and multimodal (e.g. concatenation of image and text features). Each module is also a PyTorch `nn.Module`.
- Modules can contain other modules that can contain other modules - it's modules all the way down!
- The modules are designed to be agnostic to the classifier they are within, with the actual task-specific layers are contained in the classifier itself. Every text module takes in a sequence of tokens and outputs a sequence of tensors. Every image module takes in an image and outputs a tensor of features.
  - Let's look at an example of a label classifier containing an LSTM module. The classifier takes in text, feeds it to the LSTM module which then calculates a hidden state per token within the text, and then returns this sequence of hidden states to the classifier which passes them through a linear layer to re-size them to the desired output dimensionality.
  - For a section classifier with an LSTM module: the classifier takes in the text, feeds it to the LSTM module which then returns the hidden states to the classifier, the classifier pools the hidden states and then passes them through a linear layer to make a prediction.
  - The document classifier with only a text module is the same as the section classifier. A document classifier with both a text and image module will calculate the text features and the image features, then pass both to a multimodal module that combines the two, performs some calculations to get multimodal features, and then passes these multimodal features to the classifier which makes a prediction.

Models are the glue holding everything together in a nice package. The most important model attributes are:

- `tokenizer` - a tokenizer that tokenizes text by converting a string to a list of strings.
- Vocabularies, which contain a mapping between a token (string) to an index (int), and vice versa. The types of vocabularies are:
  - `text_vocab` - the vocabulary for the document text
  - `label_vocab` - the vocabulary for the annotation labels (only if using a label classifier)
  - `section_vocab` - the vocabulary for the section labels (only if using a section classifier)
  - `category_vocab` - the vocabulary for the document categories (only if using a document classifier)
- Classifiers, which are specific to the desired task. Each classifier contains module(s). The types of classifiers are:
  - `label_classifier` - an instance of a `LabelClassifier` used to predict the annotation label for each token (only used in a `LabelSectionModel`)
  - `section_classifier` - an instance of a `SectionClassifier` used to predict the section label for each line in the document (only in a `LabelSectionModel`)
  - `document_classifier` - used the predict the category of each page in the document (only in `DocumentModel`) and can be one of three types:
    - a `DocumentTextClassifier` which classifies a document's page using only the text on that page
    - a `DocumentImageClassifier` which classifies a document's page using only an image of the page
    - a `DocumentMultiModalClassifier` which classifiers a document's page using both the text and image of that page
- Configuration dictionaries, with one configuration dictionary per classifier used in the model, e.g. `label_classifier_config`. They are used to define the hyperparameters of the classifier and also used to load the correct classifier when loading the model. Training configuration dictionaries are not saved as a model attribute as the model should operate independently on how it is trained.
- `image_preprocessing` - a dictionary that states how images should be pre-processed before being classified (only in DocumentModel)
- `image_augmentation` - a dictionary that states how images should be augmented during the training of the classifier (only in DocumentModel)

## Tokenizers

A tokenizer is a function that defines how a string should be separated into tokens (a list of strings), e.g.:

```python
tokenizer.get_tokens('hello world') -> ['hello', 'world']
```

The `konfuzio.tokenizers` module contains a few tokenizers which can either be directly imported, i.e. `from konfuzio.tokenizers import WhitespaceTokenizer` or obtained using the `get_tokenizer` function, i.e.:

```python
from konfuzio.tokenizers import get_tokenizer
tokenizer = get_tokenizer('whitespace')  # gets a WhitespaceTokenizer
from konfuzio.tokenizers import WhitespaceTokenizer
tokenizer = WhitespaceTokenzer()  # same as above
```

All tokenizers have the following methods:

- `get_tokens`, takes in a string and returns a list of tokens (strings)
- `get_entities`, same as `get_tokens` but also contains the start and end character offsets for each token (represented as a list of dicts). This is usually slower than `get_tokens`, so should only be used if we explicitly need the character offsets.
- `get_annotations`, same as `get_entities` but converts each entity to an `Annotation` object and returns a list of annotations.

Currently available tokenizers:

### WhitespaceTokenizer

```python
from konfuzio.tokenizers import WhitespaceTokenizer
from konfuzio.tokenizers import get_tokenizer
tokenizer = get_tokenizer('whitespace')
```

Uses regular expressions to split a string based on whitespace. Very fast but naive method of tokenization.

### SpacyTokenizer

```python
from konfuzio.tokenizers import SpacyTokenizer
from konfuzio.tokenizers import get_tokenizer
tokenizer = get_tokenizer('spacy')
```

Tokenizes using the `de_core_news_sm` spaCy model. Relatively slow.

### PhraseMatcherTokenizer

```python
from konfuzio.data import Project
from konfuzio.tokenizers import PhraseMatcherTokenizer
from konfuzio.tokenizers import get_tokenizer
project = Project()
tokenizer = get_tokenizer('phrasematcher', project)
```

Note, this tokenizer also has to take in a `Project`, or list of `Project`, to build the `PhraseMatcher`.

This builds a spaCy `de_core_news_sm` phrase matcher using the annotation labels for each project and then uses this learned matching to tokenize data. We can think of this as learning a simple regex pattern matcher from the data. This is relatively slow, especially when the dataset is large.

### BPETokenizer

```python
from konfuzio.data import Project
from konfuzio.tokenizers import BPETokenizer
from konfuzio.tokenizers import get_tokenizer
tokenizer = get_tokenizer('bert-base-german-cased')
```

Gets a pre-trained byte-pair encoding tokenizer from the HuggingFace Transformers library. Officially we support four different variants of the `BPETokenizer` (other variants should work, however they are not tested):

```python
tokenizer = BPETokenizer('bert-base-german-cased')
tokenizer = BPETokenizer('bert-base-german-dbmdz-cased')
tokenizer = BPETokenizer('bert-base-german-dbmdz-uncased')
tokenizer = BPETokenizer('distilbert-base-german-cased')
```

By default, `BPETokenizer` gets the `bert-base-german-cased` variant.

```python
# both of these get the same tokenizer
tokenizer = BPETokenizer()
tokenizer = BPETokenizer('bert-base-german-cased')
```

These tokenizers are special as they have their own custom vocabulary - accessed via `tokenizer.vocab` - which is because they are designed to be used with a pre-trained Transformer model that must use the vocabulary it was trained with. Initializing a model with a `BPETokenizer` automatically sets the `text_vocab` to the `tokenizer.vocab` when using a BPE tokenizer. These tokenizers still perform very well with non-Transformer models and are also the fastest tokenizers.

## Vocabularies

A vocabulary is a mapping between tokens and integers.

A vocab is usually initialized with `collections.Counter` - a dictionary where the keys are the tokens and the values are how many times that token appears in the training set. It can also be initialized with a `dict` that has the keys being the tokens and the values being the integer representations, this is used to create a `Vocab` object from an existing vocabulary already represented as a dictionary.

Two vocab arguments are `max_size` and `min_freq`. `max_size` of 30,000 means that only the most common 30,000 tokens are used to create the vocabulary. `min_freq` of 2 means that only tokens that appear at least twice are used to create the vocabulary.

Two other arguments are `unk_token` and `pad_token`. When trying to convert a token to an integer and the token is NOT in the vocabulary then this token is replaced by an `unk_token`. If the `unk_token` is set to `None` then the vocabulary will throw an error when it tries to convert a token that is not in the vocabulary to an integer - this is usually used when created a vocabulary over the labels. A `pad_token` is a token we will use for padding sequences, effectively a no-op, and can also be `None`. If the `unk_token` and `pad_tokens` are not `None` then the vocab will also have a `unk_idx` and `pad_idx` attribute which gets the integer value of the `unk_token` and `pad_token` - this is more for convenience than anything else.

The final argument is `special_tokens`, a list of tokens that are guaranteed to appear in the vocabulary.

To convert from a token to an integer, use the `stoi` (string to int) method, i.e. `vocab.stoi('hello')`. To convert from an integer to a token, use the `itos` (int to string) method, i.e. `vocab.itos(123)`.

We can get the list of all the tokens within the vocab with `vocab.get_tokens`, and a list of all integers with `vocab.get_indexes`.

The `LabelSectionModel` and `DocumentModel` will create a `text_vocab` with the provided tokenizer unless:

- a `text_vocab` is provided, at which point the model will use that vocab and not create one from the data. This is usually done when loading a trained model.
- the tokenizer used has a `vocab`, where we also use that vocab and do not create one. Usually, the case when using a BPE tokenizer.

For the `label_vocab`, `section_vocab`, and `category_vocab`, one is created from the data unless one is provided. Again, a provided vocab usually means we are loading a saved model.

## Text Modules

There are currently four text modules available. Each module takes a sequence of tokens as input and outputs a sequence of "hidden states", i.e. one vector per input token. The size of each of the hidden states can be found with the module's `n_features` parameter.

### NBOW

The neural bag-of-words (NBOW) model is the simplest of models, it simply passes each token through an embedding layer. As shown in the [fastText](https://arxiv.org/abs/1607.01759) paper this model is still able to achieve comparable performance to some deep learning models whilst being considerably faster.

One downside of this model is that tokens are embedded without regards to the surrounding context in which they appear, e.g. the embedding for "May" in the two sentences "May I speak to you?" and "I am leaving on the 1st of May" are identical, even though they have different semantics.

Important arguments:

- `emb_dim` the dimensions of the embedding vector
- `dropout_rate` the amount of dropout applied to the embedding vectors

### NBOWSelfAttention

This is an NBOW model with a multi-headed self-attention layer, detailed [here](https://arxiv.org/abs/1706.03762), added after the embedding layer. This effectively contextualizes the output as now each hidden state is now calculated from the embedding vector of a token and the embedding vector of all other tokens within the sequence.

Important arguments:

- `emb_dim` the dimensions of the embedding vector
- `dropout_rate` the amount of dropout applied to the embedding vectors
- `n_heads` the number of attention heads to use in the multi-headed self-attention layer. Note that `n_heads` must be a factor of `emb_dim`, i.e. `emb_dim % n_heads == 0`.

### LSTM

The [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) (long short-term memory) is a variant of a [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network) (recurrent neural network). It feeds the input tokens through an embedding layer and then processes them sequentially with the LSTM, outputting a hidden state for each token. If the LSTM is bi-directional then it trains a forward and backward LSTM per layer and concatenates the forward and backward hidden states for each token.

Important arguments:

- `emb_dim` the dimensions of the embedding vector
- `hid_dim` the dimensions of the hidden states
- `n_layers` how many LSTM layers to use
- `bidirectional` if the LSTM should be bidirectional
- `dropout_rate` the amount of dropout applied to the embedding vectors and between LSTM layers if `n_layers > 1`

### BERT

[BERT](https://arxiv.org/abs/1810.04805) (bi-directional encoder representations from Transformers) is a family of large [Transformer](https://arxiv.org/abs/1706.03762) models. The available BERT variants are all pre-trained models provided by the [transformers library](https://github.com/huggingface/transformers). It is usually infeasible to train a BERT model from scratch due to the significant amount of computation required. However, the pre-trained models can be easily fine-tuned on desired data.

Important arguments:

- `name` the name of the pre-trained BERT variant to use
- `freeze` should the BERT model be frozen, i.e. the pre-trained parameters are not updated

The BERT variants, i.e. `name` arguments, that are covered by internal tests are:

- `'bert-base-german-cased'`
- `'bert-base-german-dbmdz-cased'`
- `'bert-base-german-dbmdz-uncased'`
- `'distilbert-base-german-cased'`

In theory, all variants beginning with `bert-base-*` and `distilbert-*` should work out of the box. Other BERT variants come with no guarantees.

## Image Modules

We currently have two image modules available, each have several variants. The image models each have their classification heads removed and generally, they return the output of the final pooling layer within the model which has been flattened to a `[batch_size, n_features]` tensor, where `n_features` is an attribute of the model.

### VGG

The [VGG](https://arxiv.org/abs/1409.1556) family of models are image classification models designed for the [ImageNet](http://www.image-net.org/). They are usually used as a baseline in image classification tasks, however are considerably larger - in terms of the number of parameters - than modern architectures.

Important arguments:

- `name` the name of the VGG variant to use
- `pretrained` if pre-trained weights for the VGG variant should be used
- `freeze` if the parameters of the VGG variant should be frozen

Available variants are: `vgg11`, `vgg13`, `vgg16`, `vgg19`, `vgg11_bn`, `vgg13_bn`, `vgg16_bn`, `vgg19_bn`. The number generally indicates the number of layers in the model, higher does not always mean better. The `_bn` suffix means that the VGG model uses Batch Normalization layers, this generally leads to better results.

The pre-trained weights are taken from the [torchvision](https://github.com/pytorch/vision) library and are weights from a model that has been trained as an image classifier on ImageNet. Ideally, this means the images should be 3-channel color images that are at least 224x224 pixels and should be normalized with:

```python
from torchvision import transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
```

### EfficientNet

[EfficientNet](https://arxiv.org/abs/1905.11946) is a family of convolutional neural network based models that are designed to be more efficient - in terms of the number of parameters and FLOPS - than previous computer vision models whilst maintaining equivalent image classification performance.

Important arguments:

- `name` the name of the EfficientNet variant to use
- `pretrained` if pre-trained weights for the EfficientNet variant should be used
- `freeze` if the parameters of the EfficientNet variant should be frozen

Available variants are: `efficientnet_b0`, `efficientnet_b1`, ..., `efficienet_b7`. With `b0` having the least amount of parameters and `b7` having the most.

The pre-trained weights are taken from the [timm](https://github.com/rwightman/pytorch-image-models) library and have been trained on ImageNet, thus the same tips, i.e. normalization, that apply to the VGG models also apply here.

## Loading Pre-trained Modules

All modules - not classifiers - can load parameters from a saved state using the `load` argument, which can either be a path to a saved PyTorch `nn.Module` `state_dict` or the `state_dict` itself, e.g.:

```python
from konfuzio.data import Project
from konfuzio.default_models import LabelSectionModel

# load a project
project = Project()

# load the label classifier text module parameters from a given path
label_classifier_config = {'dropout_rate': 0.25,
                           'text_module': {'name': 'lstm',
                                           'n_layers': 2,
                                           'bidirectional': False,
                                           'load': 'saved_modules/label_text_module.pt'}}

# load a saved state dict from a path
section_text_module_state_dict = torch.load('saved_modules/section_text_module.pt')

# load the section classifier text module parameters from a state_dict directly
section_classifier_config = {'dropout_rate': 0.5,
                             'text_module': {'name': 'nbow',
                                             'emb_dim': 64,
                                             'load': section_text_module_state_dict}}

# create label section model with classifiers that contain pre-trained text modules
model = LabelSectionModel(project,
                          tokenizer=tokenizer,
                          label_classifier_config=label_classifier_config,
                          section_classifier_config=section_classifier_config)
```

## Extraction

Each model has an `extract` method that gets the predictions from that model.

## OCR 

The ability to do OCR tasks in bundled into the FileScanner class. The FileScanner supports multuple OCR solution and takes text embeddings into account.

The following example runs OCR on a PDF with the default settings.
```python
from konfuzio.ocr import FileScanner

path = 'example.pdf'  # Path to a pdf or image file

with FileScanner(path) as f:
  document_text: str = f.ocr()
```

In a first step the FileScanner checks if the file has some text embeddings and whether if its likely that the detected text embeddings cover the whole document. This is done by checking the frequency of specific characters like 'e', the ratio  of ASCII characters, and the amount of character on the pages and the overall document.

If its likely that some characters are missing in the embeddings, the OCR process is started. The OCR text is then returned, except if the amount of OCR characters is less then the amount of text embeddings characters, in this case the text embeddings are used.

The default OCR process is based on tesseract with presets for images and scans. In case the document contains some text embeddings the scan preset is always used. If there are no text embeddings present the FileScanner uses a blurryness score to decide which preset should be used.

### OCR with the Azure Read API

In order to use the Azure Read API you need to set the credentials of an appropriate azure accunt via environment variables or the .env file.

```text
AZURE_OCR_BASE_URL = https://****.api.cognitive.microsoft.com
AZURE_OCR_KEY = **********************
```

```python
from konfuzio.ocr import FileScanner

path = 'example.pdf'  # Path to a pdf or image file

with FileScanner(path, ocr_method='read_v3_fast') as f:
  document_text: str = f.ocr()
```
The way text embeddings are used does not differ from the default OCR, however no blurriness score is calculated for the Azure Read API.

The Azure Read API has some limititation regarding file size, page numbers, and rate limits. These lmites are updated over time and can be found here: https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/concept-recognizing-text

### Additional FileScanner OCR results

The FileScanner provides the ocr results in text, bounding bboxes and sandwich PDF (PDF with text embeddings).
```python
from konfuzio.ocr import FileScanner

with FileScanner('example.jpeg') as v:
  f_scanner.ocr()

f_scanner.text  # str: String representation of the document or image
f_scanner.bbox  # dict: Bounding boxes on a character level
f_scanner.sandwich_file  # BytesIO: When using Azure you need to pass 'read_v3' as ocr_method to get the sandwich file.
f_scanner.is_blurry_image  # boolean: Whether the image was blurry (only set for default OCR)
f_scanner.used_ocr_method  # str: the OCR method used.
```

### Further usages of the FileScanner

`use_text_embedding_only`: In order to rely on text embeddings only, you can pass `use_text_embedding_only=True` to the ocr() method call.

`file`: A file like objects can be used to initialize the FileScanner (instead of the `path` argument.)

### LabelSectionModel extraction

```python
from konfuzio.data import Project
from konfuzio.default_models import LabelSectionModel

# load the project
project = Project()

# create a default label section model from the project
model = LabelSectionModel(project)

# build (i.e. train) the label section model
label_classifier_metrics, section_classifier_metrics = model.build()

# save the label section model
model.save('saved_label_section_model.pt')

# ... later on in another file

from konfuzio.default_models import load_label_section_model

# load the saved section label model
model = load_label_section_model('saved_label_section_model.pt')

pdf_texts: List[str] = [pdf1_text, pdf2_text, ...]  # the text of each pdf, extracted via ocr

# list of extraction results for each document
results = [model.extract(pdf_text) for pdf_text in pdf_texts]
```

Each element of results is a `Dict[str, Union[List[Dict[str, pd.DataFrame]], pd.DataFrame]]`, where the keys are the label and section names.

If the key is a label then the value is a DataFrame with columns `Label`, `Accuracy`, `Candidate`, `Translated Candidate`, `Start`, and `End`. `Label` is the name of the label, `Accuracy` is the confidence, `Candidate` and `Translated Candidate` are the actual token string, `Start`, and `End` are the start and end offsets (number of characters from the beginning of a document).

If the key is a section, the value is a list, one element for each detected instance of that section. Each element is a dictionary with the same format as the `results` dictionary - i.e. keys are either labels or sections, values are DataFrames or list of dictionaries - with information about the labels and sections within that section. This recursive format allows nested sections.

### DocumentModel extraction

```python
from konfuzio.data import Project
from konfuzio.default_models import DocumentModel

# need to write `get_project` ourselves
projects = [get_project(project_id) for project_id in project_ids]

# create default document model from a list of projects
model = DocumentModel(projects)

# build (i.e. train) the document model
document_classifier_metrics = model.build()

# save the document model
model.save('saved_document_model.pt')

# ... later on in another file

from konfuzio.default_models import load_document_model

# load the saved document model
model = load_document_model('saved_document_model.pt')

pdf_paths: List[str] = ['data/1.pdf', 'data/2.pdf', ...]  # path to the pdf file
pdf_texts: List[str] = [pdf1_text, pdf2_text, ...]  # the text of each pdf, extracted via ocr

# list of extraction results for each document
results = [model.extract(pdf_path, pdf_text) for (pdf_path, pdf_text) in zip(pdf_paths, pdf_texts)]
```

Each element of results is a `Tuple[str, float], pandas.DataFrame`. The first element of the tuple is the predicted label as a string, the second element is the confidence of that prediction, i.e. `('insurance_contract', 0.6)`. The DataFrame has a `category` and a `confidence` column. `category` is the predicted label as a string for each class and `confidence` is the confidence of each of the predictions.

Note: when a pdf has multiple pages the `DocumentModel` makes a prediction on each page individually and then averages the predictions together.
