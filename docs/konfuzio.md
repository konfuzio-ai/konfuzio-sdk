<meta name="description" content="Technical documentation of Konfuzio.">

# Konfuzio

Our **API Documentation** is available online via https://app.konfuzio.com/api. Have a look at our [YouTube API Tutorial](https://www.youtube.com/watch?v=NZKUrKyFVA8) too.

The Konfuzio Software Development Kit, the **Konfuzio SDK**, can be installed via [pip install konfuzio-sdk](https://pypi.org/project/konfuzio-sdk/). Find examples in Python and review the source code on [GitHub](https://github.com/konfuzio-ai/document-ai-python-sdk).

In addition, enterprise clients do have access to our [Python Konfuzio Trainer Module](./training_documentation.html) to define, train and run custom Document AI.

Download the **On-Prem and Private Cloud** documentation to deploy Konfuzio on a single VM or on your cluster. Use [this link](./_static/pdf/konfuzio_on_prem.pdf) to access the latest version.

The [Changelog of our Server](./changelog_app.html) provides you with insights about any (future) release.

**This technical documentation is updated frequently. Please feel free to share your feedback via info@konfuzio.com**.

Konfuzio is a cloud and on-premises B2B platform used thousands of times a day to train and run AI.
SMEs and large companies train their AI to understand and process documents, e-mails and texts like human beings.
Find out more on our [Homepage](https://konfuzio.com).

## What does Konfuzio do under the hood?

In the following section we try to give an answer to the question "How can you, as a technical user, describe Konfuzio to a non-technical user?"

The majority of documents consist of unstructured data, usually in the form of a PDF file. This data usually consists of multiple modalities -- text, images, tables. Konfuzio applies a five step process to automatically detect structures within the unstructured data. Automatic detection of these structures greatly reduces human labor in the automated business process to a minimum. Our algorithms use state-of-the-art deep learning and transfer learning techniques in order to learn to detect structures with only a handful of human labeled examples.

### 1.1. First Step: Page Segmentation

First, page segmentation is performed in order to locate and identify structural elements -- like lists, tables, paragraphs, images, and headings -- on each page of a document. At Konfuzio, we perform page segmentation with a [mask R-CNN](https://arxiv.org/abs/1703.06870), a deep learning model that is able to detect the position and classification of objects within an image. The model is able to operate across all document types, regardless of the language of the document.

### 1.2. Second Step: Text Recognition

Next, we perform optical character recognition (OCR), which allows us to extract the text from each document. At Konfuzio, we use multiple OCR algorithms and select the appropriate one depending on the quality of the incoming document, enabling us to extract text from low-quality images and scanned documents. The extracted text is appended to the document so it can be searched and used in the next steps.

### 1.3. Third Step: Entity Extraction

Once we have the extracted text we can began to recognize entities within the text using a named entity recognition (NER) model. Our NER models use state-of-the-art deep learning-based natural language processing (NLP) models, such as [BERT](https://arxiv.org/abs/1810.04805). For each word in the document the NER model determines if it belongs into one of many useful generic categories, such as: date, country and currency, as well as user-defined domain-specific entities such as: currency, gross amount and house number.

### 1.4. Fourth Step: Entity Aggregation

In the fourth step the recognized entities are group together into user-defined sections, such as grouping "departure time", "arrival time", "destination" and "cost" entities together into a "travel booking" template. There can be multiple types of sections per document and multiple instances of each section, e.g. a document can have one "customer information" section and multiple "travel booking" sections. For each extracted entity we predict which type of section it belongs to, using information from the entity text, entity class and the document text surrounding the entity itself. As with the tntiy extraction, these predictions use state-of-the-art NLP techniques.

### 1.5 Fifth Step: Active Learning

To improve performance of the entity extraction and aggregation steps we make use of *active learning* techniques. User feedback, such as the acceptance or rejection of an extracted entity, is sent back to the NLP models which can adapt to domain-shifts within the training data.
