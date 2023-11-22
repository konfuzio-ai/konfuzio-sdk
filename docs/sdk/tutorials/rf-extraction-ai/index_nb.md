---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: konfuzio
    language: python
    name: python3
---

<!-- #region id="9f92e353" -->
## RFExtractionAI with Whitespace Tokenizer

---

**Prerequisites:**
- Ensure you have the `konfuzio_sdk` package installed. See [here](https://github.com/konfuzio-ai/konfuzio-sdk) for more info.

**Difficulty:** Intermediate

**Goal:**
The goal of this tutorial is to train a model using the Konfuzio SDK to extract relevant information like Name, Date, and Recipient from payslip Documents.

---

### Introduction
In this notebook we will see how to use the Konfuzio SDK to train a model to find and extract relevant information like Name, Date and Recipient from payslip Documents. 

Here we will see how to use the basic Konfuzio information extraction pipeline.
<!-- #endregion -->

<!-- #region id="2d3e2ae0" -->
### Setting things up
<!-- #endregion -->

<!-- #region id="0328f306" -->
First, we need to install the Konfuzio SDK. See [here](https://github.com/konfuzio-ai/konfuzio-sdk) for more info.

```python tags=["nbval-skip", "skip-execution"]
!pip install konfuzio-sdk
```

If you are using an online Project, you will also need to initialize the SDK and input your credentials.

```python tags=["nbval-skip", "skip-execution"]
konfuzio_sdk init
```
<!-- #endregion -->

<!-- #region id="4cf76443" -->
In this demo, we will just download a Project folder from the Konfuzio repository that we use for testing:
<!-- #endregion -->

<!-- #region -->
```bash
git init
git remote add origin https://github.com/konfuzio-ai/konfuzio-sdk/
git config core.sparseCheckout true
echo "tests/example_project_data" > .git/info/sparse-checkout 
echo "tests/variables.py" >> .git/info/sparse-checkout
echo "tests/__init__.py" >> .git/info/sparse-checkout
git pull origin master
```
<!-- #endregion -->

We now import the required packages:

```python colab={"base_uri": "https://localhost:8080/"} id="17589f76" outputId="0e8baed3-7603-4f6a-89ea-7a593e589901"
import os
import sys

import konfuzio_sdk
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.information_extraction import RFExtractionAI
from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer
from konfuzio_sdk.api import upload_ai_model
```

<!-- #region id="7a5469a7" -->
### Initializing the Project and the training pipeline
<!-- #endregion -->

<!-- #region id="9cc2176a" -->
Now we can load the Konfuzio Project. Here we use a simple offline Project included in the tests for the Konfuzio SDK. We can find it in the `OFFLINE_PROJECT` folder.
<!-- #endregion -->

```python tags=["remove-cell"]
# This is necessary to make sure we can import from 'tests'
import sys
sys.path.insert(0, '../../../../')
```

```python id="e0a04d2d"
from tests.variables import OFFLINE_PROJECT, TEST_DOCUMENT_ID, TEST_PAYSLIPS_CATEGORY_ID
```

<!-- #region id="fb7ae58f" -->
We now can initialize the Project. Since we're working with a local Project, we don't need to specify a Project `id_`. If you're working with a Project that can be found on the Konfuzio platform, you can just specify the `id_` of the Project and then start working with it in your local setup. 
<!-- #endregion -->

```python id="dfb713d3" tags=["remove-output"]
project = Project(id_=None, project_folder=OFFLINE_PROJECT)
```

<!-- #region id="64036925" -->
Each Project has one or more categories which will tell us how to process the Documents belonging to that category. Here we only have one category: `Lohnabrechnung` (i.e. payslip).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="8b02ab11" outputId="751a5ae6-b81e-40d3-e69e-1d57dfd523a1"
project.categories
```

```python id="84a68999"
category = project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID)
```

<!-- #region id="a4b602d3" -->
Now we can initialize the training pipeline. Here we use the `RFExtractionAI` class.

The model is composed of a Label and LabelSet classifier both using a [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) from scikit-learn. Our approach is inspired by [Sun et al. (2021)](https://arxiv.org/abs/2103.14470)

We first split the text of the Document into smaller chunks of text based on whitespaces and then use a combination of features to train the first Label classifier to assign labels to these chunks of text.

After the Label classifier, the LabelSet classifier takes the predictions from the Label classifier as input. The LabelSet classifier groups Annotations into Annotation sets.

Here we set `use_separate_labels` to True. It can be help with the training of the LabelSet classifier when there are labels that can be contained in more than one LabelSet.
pipeline = RFExtractionAI(use_separate_labels=True)
pipeline.category = category
<!-- #endregion -->

```python id="79f54bbe" tags=["remove-output"]
pipeline = RFExtractionAI(use_separate_labels=True)
pipeline.category = category
```

<!-- #region id="bbf86d76" -->
We can also set the pipeline `test_documents` attribute. This will be used to evaluate our model later on. In this Project we have 3 test Documents.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="25ee6de6" outputId="8f0fc7da-3138-4988-b992-d9271dd4e6c5"
pipeline.test_documents = category.test_documents()
pipeline.test_documents
```

<!-- #region id="9bb807cf" -->
Now it's time to look at our training Documents.
<!-- #endregion -->

```python id="aa9d508f"
# getting all the documents in category TEST_PAYSLIPS_CATEGORY_ID
documents = category.documents()
```

<!-- #region id="ead6bb1f" -->
We have 25 training Documents we can use to train our classifier. 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="e78ec8ca" outputId="f4390bda-88ed-49ff-b799-f0fcc251fb03"
len(documents)
```

<!-- #region id="6afe1c81" -->
Let's have a look at what exactly a Document in this dataset looks like. 
<!-- #endregion -->

```python id="622c5089"
document = documents[0]
document.text
```

<!-- #region id="983fb505" -->
This is the output of Optical Character Recognition (OCR) model on the following image:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="0d42384b" outputId="df1dadfb-f9af-4fd6-8b5e-50f676ee06dc"
print(document.pages())
document.get_page_by_index(0).get_image().convert('RGB')
```

<!-- #region id="723bc250" -->
And those are the annotations we want to find:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="b92928df" outputId="598a225f-0234-4a2f-b6d0-05e2ac2ca9c6"
document.annotations()
```

<!-- #region id="77013d8e" -->
Documents in this category may include the following fields:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="9b0f5e1d" outputId="471386ee-25ca-44be-e371-c8dae2dddef8"
for label in category.labels:
    print(label.name)
```

<!-- #region id="f30c7162" -->
This is all the information we may want to identify in Documents of this category.
<!-- #endregion -->

<!-- #region id="44956f4e" -->
### Set the Whitespace Tokenizer

Now we need to decide how we will segment our text for entity detection and classification. In this example we will use a simple [Whitespace tokenizer](https://dev.konfuzio.com/sdk/sourcecode.html#konfuzio_sdk.tokenizer.regex.WhitespaceTokenizer). It will seperate text segments by whitespace and allow us to label each individual segment. 
<!-- #endregion -->

```python id="0e7cfdc4"
pipeline.tokenizer = WhitespaceTokenizer()
```

```python id="c7904591"
import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
```

<!-- #region id="a7cd37ce" -->
To train our label classifier, we first need to extract all the features for each segment of text, which will allow the classifier to automatically detect where the relevant information is. These include features about the text in a segment, its position on the page and its relation with and features about nearby segments of text.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="00f1ec7c" outputId="acccd209-5939-467a-b8a4-3d00cd78aabe"
%%capture
# Extract features
pipeline.df_train, pipeline.label_feature_list = pipeline.feature_function(documents=documents, require_revised_annotations=False)
```

```python colab={"base_uri": "https://localhost:8080/"} id="7bfad348" outputId="7111e70e-cca4-43b5-a77a-25c3737e9ef2"
pipeline.df_train.shape
```

<!-- #region id="364015e5" variables={" len(pipeline.label_feature_list) ": "270"} -->
This is the number of features we use to classify each annotation:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="84e1e044" outputId="4a0f5d4c-f34e-473c-f12e-293e9eef2e83"
len(pipeline.label_feature_list)
```

<!-- #region id="d7f4afa9" -->
Now we're ready to train our classifier:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="cc94f619" outputId="a8a736d7-35f4-4c33-e1cb-0bf1da1b5c6e"
pipeline.fit()
```

<!-- #region id="30a50584" -->
### Evaluation
We can now evaluate our classifier.
<!-- #endregion -->

```python id="8483882e"
%%capture
evaluation = pipeline.evaluate_full()
```

<!-- #region id="59195a0a" -->
This will return an Evaluation object with all the relevant evaluation stats like the [F1-score](https://en.wikipedia.org/wiki/F-score):
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="d8552473" outputId="07baf4dc-f633-4453-81bf-ff51e4a2fc7c"
print('F1', evaluation.f1(None))
print('tp', evaluation.tp(None))
print('fp', evaluation.fp(None))
print('fn', evaluation.fn(None))
print('tn', evaluation.tn(None))
```

<!-- #region id="f6659881" -->
### Display

We can also show the output of the model for visual inspection.
<!-- #endregion -->

```python id="9cb13ead"
%%capture
test_document = project.get_document_by_id(TEST_DOCUMENT_ID)

virtual_doc = pipeline.extract(test_document)
```

<!-- #region id="281fe15d" -->
Now we can see what the model has learned:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="86110e70" outputId="bb70d2dd-8dd7-4797-d205-7ce579799790"
virtual_doc.get_page_by_index(0).get_annotations_image()
```

<!-- #region id="f1f50a5f" -->
You may also upload the model to the Konfuzio platform and view extraction results with the Konfuzio SmartView or DVUI.
<!-- #endregion -->

<!-- #region id="7889449e" -->
We can now save the trained model:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="4fea7168" outputId="b59b8ba2-e07c-47c4-eba0-f4af87db0228" tags=["remove-output"]
pipeline_path = pipeline.save(output_dir=project.model_folder)
pipeline_path
```

<!-- #region id="c8882e8f" -->
You can only upload your Annotation classifier to your own instance of the Konfuzio Server. Please [contact](https://konfuzio.com/en/contact/) our sales team if you need an on-premise setup if the Konfuzio Server.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 311} id="7838d762" outputId="d405f4d7-0717-43ac-f44a-40107e286eb8" tags=["skip-execution", "nbval-skip"]
upload_ai_model(ai_model_path=pipeline_path)
```

### Conclusion


In this notebook, we learned how to train a model using the Konfuzio SDK to extract relevant information like Name, Date, and Recipient from payslip Documents. We used a combination of rule-based extraction and machine learning-based extraction to achieve this. The trained model can be saved and uploaded to the Konfuzio platform for further use.


### What's next?
- <a href="/sdk/tutorials/upload-your-ai">Learn how to upload your custom extraction model</a>
- <a href="/sdk/tutorials/async_upload_with_callback">Pull Documents Uploaded Asynchronously with a Webhook</a>
