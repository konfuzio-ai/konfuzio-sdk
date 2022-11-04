## Create Regex-based Annotations

Let's see a simple example of how can we use the `konfuzio_sdk` package to get information on a project and to post annotations.

You can follow the example below to post annotations of a certain word or expression in the first document uploaded.

```python
import re

from konfuzio_sdk.data import Project, Annotation, Label

my_project = Project(id_=YOUR_PROJECT_ID)

# Word/expression to annotate in the document
# should match an existing one in your document
input_expression = "John Smith"

# Label for the annotation
label_name = "Name"
# Creation of the Label in the project default label set
my_label = Label(my_project, text=label_name)
# Saving it online
my_label.save()

# Label Set where label belongs
label_set_id = my_label.label_sets[0].id_

# First document in the project
document = my_project.documents[0]

# Matches of the word/expression in the document
matches_locations = [(m.start(0), m.end(0)) for m in re.finditer(input_expression, document.text)]

# List to save the links to the annotations created
new_annotations_links = []

# Create annotation for each match
for offsets in matches_locations:
    annotation_obj = Annotation(
        document=document,
        document_id=document.id_,
        start_offset=offsets[0],
        end_offset=offsets[1],
        label=my_label,
        label_set_id=label_set_id,
        accuracy=1.0,
    )
    new_annotation_added = annotation_obj.save()
    if new_annotation_added:
        new_annotations_links.append(annotation_obj.get_link())

print(new_annotations_links)

```

## Evaluate a Trained Extraction AI Model

In this example we will see how we can evaluate a trained `RFExtractionAI` model. We will assume that we have a trained pickled model available. See [here](https://dev.konfuzio.com/sdk/examples/examples.html#train-a-konfuzio-sdk-model-to-extract-information-from-payslip-documents) for how to train such a model, and check out the [Evaluation](https://dev.konfuzio.com/sdk/sourcecode.html#evaluation) documentation for more details.

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.information_extraction import load_model

pipeline = load_model(MODEL_PATH)

# To get the evaluation of the full pipeline
evaluation = pipeline.evaluate_full()
print(f"Full evaluation F1 score: {evaluation.f1()}")
print(f"Full evaluation recall {evaluation.recall()}")

# To get the evaluation of the tokenizer alone
evaluation = pipeline.evaluate_tokenizer()
print(f"Tokenizer evaluation F1 score: {evaluation.tokenizer_f1()}")

# To get the evaluation of the Label classifier given perfect tokenization
evaluation = pipeline.evaluate_clf()
print(f"Label classifier evaluation F1 score: {evaluation.clf_f1()}")

# To get the evaluation of the LabelSet given perfect Label classification
evaluation = pipeline.evaluate_template_clf()
print(f"Label Set evaluation F1 score: {evaluation.f1()}")

```