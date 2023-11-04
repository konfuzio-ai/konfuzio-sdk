.. meta::
   :description: Documentation of the ExtractionAI merge logic.

# Merging of predicted Annotations

Our extraction AI runs a merging logic at two steps in the extraction process. The first is a horizontal merging of Spans right after the Label classifier. This can be particularly useful when using the [Whitespace tokenizer](https://dev.konfuzio.com/sdk/sourcecode.html#konfuzio_sdk.tokenizer.regex.WhitespaceTokenizer) as it can find Spans containing spaces. The second merging logic is a vertical merging of Spans into a single multiline Annotation. Checkout the [architecture diagram](https://dev.konfuzio.com/sdk/contribution.html#architecture-sdk-to-server) for more detail.

## Horizontal Merge

When using an [Extraction AI](https://dev.konfuzio.com/sdk/sourcecode.html#extraction-ai), we merge adjacent horizontal Spans right after the Label classifier. The confidence of the resulting new Span if taken to be the mean confidence of the original Spans being merged.

A horizontal merging is valid only if:
1. All Spans have the same predicted Label
2. Confidence of predicted Label is above the Label threshold
3. All Spans are on the same line
4. Spans are not overlapping
5. No extraneous characters in between Spans
6. A maximum of 5 spaces in between Spans
7. The [Label type](https://dev.konfuzio.com/web/api.html#supported-data-normalization) is not one of the following: 'Number', 'Positive Number', 'Percentage', 'Date'
 OR the resulting merging create a Span [normalizable](https://dev.konfuzio.com/web/api.html#supported-data-normalization) to the same type

|          Input          | Able to merge? | Reason | Result |
|:-----------------------:|:-----------:| :-----------: | :-----------: |
|  <span style="background-color: #ff726f">Text</span><span>&nbsp;&nbsp;</span><span style="background-color: #ff726f">Annotation</span> |      yes     |    /    | <span style="background-color: #ff726f">Text</span><span style="background-color: #ff726f">&nbsp;&nbsp;</span><span style="background-color: #ff726f">Annotation</span> |
|  <span style="background-color: #ff726f">Text</span><span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="background-color: #ff726f">Annotation</span> |      no     |    5.    | <span style="background-color: #ff726f">Text</span><span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="background-color: #ff726f">Annotation</span> |
|  <span style="background-color: #ff726f">Text</span><span>&nbsp;.&nbsp;</span><span style="background-color: #ff726f">Annotation</span> |      no     |    4.    | <span style="background-color: #ff726f">Text</span><span>&nbsp;.&nbsp;</span><span style="background-color: #ff726f">Annotation</span> |
|  <span style="background-color: #ff726f">Annotation</span><span>&nbsp;</span><span style="background-color: #86c5da">7</span> |      no     |    1.    | <span style="background-color: #ff726f">Annotation</span><span>&nbsp;</span><span style="background-color: #86c5da">7</span> |
|  <span style="background-color: #86c5da">34</span><span>&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="background-color: #86c5da">98</span> |      no     |  6. ([see here](https://dev.konfuzio.com/web/api.html#numbers))  | <span style="background-color: #86c5da">34</span><span>&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="background-color: #86c5da">98</span> |
|  <span style="background-color: #86c5da">34</span><span>&nbsp;&nbsp;</span><span style="background-color: #86c5da">98</span> |      yes     |  /  | <span style="background-color: #86c5da">34</span><span style="background-color: #86c5da">&nbsp;&nbsp;</span><span style="background-color: #86c5da">98</span> |
|  <span style="background-color: #34df00">November</span><span>&nbsp;</span><span style="background-color: #34df00">2022</span> |     yes     |   /    | <span style="background-color: #34df00">November</span><span style="background-color: #34df00">&nbsp;</span><span style="background-color: #34df00">2022</span> |
|  <span style="background-color: #34df00">Novamber</span><span>&nbsp;</span><span style="background-color: #34df00">2022</span> |     no     |    6. ([see here](https://dev.konfuzio.com/web/api.html#date-values))   | <span style="background-color: #34df00">Novamber</span><span>&nbsp;</span><span style="background-color: #34df00">2022</span> |
|  <span style="background-color: #ff8c00">34</span><span>&nbsp;&nbsp;</span><span style="background-color: #ff8c00">98%</span> |      yes     |  /  | <span style="background-color: #ff8c00">34</span><span style="background-color: #ff8c00">&nbsp;&nbsp;</span><span style="background-color: #ff8c00">98%</span> |
|  <span style="background-color: #ff8c00">34</span><span>&nbsp;&nbsp;</span><span style="background-color: #dcdcdc">98%</span> |      no     |  2.  | <span style="background-color: #ff8c00">34</span><span>&nbsp;&nbsp;</span><span style="background-color: #dcdcdc">98%</span> |


Label Type: <span style="background-color: #ff726f">Text</span><br>
Label Type: <span style="background-color: #86c5da">Number</span><br>
Label Type: <span style="background-color: #34df00">Date</span><br>
Label Type: <span style="background-color: #ff8c00">Percentage</span><br>
Label Type: <span style="background-color: #dcdcdc">NO LABEL/Below Label threshold</span>


## Vertical Merge

When using an [Extraction AI](https://dev.konfuzio.com/sdk/sourcecode.html#extraction-ai), we join adjacent vertical Spans into a single Annotation after the LabelSet classifier. 

A vertical merging is valid only if:

1. They are on the same Page
2. They are predicted to have the same Label
3. Multiline annotations with this Label exist in the training set
4. Consecutive vertical Spans either overlap in the x-axis, OR the preceding Span is at the end of the line, and following Span is at the beginning of the next
5. Confidence of predicted Label is above the Label threshold
6. Spans are on consecutive lines
7. Merged lower Span belongs to an Annotation in the same AnnotationSet, OR to an AnnotationSet with only a single Annotation

|          Input          | Able to merge? | Reason |
|:-----------------------:|:----------:| :-----------: |
|  <span style="background-color: #ff726f">Text</span><br><span style="background-color: #ff726f">Annotation</span> |      yes     |    /    | 
|  <span style="background-color: #ff726f">Annotation</span><br><span style="background-color: #86c5da">42</span> |      no     |    2.    |
|  <span style="background-color: #ff726f">Text</span><span>&nbsp;</span><span style="background-color: #dcdcdc">more text</span><br><span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="background-color: #ff726f">Annotation</span> |     no     |    4.    | 
| <span style="background-color: #dcdcdc">Some random text</span><span>&nbsp;</span><span style="background-color: #ff726f">Text</span><br><span style="background-color: #ff726f">Annotation</span> |      yes     |    /    |
| <span style="background-color: #dcdcdc">Some random text</span><span>&nbsp;</span><span style="background-color: #ff726f">Text</span><span>&nbsp;&nbsp;</span><span style="background-color: #dcdcdc">.</span><br><span style="background-color: #ff726f">Annotation</span> |      no     |    4.    |
|  <span style="background-color: #ff726f">Text</span><span>&nbsp;</span><span style="background-color: #dcdcdc">more text</span><br><span>&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="background-color: #ff726f">Annotation</span><br><span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="background-color: #ff726f">42</span> |     yes     |    /    |
|  <span style="background-color: #ff726f">Text</span><span>&nbsp;</span><br><span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="background-color: #dcdcdc">more text</span><span>&nbsp;&nbsp;&nbsp;&nbsp;</span><br><span>&nbsp;&nbsp;&nbsp;</span><span style="background-color: #ff726f">Annotation</span> |     no     |    6.    |
|  <span style="background-color: #ff726f">Annotation</span><span>&nbsp;&nbsp;</span><span style="background-color: #86c5da">Nb.</span><br><span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="background-color: #86c5da">42</span> |      yes     |   <span>*</span>  |
|  <span style="background-color: #ff726f">Annotation</span><span>&nbsp;&nbsp;</span><span style="background-color: #86c5da">41</span><br><span style="background-color: #ff726f">Annotation</span><span>&nbsp;&nbsp;</span><span style="background-color: #86c5da">42</span> |      no     |    7. <span>**</span> |

<span>* The bottom Annotation is alone in its AnnotationSet and therefore can be merged.</span><br>
<span>** The Annotations on each line have been grouped into their own AnnotationSets and are not merged.</span>

<span style="background-color: #ff726f">Label 1</span><br>
<span style="background-color: #86c5da">Label 2</span><br>
<span style="background-color: #dcdcdc">NO LABEL/Below Label threshold</span>


## Horizontal and Vertical Merge with the Paragraph and Sentence Tokenizers

When using the [Paragraph](https://dev.konfuzio.com/sdk/sourcecode.html#paragraph-tokenizer) or [Sentence Tokenizer](https://dev.konfuzio.com/sdk/sourcecode.html#sentence-tokenizer) together with our [Extraction AI model](https://dev.konfuzio.com/sdk/sourcecode.html#extraction-ai), we do not use the rule based vertical and horizontal merge logic above, and instead use the sentence/paragraph segmentation provided by the Tokenizer.

The logic is as follows:

```mermaid

   graph TD
      A[Virtual Document] -->|Paragraph/Sentence Tokenizer|B(Document_A with NO_LABEL\n Paragraph/Sentence Annotations)
      B --> |RFExtractionAI feature extraction|C(Span features)
      C --> |RFExtractionAI labeling|D(Labeled Spans)
      D --> |extraction_result_to_document |E(Document B with labeled single-Span Annotations)
      E --> |"RFExtractionAI().merge_vertical_like(Document_B, Document_A)" |F(Document_B with labeled multi-line Annotations)
```


And here's an illustrated example of the merge logic in action:

.. image:: /_static/img/merge_docs_gif.gif
