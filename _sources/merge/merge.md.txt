.. meta::
   :description: Documentation of the ExtractionAI merge logic.

# Merging Logic

Our extraction AI runs a merging logic at two steps in the extraction process. The first is a horizontal merging of Spans right after the Label classifier. This can be particularly useful when using the [Whitespace tokenizer](https://dev.konfuzio.com/sdk/sourcecode.html#konfuzio_sdk.tokenizer.regex.WhitespaceTokenizer) as it can find Spans containing spaces. The second merging logic is a vertical merging of Spans into a single multiline Annotation. Checkout the [architecture diagram](https://dev.konfuzio.com/sdk/contribution.html#architecture-sdk-to-server) for more detail.

## Horizontal Merge

When using an [Extraction AI](https://dev.konfuzio.com/sdk/sourcecode.html#extraction-ai), we merge adjacent horizontal Spans right after the Label classifier. 

A horizontal merging is valid only if:
1. All Spans have the same predicted Label
2. Confidence of predicted Label is above the Label threshold
3. All Spans are on the same line
4. No extraneous characters in between Spans
5. A maximum of 5 spaces in between Spans
6. The [Label type](https://dev.konfuzio.com/web/api.html#supported-data-normalization) is not one of the following: 'Number', 'Positive Number', 'Percentage', 'Date'
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
