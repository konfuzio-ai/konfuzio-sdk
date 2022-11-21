.. meta::
   :description: Documentation of the ExtractionAI horizontal merge logic.

# Horizontal Merge

After the Label classifier, we merge adjacent horizontal spans. 

A horizontal merging is valid only if:
1. All spans have the same predicted Label
2. Confidence of predicted Label is above the Label threshold
3. All spans are on the same line
4. No extraneous characters in between spans
5. A maximum of 5 spaces in between spans
6. The Label type is not one of the following: 'Number', 'Positive Number', 'Percentage', 'Date'
 OR the resulting merging create a span normalizable to the same type

|          Input          | Able to merge? | Reason | Result |
|:-----------------------:|:-----------:| :-----------: | :-----------: |
|  <span style="background-color: #ff726f">Text</span><span>&nbsp;&nbsp;</span><span style="background-color: #ff726f">Annotation</span> |      yes     |    /    | <span style="background-color: #ff726f">Text</span><span style="background-color: #ff726f">&nbsp;&nbsp;</span><span style="background-color: #ff726f">Annotation</span> |
|  <span style="background-color: #ff726f">Text</span><span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="background-color: #ff726f">Annotation</span> |      no     |    5.    | <span style="background-color: #ff726f">Text</span><span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="background-color: #ff726f">Annotation</span> |
|  <span style="background-color: #ff726f">Text</span><span>&nbsp;.&nbsp;</span><span style="background-color: #ff726f">Annotation</span> |      no     |    4.    | <span style="background-color: #ff726f">Text</span><span>&nbsp;.&nbsp;</span><span style="background-color: #ff726f">Annotation</span> |
|  <span style="background-color: #ff726f">Annotation</span><span>&nbsp;</span><span style="background-color: #86c5da">7</span> |      no     |    1.    | <span style="background-color: #ff726f">Annotation</span><span>&nbsp;</span><span style="background-color: #86c5da">7</span> |
|  <span style="background-color: #86c5da">34</span><span>&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="background-color: #86c5da">98</span> |      no     |  6. ([see here](https://dev.konfuzio.com/web/api.html#supported-data-normalization))  | <span style="background-color: #86c5da">34</span><span>&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="background-color: #86c5da">98</span> |
|  <span style="background-color: #86c5da">34</span><span>&nbsp;&nbsp;</span><span style="background-color: #86c5da">98</span> |      yes     |  /  | <span style="background-color: #86c5da">34</span><span style="background-color: #86c5da">&nbsp;&nbsp;</span><span style="background-color: #86c5da">98</span> |
|  <span style="background-color: #34df00">November</span><span>&nbsp;</span><span style="background-color: #34df00">2022</span> |     yes     |   /    | <span style="background-color: #34df00">November</span><span style="background-color: #34df00">&nbsp;</span><span style="background-color: #34df00">2022</span> |

Label Type: <span style="background-color: #ff726f">Text</span><br>
Label Type: <span style="background-color: #86c5da">Number</span><br>
Label Type: <span style="background-color: #34df00">Date</span>

