## Create Regex-based Annotations

**Pro Tip**: Read our technical blog post [Automated Regex](https://helm-nagel.com/Automated-Regex-Generation-based-on-examples) to find out how we use Regex to detect outliers in our annotated data.

Let's see a simple example of how can we use the `konfuzio_sdk` package to get information on a project and to post annotations.

You can follow the example below to post annotations of a certain word or expression in the first document uploaded.

.. literalinclude:: /sdk/boilerplates/test_regex_based_annotations.py
   :language: python
   :lines: 2-5,8-17,19-21,23-28,30-47,49,51


## Train Label Regex Tokenizer

You can use the `konfuzio_sdk` package to train a custom Regex tokenizer. 

In this example, you will see how to find regex expressions that match with occurrences of the "Lohnart" Label in the 
training data. 

.. literalinclude:: /sdk/boilerplates/test_train_label_regex_tokenizer.py
   :language: python
   :lines: 2-5,10-23

## Finding Spans of a Label Not Found by a Tokenizer

Here is an example of how to use the `Label.spans_not_found_by_tokenizer` method. This will allow you to determine if a RegexTokenizer is suitable at finding the Spans of a Label, or what Spans might have been annotated wrong. Say, you have a number of annotations assigned to the `IBAN` Label and want to know which Spans would not be found when using the WhiteSpace Tokenizer. You can follow this example to find all the relevant Spans.

.. literalinclude:: /sdk/boilerplates/test_spans_not_found_label.py
   :language: python
   :lines: 2-4,7-14,16-18
