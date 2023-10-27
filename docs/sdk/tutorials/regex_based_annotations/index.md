## Create Regex-based Annotations

In this guide, we'll show you how to use Python and Regular Expressions (Regex) to automatically identify and annotate 
specific text patterns within a Document. Let's say we have a Document, and we want to highlight every instance of the 
term "Musterstraße", which might represent a specific street name or location. Our task is to find this term, label it 
as "Lohnart", and associate it with the 'Brutto-Bezug' Label Set.

You can follow the example below to post Annotations of a certain word or expression in an uploaded Document. The 
posting is done with the usage of the API call to the Server in the `save()` command.

.. literalinclude:: /sdk/boilerplates/test_regex_based_annotations.py
   :language: python
   :start-after: start import
   :end-before: end import
   :dedent: 4 
.. literalinclude:: /sdk/boilerplates/test_regex_based_annotations.py
   :language: python
   :start-after: start regex based
   :end-before: end regex based
   :dedent: 4 