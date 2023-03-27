## Find possible outliers among the ground-truth Annotations

If you want to ensure that Annotations of a Label are consistent and check for possible outliers, you can use one of 
the `Label` class's methods. There are three of them available; all of them require a list of Categories passed at the 
input to iterate over them upon fetching the Annotations.

- `get_probable_outliers_by_regex` looks for the worst regexes used to find the Annotations. "Worst" is determined by
the number of True Positives calculated upon evaluating the regexes' performance. Returns Annotations predicted by the
regexes with the least amount of True Positives. The default number of top worst regexes to return the Annotation from 
is 3.

  .. literalinclude:: /sdk/boilerplates/test_outlier_annotations.py
     :language: python
     :lines: 2,6-7,20,9

- `get_probable_outliers_by_confidence` looks for the Annotations with the least confidence level, provided it is lower
than 0.5. 
   
  .. literalinclude:: /sdk/boilerplates/test_outlier_annotations.py
     :language: python
     :lines: 2,6-7,20,15

- `get_probable_outliers_by_normalization` looks for the Annotations that are unable to pass normalization by the data
type of the given Label (meaning that they are not of the same data type themselves, thus outliers).

  .. literalinclude:: /sdk/boilerplates/test_outlier_annotations.py
     :language: python
     :lines: 2,6-7,20,21

To have a more thorough check, you can use a method `get_probable_outliers` that allows for combining the 
aforementioned methods or have them run together and return only those Annotations that were detected by all of them.

Here's an example of running the latter method with one of the search methods disabled explicitly. By default, all 
three of the search methods are enabled.

.. literalinclude:: /sdk/boilerplates/test_outlier_annotations.py
   :language: python
   :lines: 2,6-7,20,26
