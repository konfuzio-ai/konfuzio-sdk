## Find possible outliers among the ground-truth Annotations

If you want to ensure that Annotations of a Label are consistent and check for possible outliers, you can use one of 
the `Label` class's methods. There are three of them available.

- `get_probable_outliers_by_regex` looks for the worst regexes used to find the Annotations. "Worst" is determined by
the number of True Positives calculated upon evaluating the regexes' performance. Returns Annotations predicted by the
regexes with the least amount of True Positives. By default, the method returns Annotations retrieved by the regex that
performs on the level of 10% in comparison to the best one.

  .. literalinclude:: /sdk/boilerplates/test_outlier_annotations.py
     :language: python
     :start-after: start project
     :end-before: end project
     :dedent: 4

- `get_probable_outliers_by_confidence` looks for the Annotations with the least confidence level, provided it is lower
than the specified threshold (the default threshold is 0.5). Accepts an instance of EvaluationExtraction class as an input and uses confidence predictions from there.
   
  .. literalinclude:: /sdk/boilerplates/test_outlier_annotations.py
     :language: python
     :start-after: start project
     :end-before: end project
     :dedent: 4
  .. literalinclude:: /sdk/boilerplates/test_outlier_annotations.py
     :language: python
     :start-after: start get austellungsdatum
     :end-before: end get austellungsdatum
     :dedent: 4
  .. literalinclude:: /sdk/boilerplates/test_outlier_annotations.py
     :language: python
     :start-after: start confidence_outliers
     :end-before: end confidence_outliers
     :dedent: 4

- `get_probable_outliers_by_normalization` looks for the Annotations that are unable to pass normalization by the data
type of the given Label (meaning that they are not of the same data type themselves, thus outliers).

  .. literalinclude:: /sdk/boilerplates/test_outlier_annotations.py
     :language: python
     :start-after: start normalization
     :end-before: end normalization
     :dedent: 4

All three of the methods return a list of Annotations that are deemed outliers by the logic of the current method; the 
contents of the output are not necessarily wrong, however, they may have some difference from the main body of the 
Annotations under a given Label.

To have a more thorough check, you can use a method `get_probable_outliers` that allows for combining the 
aforementioned methods or have them run together and return only those Annotations that were detected by all of them.

Here's an example of running the latter method with one of the search methods disabled explicitly. By default, all 
three of the search methods are enabled.

.. literalinclude:: /sdk/boilerplates/test_outlier_annotations.py
 :language: python
 :start-after: start combined
 :end-before: end combined
 :dedent: 4
