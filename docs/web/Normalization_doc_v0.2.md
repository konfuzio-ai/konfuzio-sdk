## Supported Data Normalization

Our Konfuzio application is able to normalize various data formats for different data types used in the annotations of the documents within your project.
It translates different formats of numbers, percentages, date and boolean values into a unique and machine-readable format. Down below you find an overview of the normalizations of the different data types with specific examples.

### 1. Numbers

Konfuzio is able to recognize numbers encoded in a written format to translate them into a data type that represents machine-readable numbers (i.e. float). Numbers from one to twelve are transformed, either from german or english language.
Given an annotation as an offset-string, this annotation will be translated into an (absolute) number.  
Starting with the first case, a dash or a series of dashes which also might include commas or decimal points will be recognized as Zero.   
For the specific context not relevant signs, like quotation marks or whitespaces, are completely removed form the annotation. If the annotation contains the letter "S" or "H" after the digits, it will also be removed.  

Negative expressions are recognized either by dashes in front or after the digits, the letter "S" after the digits or the number being framed into brackets.
For regular numbers, positive signs will be completely removed, while negative signs will be placed in front of the digits.
When dealing with absolute numbers, negative or positive signs are fully removed, as well as whitespaces or quotation marks for all number formats.

As there are varying ways to display float numbers, e.g. by using dots instead of commas to display the thousands mark or decimal numbers, Konfuzio also takes care of this to display it in a uniform format. 
The uniform chosen format uses a the english/american standard. Hence, dots are used as a separation for decimal numbers and commas to mark the thousands with two decimal places. 

**To give you specific examples:**  
1) Expressions only consisting of one or multiple dashes will be translated into Zero:    
e.g.: -,-    

2) For regular numbers, negative signs will be placed in front of the digits and zeros after digits will be removed; floats are displayed with two decimal places:   
e.g.: 59,00-     
e.g.: 786,71-    
e.g.: (118.704)   

3) Absolute numbers are shown without negative or positive signs in the common format as described above:    
e.g.: 59,00-  
e.g.: 786,71-  
e.g.: -2.759,7°   
e.g.: +159,;03  

4) Irrelevant signs and whitespaces will be removed and it will be transformed into the unique format with a dot instead of a comma as the decimal separator:    
e.g.: :2.000, 08   
e.g.: -2.759,7°    
e.g.: €1.010.296  
e.g.: 7,375,009+

5) Written numbers will be changed into a digits number format:  
e.g.: ein  
e.g.: eleven 

5) Certain cases can't be normalized from Konfuzio:  
e.g.: 43.34.34 

6) The expression "NIL" meaning "nothing" will be translated into 0, however strings including this expression can't be normalized:  
e.g.: NIL   
e.g.: StringThatIncludesNIL

   
| Input      | Able to convert?     | Output Excel/CSV | Output API | example |
| :-------------: | :----------: | :-----------: | :-----------: |:-----------:|
|  -,- | yes   | 0.0    | 0| 1 |
|  59,00- | yes   | -59.0    | -59 |2|
|  786,71- | yes   | -786.71    | -786.71 |2|
| (118.704) | yes   | -118704.0    | -118704 |2|
|  absolute no.: 59,00- | yes   | 59.0    | 59 |3|
|  absolute no.: 786,71- | yes   | 786.71    | 786.71 |3|
|  absolute no.: -2.759,7° | yes   | 2759.7    | 2759.7 |3|
|  absolute no.: +159,;03 | yes   | 159.03    | 159.03 |3|
|  :2.000, 08 | yes   | 2000.08    | 2000.08 |4|
|  -2.759,7° | yes   | -2759.7    | -2759.7 |4|
|  €1.010.296 | yes   | 1010296.0    | 1010296 |4|
|  7,375,009+ | yes   | 7375009.0  | 7375009 |4|
|  ein | yes   | 1.0   | 1 |5|
|  eleven | yes   | 11.0   | 11 |5|
| 43.34.34 | no   | None    | null |6|
|  NIL | yes   | 0.0   | 0  |6|
| StringThatIncludesNIL | no  | None   | null  |6|


### 2. Percentage Numbers

Konfuzio also handels percentage numbers of different formats and brings them into a machine-readable one. Percentage expressions will not be displayed in the classic percentage format with the percentage sign %. The decimal number format without the percentage sign will be used, but with a dot as a decimal separator and 4 decimal places. 

**To give you specific examples:**  
1) Digits which are separated by commas with two decimal places will be easily converted into a uniform format, either with or without the percentage sign in its original format.  
e.g.: 12,34 = 0.1234  
e.g.: 12,34 % = 0.1234  
e.g.: 434,27% = 4.3427    
e.g.: 59,00- = 0.59  
e.g.: 123,45 = 12.345  
e.g.: 0,00 = 0  


| Input      | Able to convert?     | Output Excel/CSV | Output API |
| :-------------: | :----------: | :-----------: | :-----------: |
|  12,34 | yes   | 0.1234    | 0.1234 |
|  12,34 % | yes   | 0.1234    | 0.1234 |
|  434,27% | yes   | 4.3427  | 4.3427 |
|  59,00- | yes   | 0.59  | 0.59 |
|  123,45 | yes   | 1.2345  | 1.2345 |
|  0,00 | yes   | 0.0    | 0 |


### 3. Date Values

Konfuzio applies the so-called iso format (XXXX.XX.XX) for dates. It checks initially whether this format is already used which then won't be altered in this case. However, if another format is used, it will be adapted. In the next step, it checks if a format consisting only of a month and a year is used or even just a year without any other date information besides that.

Konfuzio recognizes written months either in german or english language and translates them into the iso format. 
                         
                              
**To give you specific examples:**
1) Dates where the month is posted in a written expression with the year and the day as digits can be transformed into the iso format without any problems.  
e.g.: 1. November 2019 = 2019-11-01   
e.g.: 13 Mar 2020 = 2020-03-13

2) If the year is indicated with just two digits, it will also be recognized, even if it's not separated with a sign like a dot or something similar.  
e.g.: 23.0919 = 2019-09-23  
e.g.: (29.03.2018) = 2018-03-29  

3) Given no information for the year, Konfuzio will assume 0000 by default.  
e.g.: /04.12. = 0000-12-04

4) If there is no information about the specific day, or even day and month, our application assumes the first day either of the respective year or year and month.  
e.g.: Oktober 2011 = 2011-10-01  
e.g.: 2001 = 2001-01-01

5) Date time values are translated into the iso format as well, but removing the time values:  
e.g.: 1993-02-05T00:00:00 = 1993-02-05

6) Some cases can't be identified correctly or uniquely. Thus, Konfuzio can't transfer it into a date format and will return "None".  
e.g.: 14132020 = None  
e.g.: 23.0K.2010 = None  
e.g.: 30.07.2.90 = None
   

| Input      | Able to convert?     | Output Excel/CSV | Output API |
| :-------------: | :----------: | :-----------: | :-----------: |
|  1. November 2019 | yes   | 2019-11-01   | 2019-11-01 |
|  13 Mar 2020 | yes   | 2020-03-13    | 2020-03-13 |
|  23.0919 | yes   | 2019-09-23    | 2019-09-23 |
|  (29.03.2018) | yes   | 2018-03-29    | 2018-03-29 |
|  /04.12. | yes   | 0000-12-04    | 0000-12-04 |
|  Oktober 2011 | yes   | 2011-10-01    | 2011-10-01 |
|  2001 | yes   | 2001-01-01    | 2001-01-01 |
|  1993-02-05T00:00:00| yes  | 1993-02-05  | 1993-02-05 |
|  14132020 | no   | None    | null |
|  23.0K.2010 | no   | None  | null |
|  30.07.2.90 | no   | None  | null |


### 4. Boolean values

Our application is also able to translate certain expressions into so-called boolean values, representing true or false values. This is based on certain pre-specified words. These words are representing positive or negative connotated responses with certain signal words which can be found down below in the _no_list_ and _yes_list_.


The pre-specified positive and negative (hence true and false) expressions:  
no_list = ['NEIN', 'NICHT', 'KEIN', 'OHNE', 'NO']  
yes_list = ['VORHANDEN', 'JA', 'MIT', 'YES']

Given the following examples, you can recognize how certain expressions are clustered into either "False" or "True" boolean values. If the expression is including one of the "no" or "yes" words from above, Konfuzio allocates it to True or False. However, _only if yes/no word is not on last place in expression_!?

**To give you specific examples:** 
1) The word "nicht" ("no") will be assigned to "false".  
2) The expression "ja" ("yes") will be translated into "true".
3) Expressions including the no or yes signal words can be translated, but only if the expression is starting with this word:  
e.g.: nicht versichert = false  
e.g.: not insured = false  
e.g.: inkludiert: ja  = None  
e.g.: included: yes = None  
e.g.: inkludiert ja = None  
e.g.: included yes = None  
e.g.: ja inkludiert = true  
e.g.: yes included = true  
e.g.: alleinstehend ohne Kind = None  

5) Empty expressions, like " ", can't be translated. 


| Input      | Able to convert?     | Output Excel/CSV | Output API |
| :-------------: | :----------: | :-----------: | :-----------: |
|  nicht | yes   | false    | false |
|  no | yes   | false     | false |
|  ja | yes   | true    | true |
|  yes | yes   | true   | true |
|  nicht versichert| yes   | false    | false |
|  not insured | yes   | false     | false |
|  inkludiert: ja | no   | None    | null |
|  included: yes | no   | None   | null |
|  inkludiert ja | no   | None     | null |
|  included yes | no   | None   | null |
|  ja inkludiert | yes   | true    | true|
|  yes included | yes   | true  | true |
|  alleinstehend ohne Kind | no  | None  | null |
|   | no   |  not recognizable as annotation | not recognizable as annotation |


### 5. Known Issues and remarks

1) Limitations to other languages besides german: Konfuzio is optimized for the german language. It may partly support english expressions, but is limited in the usage in english.

2) Limitations to case sensitivity of boolean values: Classifying certain expressions as true or false values is based on pre-specified expressions. Thus, it is limited to these signal words as shown above in the lists when normalizing the inputs.

3) The non-normalizable values which can't be converted from Konfuzio will be marked as "not machine-readable" on app.konfuzio.com and can be viewed in the smart view of the documents. 
 
5) Your excel output might differ in its format from the one defined above. This is due to the default formats from excel, e.g. the default date format which might change with the version of excel of your respective country.  


## Support

To qualify for minimum response times according to your SLA, you must report the issue using the email [support@konfuzio.com](emailto:support@konfuzio.com).
