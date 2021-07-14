## Supported Data Normalization

Our Konfuzio application can normalize various data formats for different data types used in the annotations of the documents within your project.
It translates different formats of numbers, percentages, date, and boolean values into a unique and machine-readable format. Down below you find an overview of the normalizations of the different data types with specific examples.

### 1. Numbers

Konfuzio is able to recognize numbers encoded in a written format to translate them into a data type that represents machine-readable numbers (i.e. float). Numbers from one to twelve are transformed, either from German or English language.
Given an annotation as an offset string, this annotation will be translated into an (absolute) number.  
Starting with the first case, a dash or a series of dashes which also might include commas or decimal points will be recognized as Zero.   
For the specific context not relevant signs, like quotation marks or whitespaces, are completely removed from the annotation. If the annotation contains the letter "S" or "H" after the digits, it will also be removed.  

Negative expressions are recognized either by dashes in front or after the digits, the letter "S" after the digits, or the number being framed into brackets.
For regular numbers, positive signs will be completely removed, while negative signs will be placed in front of the digits.
When dealing with absolute numbers, negative or positive signs are fully removed, as well as whitespaces or quotation marks for all number formats.

As there are varying ways to display float numbers, e.g. by using dots instead of commas to display the thousands mark or decimal numbers, Konfuzio also takes care of this to display it in a uniform format. 
The uniform chosen format uses the English/American standard. Hence, dots are used as a separation for decimal numbers and commas to mark the thousands with two decimal places. 

**To give you specific examples with the outputs down below in the table:**  
1) Expressions only consisting of one or multiple dashes are translated into Zero:    
e.g.: -,-    

2) For regular numbers, negative signs are placed in front of the digits; floats are displayed with two decimal places:   
e.g.: 59,00-     
e.g.: 786,71-    
e.g.: (118.704)   

3) Absolute numbers are shown without negative or positive signs in the common format as described above:    
e.g.: 59,00-  
e.g.: 786,71-  
e.g.: -2.759,7°   
e.g.: +159,;03  

4) Irrelevant signs and whitespaces are removed and it will be transformed into the unique format with a dot instead of a comma as the decimal separator:    
e.g.: :2.000, 08   
e.g.: -2.759,7°    
e.g.: €1.010.296  
e.g.: 7,375,009+

5) Written numbers are changed into a digits number format:  
e.g.: ein  
e.g.: eleven 

5) Certain cases can't be normalized from Konfuzio:  
e.g.: 43.34.34 

6) The expression "NIL" meaning "nothing" is translated into 0; however, strings including this expression can't be normalized:  
e.g.: NIL   
e.g.: StringThatIncludesNIL

   
| Input      | example no.| Able to convert?     | Output Excel/CSV | Output API | Datatype CSV | Datatype JSON |
| :-------------: | :----------: | :-----------: | :-----------: |:-----------:|:-----------:|:-----------:|
|  -,- |1 | yes   | 0.0    | 0| string | number |
|  59,00- | 2| yes   | -59.0    | -59 | string | number |
|  786,71- |2| yes   | -786.71    | -786.71 | string | number |
| (118.704) | 2|yes   | -118704.0    | -118704 | string | number |
|  absolute no.: 59,00- |3| yes   | 59.0    | 59 | string | number |
|  absolute no.: 786,71- |3| yes   | 786.71    | 786.71 | string | number |
|  absolute no.: -2.759,7° |3| yes   | 2759.7    | 2759.7 | string | number |
|  absolute no.: +159,;03 |3| yes   | 159.03    | 159.03 | string | number |
|  :2.000, 08 |4| yes   | 2000.08    | 2000.08 | string | number |
|  -2.759,7° |4| yes   | -2759.7    | -2759.7 | string | number |
|  €1.010.296 | 4| yes   | 1010296.0    | 1010296 | string | number |
|  7,375,009+ | 4|yes   | 7375009.0  | 7375009 | string | number |
|  ein |5| yes   | 1.0   | 1 | string | number |
|  eleven |5| yes   | 11.0   | 11 | string | number |
| 43.34.34 | 6|no   | None    | null | - | - |
|  NIL |6| yes   | 0.0   | 0  | - |- | 
| StringThatIncludesNIL |6| no  | None   | null  | - | - |


### 2. Percentage Numbers

Konfuzio also handles percentage numbers of different formats and brings them into a machine-readable one. Percentage expressions will not be displayed in the classic percentage format with the percentage sign %. The decimal number format without the percentage sign will be used, but with a dot as a decimal separator and 4 decimal places. 

**To give you specific examples with the outputs down below in the table::**  
1) Digits that are separated by commas with two decimal places will be easily converted into a uniform format, either with or without the percentage sign in its original format.  
e.g.: 12,34   
e.g.: 12,34 %  
e.g.: 434,27%    
e.g.: 59,00-  
e.g.: 123,45  
e.g.: 0,00 


| Input      | example no. | Able to convert?     | Output Excel/CSV | Output API | Datatype CSV | Datatype JSON |
| :-------------: | :----------: | :-----------: | :-----------: |:-----------: | :-----------: |  :-----------: |
|  12,34 |1 | yes   | 0.1234    | 0.1234 |  string | number |
|  12,34 % |1 | yes   | 0.1234    | 0.1234 |  string | number |
|  434,27% | 1| yes   | 4.3427  | 4.3427 |string | number |
|  59,00- | 1| yes   | 0.59  | 0.59 |string | number |
|  123,45 | 1| yes   | 1.2345  | 1.2345 |string | number |
|  0,00 |  1| yes   | 0.0    | 0 |string | number |


### 3. Date Values

Konfuzio applies the so-called iso format (XXXX.XX.XX) for dates. It checks initially whether this format is already used which then won't be altered in this case. However, if another format is used, it will be adapted. In the next step, it checks if a format consisting only of a month and a year is used or even just a year without any other date information besides that.

Konfuzio recognizes written months either in German or English language and translates them into the iso format. 
                         
                              
**To give you specific examples with the outputs down below in the table::**
1) Dates, where the month is posted in a written expression with the year and the day as digits, can be transformed into the iso format without any problems.  
e.g.: 1. November 2019   
e.g.: 13 Mar 2020

2) If the year is indicated with just two digits, it will also be recognized, even if it's not separated with a sign like a dot or something similar.  
e.g.: 23.0919  
e.g.: (29.03.2018)

3) Given no information for the year, Konfuzio will assume 0000 by default.  
e.g.: /04.12.

4) If there is no information about the specific day, or even day and month, our application assumes the first day either of the respective year or year and month.  
e.g.: Oktober 2011  
e.g.: 2001

5) Date time values are translated into the iso format as well, but removing the time values:  
e.g.: 1993-02-05T00:00:00

6) Some cases can't be identified correctly or uniquely. Thus, Konfuzio can't transfer it into a date format and will return "None".  
e.g.: 14132020  
e.g.: 23.0K.2010  
e.g.: 30.07.2.90
   

| Input      | example no. | Able to convert?     | Output Excel/CSV | Output API | Datatype CSV | Datatype JSON |
| :-------------: | :----------: | :-----------: | :-----------: | :-----------: | :-----------: |  :-----------: |
|  1. November 2019 | 1| yes   | 2019-11-01   | 2019-11-01 |string | string | 
|  13 Mar 2020 | 1 | yes   | 2020-03-13    | 2020-03-13 |string | string | 
|  23.0919 | 2| yes   | 2019-09-23    | 2019-09-23 |string |string | 
|  (29.03.2018) |  2| yes   | 2018-03-29    | 2018-03-29 |string |string | 
|  /04.12. |  3 | yes   | 0000-12-04    | 0000-12-04 |string |string | 
|  Oktober 2011 | 4 | yes   | 2011-10-01    | 2011-10-01 |string |string | 
|  2001 | 4 | yes   | 2001-01-01    | 2001-01-01 | string |string | 
|  1993-02-05T00:00:00| 5 |yes  | 1993-02-05  | 1993-02-05 |string |string | 
|  14132020 |6 | no   | None    | null | - |- | 
|  23.0K.2010 |  6 |no   | None  | null |- |- | 
|  30.07.2.90 | 6 | no   | None  | null |- |- | 


### 4. Boolean values

Our application is also able to translate certain expressions into so-called boolean values, representing true or false values. This is based on certain pre-specified words. These words are representing positive or negative connotated responses with certain signal words which can be found down below in the _no_list_ and _yes_list_.


The pre-specified positive and negative (hence true and false) expressions:  
no_list = ['NEIN', 'NICHT', 'KEIN', 'OHNE', 'NO']  
yes_list = ['VORHANDEN', 'JA', 'MIT', 'YES']

Given the following examples, you can recognize how certain expressions are clustered into either "False" or "True" boolean values. If the expression is including one of the "no" or "yes" words from above, Konfuzio allocates it to True or False. However, normalization is only possible if the signal words are the first word of the annotation to avoid false positives with word combinations of words from the _no_list_ and _yes_list_, e.g. nicht vorhanden or mit ohne.

**To give you specific examples with the outputs down below in the table::** 
1) The word "nicht" ("no") will be assigned to "false".    
2) The expression "ja" ("yes") will be translated into "true".  
3) Expressions including the no or yes signal words can be translated, but only if the expression is starting with this word:    
e.g.: nicht versichert   
e.g.: not insured   
e.g.: ja versichert   
e.g.: yes insured  

4) If the expression is not starting with the signal word, it can't be translated as there is the possibility of a combination of positive and negative connotated words:  
e.g.: versichert: ja  
e.g.: insured: yes   
e.g.: versichert ja   
e.g.: insured yes   
e.g.: alleinstehend ohne Kind   

| Input      |example no. | Able to convert?     | Output Excel/CSV | Output API | Datatype CSV | Datatype JSON |
| :-------------: | :----------: | :-----------: | :-----------: |  :-----------: | :-----------: |  :-----------: |
|  nicht | 1  | yes   | false    | false | string | boolean |
|  no | 1  | yes   | false     | false |string | boolean |
|  ja | 2  | yes   | true    | true |string | boolean |
|  yes | 2  | yes   | true   | true |string | boolean |
|  nicht versichert| 3  | yes   | false    | false |string | boolean |
|  not insured | 3  | yes   | false     | false | string | boolean |
|  ja versichert | 3  | yes   | true    | true| string | boolean |
|  yes insured | 3  | yes   | true  | true | string | boolean |
|  versichert: ja | 4  | no   | None    | null | - | - |
|  insured: yes | 4  | no   | None   | null | - | - |
|  versichert ja | 4  | no   | None     | null | - | - |
|  insured yes | 4  | no   | None   | null | - | - |
|  alleinstehend ohne Kind | 4  | no  | None  | null | - | - |


### 5. Known Issues and remarks

1) Limitations to other languages besides german: Konfuzio is optimized for the german language. It may partly support English expressions but is limited in the usage in English.

2) Limitations to the case sensitivity of boolean values: Classifying certain expressions as true or false values is based on pre-specified expressions. Thus, it is limited to these signal words as shown above in the lists when normalizing the inputs.

3) The non-normalizable values which can't be converted from Konfuzio will be marked as "not machine-readable" on the Konfuzio server and can be viewed in the smart view of the documents. 
 
4) Your excel output might differ in its format from the one defined above. This is due to the default formats from excel, e.g. the default date format which might change with the version of excel of your respective country.  

## Support

To qualify for minimum response times according to your SLA, you must report the issue using the email [support@konfuzio.com](emailto:support@konfuzio.com).
