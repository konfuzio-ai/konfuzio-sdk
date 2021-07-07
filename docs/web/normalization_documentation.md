### [draft] Documentation Data Normalization
***
Our Konfuzio application is able to normalize various data formats for different data types used in the annotations of the documents. It translates different formats of numbers, percentages, date and boolean values into a unique format. Down below you find an overview of the normalizations of the different data types with specific examples.

### 1. Numbers
---
Konfuzio is able to recognize numbers encoded in a written format to translate them into a data type that represents machine-readable numbers (i.e. float). Numbers from one to twelve are transformed, either from german or english language.
Given an annotation as an offset-string, this annotation will be translated into a (absolute) number.  
Starting with the first case, a dash or a series of dashs which also might include commas or decimal points will be recognized as Zero.   
For the specific context not relevant signs, like quotation marks or whitespaces, are completely removed form the annotation. If the annotation contains the letter "S" or "H" after the digits, it will also be removed.  

Negative expressions are recognized either by dashes in front or after the digits, the letter "S" after the digits or the number being framed into brackets. 
Negative or positive signs are removed, as well as whitespaces or quotation marks.  

As there are varying ways to display float numbers, e.g. by using dots instead of commas to display the thousands mark or decimal numbers, Konfuzio also takes care of this to display it in a uniform format. 
The uniform chosen format uses a the english/american standard. Hence, dots are used as a separation for decimal numbers and commas to mark the thousands. 

**To give you specific examples:**  
1) Negative signs and zeros as after digits will be removed:  
e.g.: 59,00- = 59  
e.g.: 786,71- = 786.71  

2) Irrelevant signs and whitespaces will be removed and it will be transformed into the unique format with a dot instead of a comma as the decimal separator and an absolute number:  
e.g.: :2.000, 08 = 2000.08  
e.g.: -2.759,7° = 2759.7  
e.g.: +159,;03 = 159.03  
e.g.: €1.010.296 = 1010296

3) Written numbers will be changed into a digits number format:  
e.g.: ein = 1  
e.g.: eleven = 11

4) The expression "Nil" standing for "nothing" will be translated into 0:  
e.g.: NIL = 0 

```test_data_numbers = [
    ('59,00-', 59, 50945),
    ("'786,71-", 786.71, 51429),
    (':2.000, 08 ', 2000.08, 51437),
    ('-2.759,7°', 2759.7, 51447),
    ('+159,;03', 159.03, 56253),
    ('€1.010.296', 1010296, 93255),
    ('ein', 1, 99479),
    ('eleven', 11, None),
    ('–100', 100, 109610),
    ('NIL', 0, None)
]
```


### 2. Percentage Numbers
---
Konfuzio also handels percentage numbers of different formats and brings them into a machine-readable one. Percentage expressions will not be displayed in the classic percentage format with the percentage sign %, but in the decimal number format _out of 1_ without the percentage sign, but with a dot as a decimal separator and 4 decimal places. 

**To give you specific examples:**  
1) Digits which are separated by ccommas with two decimal places will be easily converted into an uniform format, either with or without the percentage sign in its original format.  
e.g.: 12,34 = 0.1234  
e.g.: 12,34 % = 0.1234  
e.g.: 434,27% = 4.3427  
e.g.: 0,00 = 0

```test_data_percentage = [
    ('12,34', 0.1234, None),
    ('12,34 %', 0.1234, None),
    ('434,27%,', 4.3427, None),
    ('0,00', 0, None)
]
```

### 3. Date Values
---
Konfuzio applies the so-called iso format (XXXX.XX.XX) for dates. It checks initially whether this format is already used which then won't be altered in this case. However, if another format is used, it will be adapted. In the next step, it checks if a format consisting only of a month and a year is used or even just a year without any other date information besides that.

Konfuzio recognizes written months either in german or english language and translates them into the iso format. 

```# written months to be replaced with the according numbers in german/english:
month_dict_de_en = {'JANUAR': '01.', 'JANUARY': '01.', 'FEBRUAR': '02.', 'FEBRUARY': '02.', 'MÄRZ': '03.',
                        'MARCH': '03.', 'APRIL': '04.', 'MAI': '05.', 'MAY': '05.', 'JUNI': '06.', 'JUNE': '06.',
                        'JULI': '07.', 'JULY': '07.', 'AUGUST': '08.', 'SEPTEMBER': '09.',
                        'OKTOBER': '10.', 'NOVEMBER': '11.', 'DEZEMBER': '12.', 'DECEMBER': '12.'}

month_dict_short_eng_de = {'JAN': '01.', 'FEB': '02.', 'MAR': '03.', 'APR': '04.', 'MAY': '05.',
                               'JUN': '06.', 'JUL': '07.', 'AUG': '08.', 'SEP': '09.',
                               'OCT': '10.', 'OKT': '10.', 'NOV': '11.', 'DEC': '12.', 'DEZ': '12.'}
```
                              
                              
**To give you specific examples:**
1) Dates where the month is posted in a written expression with the year and the day as digits can be transformed into the iso format without any problems.  
e.g.: 1. November 2019/1st November of 2019 = 2019-11-01 

2) If the year is indicated with just two digits, it will also be recognized, even if it's not separated with a sign like a dot or something similar.  
e.g.: 23.0919 = 2019-09-23

3) Given no information for the year, Konfuzio will assume 0000.  
e.g.: /04.12. = 0000-12-04

4) If there is no information about the specific day, or even day and month, our application assumes the first day either of the respective year or year and month.  
e.g.: Oktober 2011 = 2011-10-01  
e.g.: 2001 = 2001-01-01

3) Some cases can't be identified correctly or uniquely. Thus, Konfuzio can't transfer it into a date format and will return "None".  
e.g.: 14132020 = None  
e.g.: 23.0K.2010 = None

```test_data_dates = [
('1. November 2019', '2019-11-01', 51453),
('23.0919', '2019-09-23', 51449),
('/04.12.', '0000-12-04', 47776),
('Oktober 2011', '2011-10-01', 0),
('2001', '2001-01-01', 0),
('14132020', None, 0),
('23.0K.2010', None, 0)
]
```


### 4. Boolean values
---   
Our application is also able to translate certain expressions into a boolean value, representing true or false values. This is based on certain pre-specified words. These words are representing positive or negative connotated responses with certain signal words which can be found down below in the _no_list_ and _yes_list_.

_no english expressions here?_

the pre-specified positive and negative (hence true and false) expressions:
no_list = ['NEIN', 'NICHT', 'KEIN', 'OHNE', 'NO']
yes_list = ['VORHANDEN', 'JA', 'MIT', 'YES']

Given the following examples, you can recognize how certain expressions are clustered into either "False" or "True" boolean values. If the expression is including one of the "no" or "yes" words from above, Konfuzio allocates it to True or False.  

**To give you specific examples:** 
1) The sentence "Hierfür sind sie leider nicht versichert." ("Unfortunately, you're not insured for this.") will be assigned to "False".  
2) "Lieferung inkludiert: ja" ("Delivery included: yes") will be translated into "True".
3) Empty expressions, like "", can't be translated and Konfuzio will return "None". 

```test_data_bool = [
    ('nicht versichert', False, 0),
    ('ja', True, 0),
    ('', None, 0),
]
```