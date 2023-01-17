## Architecture overview

we'll take a closer look at the most important ones and give you some examples of how they can be implemented 
consequentially or individually in case you want to experiment. 

The first step we're going to cover is [File Splitting](https://dev.konfuzio.com/sdk/examples/examples.html#splitting-for-multi-file-documents-step-by-step-guide) 
– this happens when the original Document consists of several smaller sub-Documents and needs to be separated 
 so that each one can be processed individually.

Second part is on [Categorization](https://dev.konfuzio.com/sdk/examples/examples.html#document-categorization), where a Document is labelled to be of a certain Category within the Project. 

Third part describes [Information Extraction](https://dev.konfuzio.com/sdk/examples/examples.html#train-a-konfuzio-sdk-model-to-extract-information-from-payslip-documents), during which various information is obtained from unstructured texts, 
i.e. Name, Date, Recipient, or any other custom Labels.

For a more in-depth look at each step, be sure to check out the 
[diagram](https://dev.konfuzio.com/sdk/contribution.html#architecture-sdk-to-server) that reflects each step of the 
document-processing pipeline.