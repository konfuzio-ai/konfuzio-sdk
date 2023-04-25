"""Run consecutively all tests."""
import os
import glob

for file in glob.iglob("docs/sdk/boilerplates/*.py"):
    if file != 'docs/sdk/boilerplates/run_all.py':
        if file == 'docs/sdk/boilerplates/test_train_label_regex_tokenizer.py':
            os.system("python " + file)
