#! /usr/bin/env python

import pandas as pd
import re


train_幽默类型 = open("txt文件/幽默类型任务_train.txt", encoding = "utf-8")
test_幽默类型 = open("txt文件/幽默类型_test.txt",encoding = "utf-8")

train = pd.read_csv(train_幽默类型, header = None, sep='\t', quoting=3, engine = "python", error_bad_lines=False)
test = pd.read_csv(test_幽默类型, header = None, sep='\t', quoting=3, engine = "python", error_bad_lines=False)
train.columns = ["ID","Contents","Class"]
test.columns = ["ID","Contents","Class"]

train["Class"] = train["Class"] .replace({"谐音":1, "谐义":2, "反转":3})
test["Class"] = test["Class"] .replace({"谐音":1, "谐义":2, "反转":3})
count = 0
for content in train["Contents"]:
    if len(content) > 0:
        # res = re.findall(r"[^\u4e00-\u9fa5]",content)
        res = re.findall(r"[a-zA-Z]", content)
        # res = re.findall(r"[0-9]", content)
        print(res)
