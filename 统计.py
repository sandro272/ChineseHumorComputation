#! /usr/bin/env python

import numpy as np
import pandas as pd
from collections import Counter


train_幽默类型 = open("txt文件/幽默类型任务_train.txt", encoding = "utf-8")
test_幽默类型 = open("txt文件/幽默类型_test.txt",encoding = "utf-8")


train = pd.read_csv(train_幽默类型, header = None, sep='\t', quoting=3, engine = "python", error_bad_lines=False)
test = pd.read_csv(test_幽默类型, header = None, sep='\t', quoting=3, engine = "python", error_bad_lines=False)
train.columns = ["ID","Contents","Class"]
test.columns = ["ID","Contents","Class"]

train["Class"] = train["Class"] .replace({"谐音":0, "谐义":1, "反转":2})
test["Class"] = test["Class"] .replace({"谐音":0, "谐义":1, "反转":2})

'''
train_humous_rank = open("txt文件/幽默等级任务_train.txt", encoding = "utf-8")
test_humous_rank = open("txt文件/幽默等级_test.txt",encoding = "utf-8")

train = pd.read_csv(train_humous_rank, header = None, sep='\t', quoting=3, engine = "python", error_bad_lines=False)
test = pd.read_csv(test_humous_rank, header = None, sep='\t', quoting=3, engine = "python", error_bad_lines=False)

train.columns = ["ID","Contents","Level"]
test.columns = ["ID","Contents","Level"]

train["Level"] = train["Level"] .replace({1:0, 5:1,})
test["Level"] = test["Level"] .replace({1:0, 5:1, })
'''
res1 = train["Class"]
res2 = test["Class"]



def counter(arr):
    """获取每个元素的出现次数，使用标准库collections中的Counter方法"""
    return Counter(arr).most_common(3) # 返回出现频率最高的两个数


def single_list(arr, target):
    """获取单个元素的出现次数，使用list中的count方法"""
    return arr.count(target)


def all_list(arr):
    """获取所有元素的出现次数，使用list中的count方法"""
    result = {}
    for i in set(arr):
        result[i] = arr.count(i)
    return result


def single_np(arr, target):
    """获取单个元素的出现次数，使用Numpy"""
    arr = np.array(arr)
    mask = (arr == target)
    arr_new = arr[mask]
    return arr_new.size


def all_np(arr):
    """获取每个元素的出现次数，使用Numpy"""
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result


if __name__ == "__main__":
    print(counter(res1))
    print(counter(res2))
    # print(single_list(res2, 2))
    # print(all_list(array))
    # print(single_np(array, 2))
    # print(all_np(array))
