#! /usr/bin/env python
from xml.dom import minidom

def xml_to_txt(file_input_name,file_out_name):
	# file_input_name = "幽默数据/任务一/幽默类型任务_train.xml"
	# file_out_name = "txt文件/幽默类型任务123_train.txt"
	file_out = open(file_out_name, 'w', encoding='utf-8')
	dom = minidom.parse(file_input_name)
	root = dom.documentElement
	case = root.getElementsByTagName('Humor')
	for t in case:
		ID = t.getElementsByTagName('ID')[0].childNodes[0].data
		Contents = t.getElementsByTagName('Contents')[0].childNodes[0].data
		# Class = t.getElementsByTagName('Class')[0].childNodes[0].data
		# print(Class)
		# print(ID,Contents,Class)
		file_out.write(ID+"\t"+Contents+"\n")
	file_out.close()



xml_to_txt("中文幽默计算/任务一___幽默类型识别_test.xml","txt文件/任务一___幽默类型识别_test.txt")
xml_to_txt("中文幽默计算/任务二___幽默等级划分_test.xml","txt文件/任务二___幽默等级划分_test.txt")
