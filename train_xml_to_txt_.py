from xml.dom import minidom

def xml_to_txt1(file_input_name,file_out_name):
	# file_input_name = "幽默数据/任务一/幽默类型任务_train.xml"
	# file_out_name = "txt文件/幽默类型任务123_train.txt"
	file_out = open(file_out_name, 'w', encoding='utf-8')
	dom = minidom.parse(file_input_name)
	root = dom.documentElement
	case = root.getElementsByTagName('Humor')
	for t in case:
		ID = t.getElementsByTagName('ID')[0].childNodes[0].data
		Contents = t.getElementsByTagName('Contents')[0].childNodes[0].data
		Class = t.getElementsByTagName('Class')[0].childNodes[0].data
		# print(Class)
		# print(ID,Contents,Class)
		file_out.write(ID+"\t"+Contents+"\t"+Class+"\n")
	file_out.close()


def xml_to_txt2(file_input_name,file_out_name):
	# file_input_name = "幽默数据/任务一/幽默类型任务_train.xml"
	# file_out_name = "txt文件/幽默类型任务123_train.txt"
	file_out = open(file_out_name, 'w', encoding='utf-8')
	dom = minidom.parse(file_input_name)
	root = dom.documentElement
	case = root.getElementsByTagName('Humor')
	for t in case:
		ID = t.getElementsByTagName('ID')[0].childNodes[0].data
		Contents = t.getElementsByTagName('Contents')[0].childNodes[0].data
		Level = t.getElementsByTagName('Level')[0].childNodes[0].data
		# print(Level)
		# print(ID,Contents,Level)
		file_out.write(ID+"\t"+Contents+"\t"+Level+"\n")
	file_out.close()


# xml_to_txt1("幽默数据/任务一/幽默类型任务_train.xml","txt文件/幽默类型任务_train.txt")
# xml_to_txt1("幽默数据/任务一/幽默类型_demo.xml","txt文件/幽默类型_test.txt")

xml_to_txt2("幽默数据/任务二/幽默等级任务_train.xml","txt文件/幽默等级任务_train.txt")
xml_to_txt2("幽默数据/任务二/幽默等级_demo.xml","txt文件/幽默等级_test.txt")
