import os
import xml.etree.ElementTree as ET

path = 'C:/Users/Patrick/Desktop/darkflow/raccoons/images/'

ann_path = 'C:/Users/Patrick/Desktop/darkflow/raccoons/annotations/'

list_of_anns = os.listdir('C:/Users/Patrick/Desktop/darkflow/raccoons/annotations/')

for filename in list_of_anns:
	tree = ET.parse(ann_path + filename)
	tree.find('path').text = path+filename
	# a.text = "BEEP"
	tree.write(ann_path + filename)

	#print(a.text)
#print(list_of_anns)