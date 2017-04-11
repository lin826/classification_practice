import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, isdir, join

PATH = "../demo"
BMP_FILE = "../tmp/data_demo.npy"

def get_bmp_data(file_path):
	# img = Image.open("../Data_Train/Class1/faceTrain1_1.bmp")
	img = Image.open(file_path)
	result = np.array([img.getpixel((i,j)) for j in range(img.size[1]) for i in range(img.size[0])])
	img.close()
	return result

def get_training_files(folder_path):
	onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
	return onlyfiles

def get_class_path(folder_path):
	return [f for f in listdir(folder_path) if isdir(join(folder_path, f))]

def get_training_data(path):
	result = list()
	for folder_path in get_class_path(path):
		class_list = list()
		for file_path in get_training_files(join(path,folder_path)):
			file_name = join(path,folder_path,file_path)
			class_list.append(get_bmp_data(file_name))
		result.append(np.array(class_list))
	return np.array(result)
def get_demo_data(path):
	result = list()
	for file_path in get_training_files(path):
		file_name = join(path,file_path)
		result.append(get_bmp_data(file_name))
	return np.array(result)

def convert_data():
	data = get_training_data(PATH)
	np.save(join(PATH,BMP_FILE), data)

def get_data():
	return np.load(join(PATH,BMP_FILE))

def save_item(item,f_path):
	np.save(f_path, item)

def get_item(f_path):
	return np.load(f_path)
### Convert all training data into Data.npy
# convert_data()
# print(get_data())
data = get_demo_data(PATH)
np.save(join(PATH,BMP_FILE), data)
