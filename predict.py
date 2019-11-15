from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.image import imread


##Prediction models
#load testing image
def load_image(filename,HEIGTH,WIDTH):
	img = load_img(filename, target_size=(HEIGTH, WIDTH))
	img = img_to_array(img)
	img = img.reshape(1, HEIGTH, WIDTH, 3)
	img = img.astype('float32')
	return img

#Chategorical model
def run_example(model,filename,HEIGTH,WIDTH):
    img = load_image(filename,HEIGTH,WIDTH)
    result = model.predict(img)
    image = imread(filename)
    pyplot.imshow(image)
    print(result[0])
    if result[0,1] == 1.0:
        if result[0,0] == 0.0:
            pyplot.title("dog")
        else:
            pyplot.title("cat and dog")
    else:
        if result[0,0] == 0.0:
            pyplot.title("Not cat or dog")
        else:
            pyplot.title("cat")
    pyplot.show()
