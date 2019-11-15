import random
from shutil import copyfile
from os import listdir, makedirs, walk, path

#Format choosen: format contains all the images within separate folders named
#after their respective class names
def split(dir, train_size, seed):
     # Set up empty folder structure for training and validation sets
    if not path.exists(dir +'train'):
        makedirs(dir + 'train')
    if not path.exists(dir + 'validation'):
        makedirs(dir + 'validation')

    #create subdirectories of the 2 classes
    subdirs = [subdir for subdir in listdir(dir) if not subdir.endswith(".jpg")]
    for subdir in subdirs:
        # create label subdirectories
        if not path.exists(dir + subdir + '/' +'dogs'):
            makedirs(dir + subdir +'/dogs')
        if not path.exists(dir + subdir + '/' +'cats'):
            makedirs(dir + subdir +'/cats')

    #copy images to the subdirectories of the 2 classes
    train_counter = 0
    validation_counter = 0
    random.seed(seed)
    for filename in listdir(dir):

            if filename.endswith(".jpg"):
                if random.uniform(0, 1) <= train_size:
                    #print(random.uniform(0, 1))
                    dir_dest =  dir + "train"
                    train_counter += 1
                else:
                    dir_dest =  dir + "validation"
                    validation_counter += 1
                src =  dir + filename
                if filename.startswith('cat'):
                	dst = dir_dest + '/cats/'  + filename
                	copyfile(src, dst)
                elif filename.startswith('dog'):
                	dst = dir_dest + '/dogs/'  + filename
                	copyfile(src, dst)

    print('Copied ' + str(train_counter) + ' images to train')
    print('Copied ' + str(validation_counter) + ' images to validation')
