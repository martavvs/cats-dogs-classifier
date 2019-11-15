from keras.preprocessing.image import ImageDataGenerator

from model import def_model
from metrics import summarize_diagnostics


def train_model(model_name, train_dir,HEIGTH,WIDTH,BATCH_SIZE,EPOCHS,
            validation_dir, train=True):
    model = def_model()
    datagenerat = ImageDataGenerator(rescale=1./255)
    #iterator
    if train:
        train_generator = datagenerat.flow_from_directory(
            train_dir,
            target_size=(HEIGTH, WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical')

        model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=EPOCHS, verbose=0 )
    else:
        validation_generator = datagenerat.flow_from_directory(
            validation_dir, # same directory as training data
            target_size=(HEIGTH, WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical')

        history = model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            verbose=0)
        #Plots
        summarize_diagnostics(history, args)

    model.save(model_name+'.h5')
