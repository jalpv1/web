from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator

def recognize_image():
        image_size = 256
        datagen = ImageDataGenerator(rescale=1. / 255)
        from numpy.random import seed
        seed(1)
        model = load_model('E:/Documents/diplomaDev/pneumonia-recogognizer-app/app/project/my_model')
        model.summary()
        one_img = datagen.flow_from_directory(
                'E:/Documents/diplomaDev/pneumonia-recogognizer-app/app/project/images',
                target_size=(image_size, image_size),
                color_mode="grayscale",
                batch_size=1,
                class_mode='binary')
        y_pred = model.predict(one_img)
        answer = y_pred > 0.5
        print(y_pred > 0.5)
        return answer

recognize_image();