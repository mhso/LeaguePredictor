from glob import glob
import random
import cv2
from classifier import digit_dataset
from classifier.digit_classifier import DigitClassifier

if __name__ == "__main__": # Sanity check.
    digit_classifier = DigitClassifier()
    digit_classifier.load()

    base_path = "data/training_data/digits"
    files = glob(base_path + "/*.png")
    images = 20
    for image in range(images):
        rand_index = random.randint(0, len(files)-1)
        image = cv2.imread(files[rand_index], cv2.IMREAD_COLOR)

        reshaped = digit_dataset.shape_input(image)
        label = digit_classifier.predict(reshaped)

        print(f"Predicted label: {label}", flush=True)
        bigger_img = cv2.resize(image, tuple(x * 2 for x in image.shape[:2]))
        cv2.imshow("Digit", bigger_img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
