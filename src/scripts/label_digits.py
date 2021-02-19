from glob import glob
import cv2

TRAINING_DATA_PATH = "data/training_data/digits"
LABELS_PATH = "data/training_data/labeled_digits"

def format_number(num):
    to_str = str(num)
    if num < 100:
        to_str = "0" + to_str
    if num < 10:
        to_str = "0" + to_str
    return to_str

def get_latest_img(label):
    path = f"{LABELS_PATH}/{label}"
    latest_img = glob(f"{path}/*.png")[-1]
    return int(latest_img.replace("\\", "/").split("/")[-1].split(".")[0])

def label_image(img, label):
    index = get_latest_img(label) + 1
    cv2.imwrite(f"{LABELS_PATH}/{label}/{format_number(index)}.png", img)

files = glob(f"{TRAINING_DATA_PATH}/*.png")

images_labeled = int(open(f"{TRAINING_DATA_PATH}/info.txt").readline().split("=")[1])

for filename in files[images_labeled:]:
    image = cv2.imread(filename, cv2.IMREAD_COLOR)

    cv2.namedWindow("Digit")
    cv2.imshow("Digit", image)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == ord('q'):
        break

    images_labeled += 1

    label_skipped = True
    for digit in range(10):
        if key == ord(str(digit)):
            label_image(image, digit)
            label_skipped = False
            break

    if label_skipped:
        print("Skipped label. ", end="")
    else:
        print(f"Label: '{digit}'. ", end="")

    pct_labeled = int((images_labeled / len(files)) * 100)
    print(f"Images labeled: {images_labeled}/{len(files)} ({pct_labeled}%)")

with open(f"{TRAINING_DATA_PATH}/info.txt", "w") as fp:
    fp.write(f"labeled={images_labeled}")
