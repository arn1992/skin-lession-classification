#import cv2
#img = cv2.imread("D:/AI in medicine/final_project_mobilenet/skin-cancer-mnist-ham10000/ISIC_0024306.jpg")

from PIL import Image
import cv2
import glob
#im = Image.open("D:/AI in medicine/final_project_mobilenet/skin-cancer-mnist-ham10000/ISIC_0024306.jpg")

#cv_img = []
for img in glob.glob("D:/AI in medicine/final_project_mobilenet/isic2017_2/valid/seborrheic_keratosis/*.jpg"):
    im= Image.open(img)

    #cv_img.append(n)
    width, height = im.size   # Get dimensions
    new_width=900
    new_height=700
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    cropped_im=im.crop((left, top, right, bottom))
    cropped_im.save(img)

    #cropped_im.show()



