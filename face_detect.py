import cv2
import glob
import os

"""
crop and save face from images in faces folder
images will be saved in faces_resize
"""

def crop_resize_face(dirname="faces", image_type="jpg", size=(96,96)):
    fpath = os.path.join(dirname, '*.' + image_type)
    file_list = glob.glob(fpath)
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    #Make directory
    """
    directory = "faces_crop"
    if not os.path.exists(directory):
        os.mkdir(directory)
    """
    directory = "faces_resize"
    if not os.path.exists(directory):
        os.mkdir(directory)

    for i in range(len(file_list)):
        # Read the image
        image = cv2.imread(file_list[i])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.5,
            minNeighbors=5,
            minSize=(20, 20),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        # Crop and resize the faces
        for (x, y, w, h) in faces:
            sub_face = image[y:y+h, x:x+w]
            """
            crop_fname = file_list[i].replace(dirname, "faces_crop")
            cv2.imwrite(crop_fname, sub_face)
            """
            resize_face = cv2.resize(sub_face, size, interpolation = cv2.INTER_CUBIC)
            resize_fname = file_list[i].replace(dirname, "faces_resize")
            cv2.imwrite(resize_fname, resize_face)


if __name__=="__main__":
    crop_resize_face()

# cv2.imshow("Faces found", sub_face)
# cv2.waitKey(0)

# plt.imshow(cv2.cvtColor(sub_face, cv2.COLOR_BGR2RGB))
# plt.show()

