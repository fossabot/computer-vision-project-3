import fnmatch
import cv2

from config import *


def rename_class_dir(inp_dir=input_directory):
    # iterate over all classes in input folder
    for possible_class in os.listdir(inp_dir):

        # find all classes that have yet to be named
        if fnmatch.fnmatch(possible_class, "new_*"):
            combined_class_dir = os.path.join(inp_dir, possible_class)
            imgs = os.listdir(combined_class_dir)
            loaded_img = cv2.imread(os.path.join(combined_class_dir, imgs[0]))

            # Show user first face in unknown class
            cv2.imshow("WHOMSTVE IS THIS?", loaded_img)

            # hold thingy so cv2 doesn't freak out
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # User names class
            class_name = input("Who is this? ")
            new_class_dir = os.path.join(inp_dir, class_name)

            # rename unknown class to user input
            os.rename(combined_class_dir, new_class_dir)


if __name__ == "__main__":
    rename_class_dir()
    exit()
