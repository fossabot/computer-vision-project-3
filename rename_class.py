import fnmatch
import cv2

from config import *


def rename_class_dir(inp_dir=input_directory):
    # TODO: Add comments
    for possible_class in os.listdir(inp_dir):
        if fnmatch.fnmatch(possible_class, "new_*"):
            combined_class_dir = os.path.join(inp_dir, possible_class)
            imgs = os.listdir(combined_class_dir)
            loaded_img = cv2.imread(os.path.join(combined_class_dir, imgs[0]))
            cv2.imshow("normalized float64", loaded_img)
            # hold thingy so cv2 doesn't freak out
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            class_name = input("Who is this? ")
            new_class_dir = os.path.join(inp_dir, class_name)
            os.rename(combined_class_dir, new_class_dir)
            print(possible_class)

if __name__ == "__main__":
    main()
    exit()