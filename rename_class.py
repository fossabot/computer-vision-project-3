import fnmatch
import os

def main():
    dir = "C:\\Users\\chase\\Documents\\CMSC\\CompVision\\computer-vision-project-3\\input\\"
    for possible_class in os.listdir(dir):
        if fnmatch.fnmatch(possible_class, "new_*"):
            class_name = input("Who is this?")

            print(possible_class)

if __name__ == "__main__":
    main()
    exit()