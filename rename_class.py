"""
File: rename_class.py
Created by Andrew Ingson & Chase Carbaugh (aings1@umbc.edu & chasec1@umbc.edu)
Date: 5/9/2021
CMSC 491 Special Topics - Computer Vision

"""

import fnmatch
import tkinter as tk
from tkinter import simpledialog

import cv2

from config import *


def rename_class_dir(inp_dir=input_directory, ignore_dir=""):
    # create framing for the input dialog box
    prompt_window = tk.Tk()
    prompt_window.withdraw()
    # iterate over all classes in input folder
    for possible_class in os.listdir(inp_dir):
        if possible_class == ignore_dir:
            continue

        # find all classes that have yet to be named
        if fnmatch.fnmatch(possible_class, "new_*"):
            combined_class_dir = os.path.join(inp_dir, possible_class)
            imgs = os.listdir(combined_class_dir)
            loaded_img = cv2.imread(os.path.join(combined_class_dir, imgs[0]))
            loaded_img = cv2.resize(loaded_img, (300, 300), interpolation=cv2.INTER_CUBIC)
            # Show user first face in unknown class
            cv2.imshow("Current Face", loaded_img)

            # hold thingy so cv2 doesn't freak out
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # User names class
            try:
                class_name = simpledialog.askstring(title="Rename Class", prompt="Who is this?")
                new_class_dir = os.path.join(inp_dir, class_name)
            except TypeError:
                continue

            # rename unknown class to user input
            os.rename(combined_class_dir, new_class_dir)


if __name__ == "__main__":
    rename_class_dir()
    exit()
