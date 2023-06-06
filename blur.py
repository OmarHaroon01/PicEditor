import customtkinter as tk
from tkinter import filedialog

import math
from cv2 import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy.ndimage import sobel

import torch
import torch.nn as nn


def double2img(img_arr):
    return np.round(img_arr * 255)


def img2double(img_arr):
    return img_arr / 255.0


def mat2gray(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = ((image - min_val) * 255) / (max_val - min_val)
    return normalized_image


class App(tk.CTk):

    def __init__(self):
        super().__init__()

        self.original_image = None
        self.current_image = None
        self.title("PicEditor")
        # self.geometry("+%d+%d" % (self.window_start_x, self.window_start_y))
        self.minsize(1000, 600)
        # self.geometry("1000x600")
        self.resizable(False, False)
        self.IPrevImage = self.current_image
        self.PrevOperation = 'None'
        self.vigneT = 0

        left_frame = tk.CTkFrame(self, width=250, height=600)
        left_frame.pack(side="left", fill="both")
        left_frame.pack_propagate(False)
        right_frame = tk.CTkFrame(self)
        right_frame.pack(expand="True", padx="25px", pady="25px")

        self.img_label = tk.CTkLabel(master=right_frame, text="", bg_color="transparent", fg_color=None)
        self.img_label.pack(anchor="center", expand="True")

        slider_frame = tk.CTkFrame(master=left_frame)
        slider_frame.pack(side="bottom", pady="20px")

        self.slider = tk.CTkSlider(master=slider_frame, from_=1, to=250, command=self.slider_event, number_of_steps=250)
        self.slider.set(1)
        self.slider.pack(anchor="center", side="bottom")

        self.slider_value_label = tk.CTkLabel(master=slider_frame, text=int(self.slider.get()))
        self.slider_value_label.pack(anchor="center", side="top")

        upload_button = tk.CTkButton(master=left_frame, command=lambda: self.upload_button_clicked(),
                                     text="Upload Image")
        upload_button.grid(row=0, column=0, columnspan=2, padx="50px", pady=(40, 0))

        blur_button = tk.CTkButton(master=left_frame,
                                   command=lambda: self.blurr_button_clicked(int(self.slider.get())),
                                   text="Blur")
        blur_button.grid(row=1, column=0, padx="10px", pady=(40, 0))

        sketch_button = tk.CTkButton(master=left_frame, command=lambda: self.sketch_button_clicked(), text="Sketch")
        sketch_button.grid(row=1, column=1, padx="10px", pady=(40, 0))

        edge_detection_button = tk.CTkButton(master=left_frame, command=self.button_callback, text="Edge Detection")
        edge_detection_button.grid(row=2, column=0, padx="10px", pady=(15, 0))

        photocopy_button = tk.CTkButton(master=left_frame,
                                        command=lambda: self.photocopy_button_clicked(int(self.slider.get())),
                                        text="Photocopy")
        photocopy_button.grid(row=2, column=1, padx="10px", pady=(15, 0))

        erosion_button = tk.CTkButton(master=left_frame, command=lambda: self.contrast_button_clicked(), text="Erosion")
        erosion_button.grid(row=3, column=0, padx="10px", pady=(15, 0))

        dilation_button = tk.CTkButton(master=left_frame, command=self.button_callback, text="Dilation")
        dilation_button.grid(row=3, column=1, padx="10px", pady=(15, 0))

        grayscale_button = tk.CTkButton(master=left_frame, command=self.grayscale_button_clicked, text="Grayscale")
        grayscale_button.grid(row=4, column=0, padx="10px", pady=(15, 0))

        mirror_button = tk.CTkButton(master=left_frame, command=lambda: self.mirror_button_clicked(), text="Mirror")
        mirror_button.grid(row=4, column=1, padx="10px", pady=(15, 0))

        negative_button = tk.CTkButton(master=left_frame, command=lambda: self.negative_button_clicked(),
                                       text="Negative")
        negative_button.grid(row=5, column=0, padx="10px", pady=(15, 0))

        vignetting_button = tk.CTkButton(master=left_frame, command=lambda: self.vignetting_button_clicked(),
                                         text='Vignetting')
        vignetting_button.grid(row=5, column=1, padx='10px', pady=(15, 0))

        sepia_button = tk.CTkButton(master=left_frame, command=lambda: self.sepia_button_clicked(),
                                    text="Sepia")
        sepia_button.grid(row=6, column=0, padx="10px", pady=(15, 0))

        night_vision_button = tk.CTkButton(master=left_frame, command=lambda: self.night_vision_button_clicked(),
                                           text='Night Vision')
        night_vision_button.grid(row=6, column=1, padx='10px', pady=(15, 0))

    def blurr_button_clicked(self, scale):
        scale = math.ceil(scale)
        n = 2 * scale + 1

        if self.PrevOperation == "Blur":
            img_data = np.array(self.IPrevImage)
        else:
            img_data = np.array(self.current_image)

        blurI = cv2.blur(img_data, (n, n),
                         dst=None,
                         anchor=None,
                         borderType=None)

        img = Image.fromarray(blurI)
        self.PrevOperation = "Blur"
        self.IPrevImage = img_data

        self.current_image = img
        img.thumbnail((700, 550))
        show_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=show_img)
        self.img_label.image = show_img

    def sepia_button_clicked(self):
        self.PrevOperation == "Sepia"
        img_data = np.array(self.current_image)
        img = img_data

        if img_data.ndim == 2:
            return

        if img_data.ndim == 3:
            r, g, b = img_data[:, :, 0], img_data[:, :, 1], img_data[:, :, 2]
            r1 = (0.393 * r + 0.769 * g + 0.189 * b)
            g1 = (0.349 * r + 0.686 * g + 0.168 * b)
            b1 = (0.272 * r + 0.534 * g + 0.131 * b)
            r1[r1 > 255] = 255
            g1[g1 > 255] = 255
            b1[b1 > 255] = 255
            img[:, :, 0] = r1
            img[:, :, 1] = g1
            img[:, :, 2] = b1

        img = Image.fromarray(img.astype(np.uint8))
        self.current_image = img
        img.thumbnail((700, 550))
        show_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=show_img)
        self.img_label.image = show_img

    def sketch_button_clicked(self):
        img_data = np.array(self.current_image)
        self.PrevOperation == "Sketch"
        self.IPrevImage = self.current_image

        if img_data.ndim <= 2:
            img = img_data
        if img_data.ndim == 3:
            gImg = 0.299 * img_data[:, :, 0] + 0.587 * img_data[:, :, 1] + 0.114 * img_data[:, :, 2]
            gradient_x = sobel(gImg, axis=1)
            gradient_y = sobel(gImg, axis=0)
            gradient_magnitude = np.hypot(gradient_x, gradient_y)
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 255
            gradient_magnitude = 255 - gradient_magnitude

            img = gradient_magnitude

        img = Image.fromarray(img.astype(np.uint8))
        self.current_image = img
        img.thumbnail((700, 550))
        show_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=show_img)
        self.img_label.image = show_img

    def contrast_button_clicked(self):
        self.IPrevImage = self.current_image
        self.PrevOperation == "Contrast"
        img_data = np.array(self.current_image)
        self.IPrevImage = self.current_image
        n = np.min(img_data)
        m = np.max(img_data)
        img = ((img_data - n) / (m - n)) * 255
        img = Image.fromarray(img.astype(np.uint8))
        self.current_image = img
        img.thumbnail((700, 550))
        show_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=show_img)
        self.img_label.image = show_img

    def vignetting_button_clicked(self):
        self.IPrevImage = self.current_image
        self.PrevOperation = 'Vignetting'

        img_data = np.array(self.current_image)
        n = len(img_data)  # no of row
        m = len(img_data[0])  # no of column
        distM = np.ones(shape=(n, m))
        an = math.floor(n / 2)
        am = math.floor(m / 2)
        scale = 1
        D = math.sqrt(an ** 2 + am ** 2)
        distM = np.fromfunction(lambda i, j: scale * np.sqrt((abs(j - am) ** 2) + (abs(i - an) ** 2)) / D,
                                img_data.shape[:2])

        finalD = 1 - distM

        if img_data.ndim == 3:
            finalD = np.dstack([finalD] * 3)

        finalD = np.abs(finalD)
        np.clip(finalD, 0, 255, out=finalD)
        img = img_data * finalD

        img = Image.fromarray(img.astype(np.uint8))
        self.current_image = img
        img.thumbnail((700, 550))
        show_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=show_img)
        self.img_label.image = show_img

    def night_vision_button_clicked(self):
        self.IPrevImage = self.current_image
        self.PrevOperation = 'Night Vision'

        img_data = np.array(self.current_image)
        img = img_data
        if img_data.ndim == 3:
            g = img_data[:, :, 1]
            r1 = g / 2
            b1 = 2 * r1
            g1 = 2 * b1
            r1[r1 > 255] = 255
            g1[g1 > 255] = 255
            b1[b1 > 255] = 255
            img[:, :, 0] = r1
            img[:, :, 1] = g1
            img[:, :, 2] = b1

        img = Image.fromarray(img.astype(np.uint8))
        self.current_image = img
        img.thumbnail((700, 550))
        show_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=show_img)
        self.img_label.image = show_img

    def mirror_button_clicked(self):
        self.IPrevImage = self.current_image
        self.PrevOperation = 'Mirror'
        img_data = np.array(self.current_image)
        self.current_image = Image.fromarray(np.uint8(np.flip(img_data, axis=1)))
        img = self.current_image.copy()
        img.thumbnail((700, 550))
        show_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=show_img)
        self.img_label.image = show_img

    def negative_button_clicked(self):
        self.IPrevImage = self.current_image
        self.PrevOperation = 'Neg'

        img_data = np.array(self.current_image)
        if img_data.ndim == 3:
            max_val = np.max(img_data)
            img_data[:, :, :] = max_val - img_data[:, :, :]
            self.current_image = Image.fromarray(np.uint8(img_data))
            img = self.current_image.copy()
            img.thumbnail((700, 550))
            show_img = ImageTk.PhotoImage(img)
            self.img_label.configure(image=show_img)
            self.img_label.image = show_img
        else:
            img_data = 255 - img_data
            self.current_image = Image.fromarray(np.uint8(img_data))
            img = self.current_image.copy()
            img.thumbnail((700, 550))
            show_img = ImageTk.PhotoImage(img)
            self.img_label.configure(image=show_img)
            self.img_label.image = show_img

    def photocopy_button_clicked(self, threshold):
        self.IPrevImage = self.current_image

        if self.PrevOperation == "Photocopy":
            img_data = np.array(self.IPrevImage)
        else:
            img_data = np.array(self.current_image)

        # img_data = np.array(self.current_image)
        if img_data.ndim == 3:
            self.grayscale_button_clicked()
        # img_data = np.array(self.current_image)
        grayscale_arr = img2double(img_data)
        threshold /= 255.0
        grayscale_arr[grayscale_arr > threshold] = 1.0
        grayscale_arr[grayscale_arr <= threshold] = (grayscale_arr[grayscale_arr <= threshold] *
                                                     (threshold - grayscale_arr[grayscale_arr <= threshold])) / (
                                                            threshold * threshold)
        grayscale_arr = double2img(grayscale_arr)
        print(grayscale_arr.shape)
        self.PrevOperation = "Photocopy"
        self.IPrevImage = img_data
        self.current_image = Image.fromarray(grayscale_arr.astype(np.uint8))
        img = self.current_image.copy()
        img.thumbnail((700, 550))
        show_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=show_img)
        self.img_label.image = show_img

    def grayscale_button_clicked(self):
        self.PrevOperation = "Gray"
        rgb_img_data = np.array(self.current_image)
        if rgb_img_data.ndim == 2:
            return
        r, g, b = rgb_img_data[:, :, 0], rgb_img_data[:, :, 1], rgb_img_data[:, :, 2]
        grayscale_arr = 0.2989 * r + 0.5870 * g + 0.1140 * b
        self.current_image = Image.fromarray(grayscale_arr)
        img = self.current_image.copy()
        img.thumbnail((700, 550))
        show_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=show_img)
        self.img_label.image = show_img

    def slider_event(self, _event=None):
        self.slider_value_label.configure(text=int(self.slider.get()))
        print(self.slider.get())

    def button_callback(self):
        print("Button Clicked")

    def upload_button_clicked(self):
        f_types = [('Jpg Files', '*.jpg')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        self.current_image = Image.open(filename)
        self.original_image = Image.open(filename)
        self.IPrevImage = self.current_image
        self.PrevOperation = 'None'
        self.vigneT = 0

        img = self.current_image.copy()
        img.thumbnail((700, 550))
        show_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=show_img)
        self.img_label.image = show_img


if __name__ == "__main__":
    app = App()
    app.mainloop()
