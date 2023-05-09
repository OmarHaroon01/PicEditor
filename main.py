import customtkinter as tk
# import tkinter as tinker
from tkinter import filedialog
from tkinter.filedialog import askopenfile

import numpy as np
from PIL import Image, ImageTk
import numpy

class App(tk.CTk):

    def __init__(self):
        super().__init__()

        self.current_image = None
        self.grayscale_image = None
        self.title("PicEditor")
        # self.geometry("+%d+%d" % (self.window_start_x, self.window_start_y))
        self.minsize(1000, 600)
        # self.geometry("1000x600")
        self.resizable(False, False)

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

        blur_button = tk.CTkButton(master=left_frame, command=self.button_callback, text="Blur")
        blur_button.grid(row=1, column=0, padx="10px", pady=(40, 0))

        sketch_button = tk.CTkButton(master=left_frame, command=self.button_callback, text="Sketch")
        sketch_button.grid(row=1, column=1, padx="10px", pady=(40, 0))

        edge_detection_button = tk.CTkButton(master=left_frame, command=self.button_callback, text="Edge Detection")
        edge_detection_button.grid(row=2, column=0, padx="10px", pady=(15, 0))

        photocopy_button = tk.CTkButton(master=left_frame,
                                        command=lambda: self.photocopy_button_clicked(int(self.slider.get())),
                                        text="Photocopy")
        photocopy_button.grid(row=2, column=1, padx="10px", pady=(15, 0))

        erosion_button = tk.CTkButton(master=left_frame, command=self.button_callback, text="Erosion")
        erosion_button.grid(row=3, column=0, padx="10px", pady=(15, 0))

        dilation_button = tk.CTkButton(master=left_frame, command=self.button_callback, text="Dilation")
        dilation_button.grid(row=3, column=1, padx="10px", pady=(15, 0))

        grayscale_button = tk.CTkButton(master=left_frame, command=self.grayscale_button_clicked, text="Grayscale")
        grayscale_button.grid(row=4, column=0, padx="10px", pady=(15, 0))

        mirror_button = tk.CTkButton(master=left_frame, command=lambda: self.mirror_button_clicked(), text="Mirror")
        mirror_button.grid(row=4, column=1, padx="10px", pady=(15, 0))

        negative_button = tk.CTkButton(master=left_frame, command=lambda: self.negative_button_clicked(), text="Negative")
        negative_button.grid(row=5, column=0, padx="10px", pady=(15, 0))

    def mirror_button_clicked(self):
        img_data = np.array(self.current_image)
        img = Image.fromarray(np.uint8(np.flip(img_data, axis=1)))
        img.thumbnail((700, 550))
        show_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=show_img)
        self.img_label.image = show_img

    def negative_button_clicked(self):
        img_data = np.array(self.current_image)
        max_val = np.max(img_data)
        img_data[:, :, :] = max_val - img_data[:, :, :]
        img = Image.fromarray(np.uint8(img_data))
        img.thumbnail((700, 550))
        show_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=show_img)
        self.img_label.image = show_img

    def double2img(self, img_arr):
        return np.round(img_arr * 255)

    def img2double(self, img_arr):
        return img_arr / 255.0

    def photocopy_button_clicked(self, threshold):
        if self.grayscale_image is None:
            return
        grayscale_arr = self.img2double(np.array(self.grayscale_image))
        threshold /= 255.0
        # threshold = 100.0 / 255.0
        grayscale_arr[grayscale_arr > threshold] = 1.0
        grayscale_arr[grayscale_arr <= threshold] = (grayscale_arr[grayscale_arr <= threshold] *
                                                     (threshold - grayscale_arr[grayscale_arr <= threshold])) / (
                                                                threshold * threshold)
        grayscale_arr = self.double2img(grayscale_arr)
        img = Image.fromarray(grayscale_arr)
        img.thumbnail((700, 550))
        show_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=show_img)
        self.img_label.image = show_img

    def grayscale_button_clicked(self):
        img = self.grayscale_image.copy()
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
        img = self.current_image.copy()
        img.thumbnail((700, 550))
        show_img = ImageTk.PhotoImage(img)
        rgb_img_data = np.array(self.current_image)
        r, g, b = rgb_img_data[:, :, 0], rgb_img_data[:, :, 1], rgb_img_data[:, :, 2]
        grayscale_arr = 0.2989 * r + 0.5870 * g + 0.1140 * b
        self.grayscale_image = Image.fromarray(grayscale_arr)
        self.img_label.configure(image=show_img)
        self.img_label.image = show_img


if __name__ == "__main__":
    app = App()
    app.mainloop()
