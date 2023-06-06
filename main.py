import customtkinter as tk
from tkinter import filedialog
from scipy import ndimage
from scipy.ndimage import convolve
from scipy.ndimage import sobel

from scipy import misc

import math
from cv2 import cv2
import numpy as np
from PIL import Image, ImageTk


def double2img(img_arr):
    return np.round(img_arr * 255)


def img2double(img_arr):
    return img_arr / 255.0


def mat2gray(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = ((image - min_val) * 255) / (max_val - min_val)
    return normalized_image


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x = np.arange(-size, size + 1)
    y = np.arange(-size, size + 1)
    x, y = np.meshgrid(x, y)
    exponent = -(x ** 2 + y ** 2) / (2.0 * sigma ** 2)
    g = np.exp(exponent)
    g /= (2.0 * np.pi * sigma ** 2)
    return g


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.convolve(img, Kx)
    Iy = ndimage.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return G, theta


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle = np.where(angle < 0, angle + 180, angle)

    # Precompute trigonometric values
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # Compute pixel indices
    i, j = np.indices((M, N))

    # Find neighboring pixel indices
    offset_x = np.round(cos_angle).astype(int)
    offset_y = np.round(sin_angle).astype(int)

    # Find neighboring pixels
    q = img[np.clip(i + offset_y, 0, M - 1), np.clip(j + offset_x, 0, N - 1)]
    r = img[np.clip(i - offset_y, 0, M - 1), np.clip(j - offset_x, 0, N - 1)]

    # Check if pixel is local maximum
    is_local_max = np.logical_and(img >= q, img >= r)
    Z = np.where(is_local_max, img, 0)

    return Z


def threshold_func(img):
    highThreshold = img.max() * 0.15;
    lowThreshold = highThreshold * 0.05;

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(75)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res


def hysteresis(img):
    M, N = img.shape
    weak = 75
    strong = 255

    # Create a 3x3 neighborhood array
    neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)

    # Dilate the image with the neighborhood to find strong neighbors
    dilated = convolve((img >= strong).astype(int), neighborhood, mode='constant')

    # Find indices of weak pixels
    weak_indices = np.where(img == weak)

    # Check if any of the neighboring pixels are strong
    has_strong_neighbor = np.any(dilated[weak_indices[0], weak_indices[1]] > 0, axis=0)

    # Update weak pixels based on neighboring strong pixels
    img[weak_indices[0][~has_strong_neighbor], weak_indices[1][~has_strong_neighbor]] = 0

    return img


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

        left_frame = tk.CTkFrame(self, width=250, height=600)
        left_frame.pack(side="left", fill="both")
        left_frame.pack_propagate(False)
        right_frame = tk.CTkFrame(self)
        right_frame.pack(expand="True", padx="25px", pady="25px")

        self.img_label = tk.CTkLabel(master=right_frame, text="", bg_color="transparent", fg_color=None)
        self.img_label.pack(anchor="center", expand="True")

        slider_frame = tk.CTkFrame(master=left_frame)
        slider_frame.pack(side="bottom", pady="20px")

        self.slider = tk.CTkSlider(master=slider_frame, from_=0, to=1, command=self.slider_event, number_of_steps=10)
        self.slider.set(0)
        self.slider.pack(anchor="center", side="bottom")

        self.slider_value_label = tk.CTkLabel(master=slider_frame, text=round(float(self.slider.get()), 1))
        self.slider_value_label.pack(anchor="center", side="top")

        upload_button = tk.CTkButton(master=left_frame, command=lambda: self.upload_button_clicked(),
                                     text="Upload Image")
        upload_button.grid(row=0, column=0, columnspan=2, padx="50px", pady=(40, 0))

        blur_button = tk.CTkButton(master=left_frame,
                                   command=lambda: self.blurr_button_clicked(round(float(self.slider.get()), 1)),
                                   text="Blur")
        blur_button.grid(row=1, column=0, padx="10px", pady=(40, 0))

        sketch_button = tk.CTkButton(master=left_frame, command=lambda: self.sketch_button_clicked(), text="Sketch")
        sketch_button.grid(row=1, column=1, padx="10px", pady=(40, 0))

        edge_detection_button = tk.CTkButton(master=left_frame,
                                             command=lambda: self.edge_detection_button_clicked(),
                                             text="Edge Detection")
        edge_detection_button.grid(row=2, column=0, padx="10px", pady=(15, 0))

        photocopy_button = tk.CTkButton(master=left_frame,
                                        command=lambda: self.photocopy_button_clicked(
                                            round(float(self.slider.get()), 1)),
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

        negative_button = tk.CTkButton(master=left_frame, command=lambda: self.negative_button_clicked(),
                                       text="Negative")
        negative_button.grid(row=5, column=0, padx="10px", pady=(15, 0))

        vignetting_button = tk.CTkButton(master=left_frame, command=lambda: self.vignetting_button_clicked(
            round(float(self.slider.get()), 1)),
                                         text='Vignetting')
        vignetting_button.grid(row=5, column=1, padx='10px', pady=(15, 0))

        sepia_button = tk.CTkButton(master=left_frame, command=lambda: self.sepia_button_clicked(),
                                    text="Sepia")
        sepia_button.grid(row=6, column=0, padx="10px", pady=(15, 0))

        night_vision_button = tk.CTkButton(master=left_frame, command=lambda: self.night_vision_button_clicked(),
                                           text='Night Vision')
        night_vision_button.grid(row=6, column=1, padx='10px', pady=(15, 0))

        hist_equal_button = tk.CTkButton(master=left_frame, command=lambda: self.hist_equal_button_clicked(),
                                           text='Histogram Equalization')
        hist_equal_button.grid(row=7, column=0, padx='10px', pady=(15, 0))

        posterize_button = tk.CTkButton(master=left_frame, command=lambda: self.posterize_button_clicked(round(float(self.slider.get()), 1)),
                                        text="Posterize")
        posterize_button.grid(row=7, column=1, padx="10px", pady=(15, 0))

        contrast_equal_button = tk.CTkButton(master=left_frame, command=lambda: self.contrast_button_clicked(),
                                           text='Contrast Highlighting')
        contrast_equal_button.grid(row=8, column=0, padx='10px', pady=(15, 0))

    def hist_equal_button_clicked(self):
        if self.current_image is None:
            return

        if self.PrevOperation == "Hist Equalize":
            return


        self.grayscale_button_clicked()

        img_data = np.array(self.current_image)

        print(img_data)

        hist, bins = np.histogram(img_data.flatten(), bins=256, range=[0, 256])

        cdf = hist.cumsum()

        cdf_normalized = cdf * 255 / cdf[-1]

        img_final = np.interp(img_data.flatten(), bins[:-1], cdf_normalized).reshape(img_data.shape)

        img = Image.fromarray(img_final)
        self.current_image = img
        img.thumbnail((700, 550))
        show_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=show_img)
        self.img_label.image = show_img
        self.PrevOperation = "Hist Equalize"
        self.IPrevImage = self.current_image

    def edge_detection_button_clicked(self):
        if self.current_image is None:
            return

        if self.PrevOperation == "Canny":
            return

        self.grayscale_button_clicked()
        img_data = np.array(self.current_image)
        img_smoothed = convolve(img_data, gaussian_kernel(5, 1))
        gradient_mat, theta_mat = sobel_filters(img_smoothed)
        non_max_img = non_max_suppression(gradient_mat, theta_mat)
        threshold_img = threshold_func(non_max_img)
        img_final = hysteresis(threshold_img)
        img = Image.fromarray(img_final)
        self.current_image = img
        img.thumbnail((700, 550))
        show_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=show_img)
        self.img_label.image = show_img
        self.PrevOperation = "Canny"
        self.IPrevImage = self.current_image

    def blurr_button_clicked(self, scale):
        if self.current_image is None:
            return
        n = int(scale * 10)
        if self.PrevOperation == "Blur":
            if n == 0:
                blur_i = np.array(self.IPrevImage)
                img = Image.fromarray(blur_i)
                self.current_image = img
                img.thumbnail((700, 550))
                show_img = ImageTk.PhotoImage(img)
                self.img_label.configure(image=show_img)
                self.img_label.image = show_img
                return
            img_data = np.array(self.IPrevImage)
        else:
            img_data = np.array(self.current_image)

        if n != 0:
            blur_i = cv2.blur(img_data, (n, n),
                              dst=None,
                              anchor=None,
                              borderType=None)
            img = Image.fromarray(blur_i)
            self.current_image = img
            img.thumbnail((700, 550))
            show_img = ImageTk.PhotoImage(img)
            self.img_label.configure(image=show_img)
            self.img_label.image = show_img
        self.PrevOperation = "Blur"
        self.IPrevImage = img_data

    def sepia_button_clicked(self):
        if self.current_image is None:
            return
        if self.PrevOperation == "Sepia":
            return
        else:
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

        self.PrevOperation = "Sepia"

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

    def vignetting_button_clicked(self, scale):
        if self.current_image is None:
            return
        if self.PrevOperation == "Vignetting":
            img_data = np.array(self.IPrevImage)
        else:
            img_data = np.array(self.current_image)

        n = len(img_data)  # no of row
        m = len(img_data[0])  # no of column
        an = math.floor(n / 2)
        am = math.floor(m / 2)
        d = math.sqrt(an ** 2 + am ** 2)
        dist_m = np.fromfunction(lambda i, j: scale * np.sqrt((abs(j - am) ** 2) + (abs(i - an) ** 2)) / d,
                                 img_data.shape[:2])

        final_d = 1 - dist_m

        if img_data.ndim == 3:
            final_d = np.dstack([final_d] * 3)

        final_d = np.abs(final_d)
        np.clip(final_d, 0, 255, out=final_d)
        img = img_data * final_d

        img = Image.fromarray(img.astype(np.uint8))
        self.current_image = img
        img.thumbnail((700, 550))
        show_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=show_img)
        self.img_label.image = show_img

        self.IPrevImage = img_data
        self.PrevOperation = 'Vignetting'

    def night_vision_button_clicked(self):
        if self.current_image is None:
            return
        if self.PrevOperation == "Night Vision":
            return
        else:
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
        self.PrevOperation = 'Night Vision'

    def mirror_button_clicked(self):
        if self.current_image is None:
            return
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
        if self.current_image is None:
            return
        self.IPrevImage = self.current_image
        self.PrevOperation = 'Negative'

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
        if self.current_image is None:
            return
        if self.PrevOperation == "Photocopy":
            if threshold == 0:
                grayscale_arr = np.array(self.IPrevImage)
                img = Image.fromarray(grayscale_arr)
                self.current_image = img
                img.thumbnail((700, 550))
                show_img = ImageTk.PhotoImage(img)
                self.img_label.configure(image=show_img)
                self.img_label.image = show_img
                return
            img_data = np.array(self.IPrevImage)
        else:
            img_data = np.array(self.current_image)

        # img_data = np.array(self.current_image)
        if threshold != 0:
            threshold = int(90 + threshold * 60)
            grayscale_arr = img2double(img_data)
            threshold /= 255.0
            grayscale_arr[grayscale_arr > threshold] = 1.0
            grayscale_arr[grayscale_arr <= threshold] = (grayscale_arr[grayscale_arr <= threshold] *
                                                         (threshold - grayscale_arr[grayscale_arr <= threshold])) / (
                                                                threshold * threshold)
            grayscale_arr = double2img(grayscale_arr)
            img = Image.fromarray(grayscale_arr.astype(np.uint8))
            self.current_image = img
            img.thumbnail((700, 550))
            show_img = ImageTk.PhotoImage(img)
            self.img_label.configure(image=show_img)
            self.img_label.image = show_img

        self.PrevOperation = "Photocopy"
        self.IPrevImage = img_data

    def grayscale_button_clicked(self):

        if self.current_image is None:
            return
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

    def posterize_button_clicked(self, scale):

        n = int(math.ceil(scale / 0.2 + 1))  # Number of levels of quantization (2, 3, 4, 5, 6)

        indices = np.arange(0, 256)  # List of all colors

        divider = np.linspace(0, 255, n + 1)[1]  # we get a divider

        quantiz = np.intp(np.linspace(0, 255, n))  # we get quantization colors

        color_levels = np.clip(np.intp(indices / divider), 0, n - 1)  # color levels 0,1,2..

        palette = quantiz[color_levels]  # Creating the palette

        if self.PrevOperation == "Posterize":
            img_data = np.array(self.IPrevImage)
            image = self.IPrevImage.copy()
            if scale == 0:
                img = np.array(self.IPrevImage)
                img = Image.fromarray(img)
                self.current_image = self.IPrevImage

                img.thumbnail((700, 550))
                show_img = ImageTk.PhotoImage(img)
                self.img_label.configure(image=show_img)
                self.img_label.image = show_img
                return
        else:
            img_data = np.array(self.current_image)
            image = self.current_image.copy()

        if scale != 0:
            img = palette[image]  # Applying palette on image

            img = cv2.convertScaleAbs(img)  # Converting image back to uint8

            img = Image.fromarray(img)
            self.current_image = img
            img.thumbnail((700, 550))
            show_img = ImageTk.PhotoImage(img)
            self.img_label.configure(image=show_img)
            self.img_label.image = show_img

        self.PrevOperation = "Posterize"
        self.IPrevImage = img_data

    def slider_event(self, _event=None):
        self.slider_value_label.configure(text=round(float(self.slider.get()), 1))

    def button_callback(self):
        print("Button Clicked")

    def upload_button_clicked(self):
        f_types = [('Jpg Files', '*.jpg')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        if filename == "":
            return
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