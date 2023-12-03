import cv2
import numpy as np
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk

class ImageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Filter App")

        # Tạo các biến và các thành phần giao diện
        self.input_image_path = None
        self.output_image_path = "output_image.jpg"
        self.kernel_size = (5, 5)

        self.create_widgets()

    def create_widgets(self):
        # Tạo nút để chọn ảnh
        select_image_button = tk.Button(self.root, text="Select Image", command=self.select_image)
        select_image_button.pack(pady=10)

        # Tạo nút để áp dụng bộ lọc Gaussian
        apply_filter_button = tk.Button(self.root, text="Apply Gaussian Filter", command=self.apply_gaussian_filter)
        apply_filter_button.pack(pady=10)

    def select_image(self):
        # Cho phép người dùng chọn ảnh từ thư mục
        self.input_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])

        # Hiển thị ảnh đã chọn
        if self.input_image_path:
            self.display_selected_image()

    def display_selected_image(self):
        # Mở ảnh và hiển thị nó trên cửa sổ
        image = Image.open(self.input_image_path)
        image = ImageTk.PhotoImage(image)

        # Tạo một label để hiển thị ảnh
        image_label = tk.Label(self.root, image=image)
        image_label.image = image
        image_label.pack(pady=10)

    def apply_gaussian_filter(self):
        if self.input_image_path:
            # Đọc ảnh từ đường dẫn đã chọn
            image = cv2.imread(self.input_image_path)

            # Áp dụng bộ lọc Gaussian để làm mịn ảnh
            blurred_image = cv2.GaussianBlur(image, self.kernel_size, 0)

            # Hiển thị ảnh gốc và ảnh đã xử lý (cửa sổ hiển thị sẽ tự đóng sau khi nhấn phím bất kỳ)
            cv2.imshow('Blurred Image', blurred_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Vui lòng chọn ảnh trước khi áp dụng bộ lọc.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageFilterApp(root)
    root.mainloop()