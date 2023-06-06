import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import segmentation_mask

class MaskingTool:
    def __init__(self, master, init_mask, ori_img):
        self.master = master
        self.canvas = tk.Canvas(self.master, width=572, height=768)
        self.canvas.pack()

        self.image = ori_img.astype(np.uint8)
        self.init_mask = init_mask.astype(np.uint8)
        self.masked_image = None
        self.masking = False
        self.mask_points = []
        self.previous_points = []

        self.open_button = tk.Button(self.master, text="開始遮罩", command=self.start_masking)
        self.open_button.pack()

        self.save_button = tk.Button(self.master, text="儲存遮罩圖片", command=self.save_masked_image)
        self.save_button.pack()

        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<Button-3>", self.on_right_click)

        # Load result image
        self.load_result_image(self.init_mask, self.image)

        # Show image on canvas
        self.show_image()

    def load_result_image(self, init_mask, image):
        image[:,:,0] *= init_mask
        image[:,:,1] *= init_mask
        image[:,:,2] *= init_mask
        result = np.squeeze(image).astype(np.uint8) # turn into uint8
        self.image = Image.fromarray(result.astype(np.uint8).copy()).convert("RGB")
        self.masked_image = self.image.copy()

    def show_image(self):
        self.photo = ImageTk.PhotoImage(self.masked_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def start_masking(self):
        self.masking = True

    def on_mouse_click(self, event):
        if self.masking:
            self.mask_points.append((event.x, event.y))
            self.update_masked_image()

    def on_mouse_drag(self, event):
        if self.masking:
            self.mask_points.append((event.x, event.y))
            self.update_masked_image()

    def on_right_click(self, event):
        if self.masking:
            if self.mask_points:
                self.previous_points.append(self.mask_points.pop())
                self.update_masked_image()

    def update_masked_image(self):
        self.masked_image = self.image.copy()
        draw = ImageDraw.Draw(self.masked_image)
        for i in range(1, len(self.mask_points)):
            draw.line((self.mask_points[i-1], self.mask_points[i]), fill=(0, 0, 0), width=5)
        self.show_image()

    def save_masked_image(self):
        if not self.mask_points:
            return

        # Update the result variable in segmentation_mask.py
        self.result = np.array(self.masked_image)

    def get_result(self):
        return self.result

# Create the main window
window = tk.Tk()
window.title("遮罩工具")

# Run segmentation_mask.py and get the result
segmentationMask = segmentation_mask.SegmentationMask(image_name="drawing_data/ghost.jpg", isShowResult=False)
para = {"D1_kernel": 11, "D1_iter": 2, "D2_kernel": 7, "D2_iter": 1, "blockSize": 49, "tolerance": 2}
result = segmentationMask.get_segmentation_mask(**para)
ori_img = segmentationMask.img_ori

# Create the masking_tool instance
masking_tool = MaskingTool(window, result, ori_img)

# Start the GUI event loop
window.mainloop()
