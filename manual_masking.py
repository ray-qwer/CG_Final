import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import segmentation_mask
import matplotlib.pyplot as plt
import copy

class MaskingTool:
    def __init__(self, master, init_mask, ori_img):
        self.master = master
        
        # image and mask have been resized before
        self.image = ori_img.astype(np.uint8)
        self.init_mask = init_mask.astype(np.uint8)
        
        # the width and height of canvas are controlled by image
        self.canvas = tk.Canvas(self.master, width=self.image.shape[1], height=self.image.shape[0])
        self.canvas.pack()

        self.masked_image = None
        self.masking = True

        self.mask_points = copy.deepcopy(self.init_mask)

        self.open_button = tk.Button(self.master, text="開始遮罩", command=self.start_masking)
        self.open_button.pack()

        self.close_button = tk.Button(self.master, text="開始還原", command=self.stop_masking)
        self.close_button.pack()

        self.complete_button = tk.Button(self.master, text="完成", command=self.complete)
        self.complete_button.pack()

        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)

        # Load result image
        self.load_result_image(self.init_mask, self.image)

        # Show image on canvas
        self.show_image()

    def load_result_image(self, mask, image):
        image1 = np.copy(image)
        image1[:,:,0] *= mask
        image1[:,:,1] *= mask
        image1[:,:,2] *= mask
        result = image1.astype(np.uint8) # turn into uint8
        _ = Image.fromarray(result).convert("RGB")
        self.masked_image = _.copy()
        self.show_image()

    def show_image(self):
        self.photo = ImageTk.PhotoImage(self.masked_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def start_masking(self):
        self.masking = True

    def stop_masking(self):
        self.masking = False

    def on_mouse_drag(self, event):
        x = int(event.x)
        y = int(event.y)
        if self.masking:
            self.mask_points[y-3:y+4, x-3:x+4] = 0
        else:
            self.mask_points[y-3:y+4, x-3:x+4] = 1
        
        self.load_result_image(self.mask_points, np.asarray(self.image))

    def get_result(self):
        return self.mask_points
    
    def complete(self):
        print("已儲存")
        self.master.destroy()
        
if __name__ == "__main__":
    # Create the main window
    window = tk.Tk()
    window.title("遮罩工具")
    print("a")

    # Run segmentation_mask.py and get the result
    segmentationMask = segmentation_mask.SegmentationMask(image_name="drawing_data/ghost.jpg", isShowResult=False)
    para = {"D1_kernel": 11, "D1_iter": 2, "D2_kernel": 7, "D2_iter": 1, "blockSize": 49, "tolerance": 2}
    result = segmentationMask.get_segmentation_mask(**para)
    ori_img = segmentationMask.img_ori

    # Create the masking_tool instance
    masking_tool = MaskingTool(window, result, ori_img)

    # Start the GUI event loop
    window.mainloop()
    final_mask = masking_tool.get_result()
    plt.subplot(1,2,1)
    plt.imshow(result, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(final_mask, cmap='gray')
    plt.show()