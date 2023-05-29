import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

class LabelingGUI:
    def __init__(self, master, image_path):
        self.image_name = image_path.replace(".jpg", "")
        image = Image.open(image_path)
        W,H = image.size
        scale = 512 / H
        canvas_width, canvas_height = int(W*scale), int(H*scale)
        image = image.resize((int(W*scale), int(H*scale)), Image.ANTIALIAS)
        self.image = image

        self.master = master
        self.canvas = tk.Canvas(self.master, width=canvas_width, height=canvas_height)
        self.canvas.pack()
        self.points = []
        self.point_entries = []
        self.point_labels = []
        self.label_counter = 0
        
        self.image_tk = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Button-3>", self.remove_point)

        self.save_button = tk.Button(self.master, text="Save Points", command=self.save_points)
        self.save_button.pack()

    def add_point(self, event):
        self.points.append((event.x, event.y))
        entry = tk.Entry(self.master)
        entry.insert(0, f"index-{self.label_counter},  x: {event.x},  y: {event.y}")
        entry.pack()
        self.point_entries.append(entry)
        label_text = str(self.label_counter)
        self.label_counter += 1
        point_label = self.canvas.create_text(event.x + 10, event.y - 10, text=label_text, fill='black')
        self.point_labels.append((point_label, label_text))
        self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill='red')

    def remove_point(self, event):
        for i, point in enumerate(self.points):
            if abs(point[0] - event.x) < 5 and abs(point[1] - event.y) < 5:
                self.label_counter -= 1
                self.points.pop(i)
                self.point_entries[i].destroy()
                self.point_entries.pop(i)
                self.canvas.delete(self.point_labels[i][0])
                self.point_labels.pop(i)
                self.canvas.delete('all')
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
                for p, label in zip(self.points, self.point_labels):
                    self.canvas.create_oval(p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3, fill='red')
                    label_text = label[1]
                    self.canvas.create_text(p[0] + 10, p[1] - 10, text=label_text, fill='black')
                break

    def save_points(self):
        if len(self.points) == 15:
            labeled_points = []
            for i, point in enumerate(self.points):
                try:
                    label_index = int(self.point_entries[i].get().split(",")[0].replace("index-",""))
                    labeled_points.append((label_index, point[0], point[1]))
                except ValueError:
                    print(f"Invalid label index for point {i+1}. Please enter an integer.")
                    return

            self.labeled_points_array = np.array(labeled_points)[:, 1:]
            np.save(f"{self.image_name}_skeleton", self.labeled_points_array)
            np.savetxt("label_points.csv", self.labeled_points_array, delimiter=",")
            print("Label points saved successfully!")
        else:
            print("Please label exactly 15 points before saving.")

    def get_labels(self):
        return self.labeled_points_array
    
    def check_skeletal(self, npy_path=None):
        if npy_path != None:
            pts = np.load(npy_path)
        else:
            pts = self.labeled_points_array
        # plot skeleton edges
        plt.plot([pts[0,0], pts[2,0]], [pts[0,1], pts[2,1]], color="red", linewidth=3)
        plt.plot([pts[1,0], pts[2,0]], [pts[1,1], pts[2,1]], color="red", linewidth=3)
        plt.plot([pts[2,0], pts[3,0]], [pts[2,1], pts[3,1]], color="red", linewidth=3)
        plt.plot([pts[1,0], pts[4,0]], [pts[1,1], pts[4,1]], color="red", linewidth=3)
        plt.plot([pts[4,0], pts[6,0]], [pts[4,1], pts[6,1]], color="red", linewidth=3)
        plt.plot([pts[3,0], pts[5,0]], [pts[3,1], pts[5,1]], color="red", linewidth=3)
        plt.plot([pts[5,0], pts[7,0]], [pts[5,1], pts[7,1]], color="red", linewidth=3)
        plt.plot([pts[2,0], pts[9,0]], [pts[2,1], pts[9,1]], color="red", linewidth=3)
        plt.plot([pts[8,0], pts[9,0]], [pts[8,1], pts[9,1]], color="red", linewidth=3)
        plt.plot([pts[9,0], pts[10,0]], [pts[9,1], pts[10,1]], color="red", linewidth=3)
        plt.plot([pts[8,0], pts[11,0]], [pts[8,1], pts[11,1]], color="red", linewidth=3)
        plt.plot([pts[11,0], pts[13,0]], [pts[11,1], pts[13,1]], color="red", linewidth=3)
        plt.plot([pts[10,0], pts[12,0]], [pts[10,1], pts[12,1]], color="red", linewidth=3)
        plt.plot([pts[12,0], pts[14,0]], [pts[12,1], pts[14,1]], color="red", linewidth=3)
        # plot skeleton points
        for i in range(len(pts)):
            plt.plot(pts[i,0], pts[i,1], marker="o", color="blue")
        plt.imshow(self.image)
        plt.show()


if __name__ == '__main__':
    image_path = "drawing_data/dragon_cat.jpg"  
    # image_path = "drawing_data/bear.jpg" 
    root = tk.Tk()
    gui = LabelingGUI(root, image_path)
    root.mainloop()

    '''
    # to obtain the [15, 2] size numpy array after manually label:
    
    pts = gui.get_labels()
    gui.check_skeletal()
    '''

    '''
    # to load the previously labeled [15, 2] size numpy array, without manually label again:
    '''
    gui.check_skeletal(npy_path="drawing_data/dragon_cat_skeleton.npy")
    # gui.check_skeletal(npy_path="drawing_data/bear_skeleton.npy")


    '''
    to load the skeleton.npy ONLY:
    '''
    skeleton_pts = np.load(npy_path="drawing_data/dragon_cat_skeleton.npy") # a [15, 2] size numpy array
   