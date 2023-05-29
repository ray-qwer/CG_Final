import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

class LabelingGUI:
    def __init__(self, master, image_path):
        self.master = master
        self.canvas = tk.Canvas(self.master)
        self.canvas.pack()
        self.points = []
        self.point_entries = []

        image = Image.open(image_path)
        self.image_width, self.image_height = image.size
        self.image_tk = ImageTk.PhotoImage(image)

        self.canvas.config(width=self.image_width, height=self.image_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Button-3>", self.remove_point)

        self.save_button = tk.Button(self.master, text="Save Points", command=self.save_points)
        self.save_button.pack()

    def add_point(self, event):
        self.points.append((event.x, event.y))
        entry = tk.Entry(self.master)
        entry.pack()
        self.point_entries.append(entry)
        self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill='red')

    def remove_point(self, event):
        for i, point in enumerate(self.points):
            if abs(point[0] - event.x) < 5 and abs(point[1] - event.y) < 5:
                self.points.pop(i)
                self.point_entries[i].destroy()
                self.point_entries.pop(i)
                self.canvas.delete('all')
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
                for p in self.points:
                    self.canvas.create_oval(p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3, fill='red')
                break

    def save_points(self):
        if len(self.points) == 15:
            labeled_points = []
            for i, point in enumerate(self.points):
                try:
                    label_index = int(self.point_entries[i].get())
                    labeled_points.append((label_index, point[0], point[1]))
                except ValueError:
                    print(f"Invalid label index for point {i+1}. Please enter an integer.")
                    return

            labeled_points_array = np.array(labeled_points)
            np.savetxt("label_points.csv", labeled_points_array[:, 1:], delimiter=",")
            print("Label points saved successfully!")
        else:
            print("Please label exactly 15 points before saving.")

if __name__ == '__main__':
    image_path = "drawing_data/bear.jpg"  # Replace with the actual path to your image file
    root = tk.Tk()
    app = LabelingGUI(root, image_path)
    root.mainloop()