import tkinter as tk
from tkinter import filedialog, simpledialog, Scale
from PIL import Image, ImageDraw, ImageFont, ImageTk
import cv2
import numpy as np
import json
import os
import tkinter.ttk as ttk
import random
from rembg import remove

FONT_SIZE_RATIO = 1.25

class SyntheticIDGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Swiss ID Synthetic Data Generator")

        # Initialize variables
        self.image_path = None
        self.image = None
        self.original_image = None  # Initialize here
        self.tk_image = None
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.bboxes = []
        self.rotation_angle = 0

        # a frame for the main content and side panel
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Move existing widgets to the main frame
        self.load_btn = tk.Button(self.main_frame, text="Load Image", command=self.load_image)
        self.load_btn.pack()

        # Canvas to display the image
        self.canvas = tk.Canvas(self.main_frame, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Resize slider
        self.resize_scale = Scale(self.main_frame, from_=10, to=200, orient=tk.HORIZONTAL, label="Resize (%)", command=self.resize_image)
        self.resize_scale.set(100)
        self.resize_scale.pack()

        # rotation buttons
        self.rotate_frame = tk.Frame(self.main_frame)
        self.rotate_frame.pack()
        
        self.rotate_ccw_btn = tk.Button(self.rotate_frame, text="Rotate CCW", command=lambda: self.rotate_image(-90))
        self.rotate_ccw_btn.pack(side=tk.LEFT)
        
        self.rotate_cw_btn = tk.Button(self.rotate_frame, text="Rotate CW", command=lambda: self.rotate_image(90))
        self.rotate_cw_btn.pack(side=tk.LEFT)

        # Button to start generating synthetic data
        self.generate_btn = tk.Button(self.main_frame, text="Generate Synthetic Data", command=self.generate_synthetic_data)
        self.generate_btn.pack()

        self.side_panel = tk.Frame(root, width=300, bg='lightgray')
        self.side_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # treeview for modifications
        self.tree = ttk.Treeview(self.side_panel, columns=('Text', 'Color'), show='headings')
        self.tree.heading('Text', text='Text')
        self.tree.heading('Color', text='Color')
        self.tree.pack(fill=tk.BOTH, expand=True)

        # edit button
        self.edit_btn = tk.Button(self.side_panel, text="Edit Selected", command=self.edit_selected)
        self.edit_btn.pack()

        # Dictionary to store modifications
        self.modifications = {}

        # Bind mouse events for drawing rectangles
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
        self.last_file_path = "last_file.json"
        self.load_last_file_path()

        self.toolbar = tk.Frame(self.main_frame)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.current_tool = tk.StringVar(value="red")
        self.current_tool.trace('w', self.update_cursor)
        
        self.red_selector = tk.Radiobutton(self.toolbar, text="Red Selector", variable=self.current_tool, value="red")
        self.red_selector.pack(side=tk.LEFT)
        
        self.green_selector = tk.Radiobutton(self.toolbar, text="Green Selector", variable=self.current_tool, value="green")
        self.green_selector.pack(side=tk.LEFT)
        
        self.purple_selector = tk.Radiobutton(self.toolbar, text="Purple Selector", variable=self.current_tool, value="purple")
        self.purple_selector.pack(side=tk.LEFT)
        self.selection_folder = "alpha_bg_selection"
        
        self.black_threshold = tk.IntVar(value=100)
        self.black_threshold_scale = Scale(self.main_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                                           label="Black Threshold", variable=self.black_threshold)
        self.black_threshold_scale.pack()
        
        # blue selector to the toolbar
        self.blue_selector = tk.Radiobutton(self.toolbar, text="Blue Selector", variable=self.current_tool, value="blue")
        self.blue_selector.pack(side=tk.LEFT)
        
        # black selector to the toolbar
        self.black_selector = tk.Radiobutton(self.toolbar, text="Black Selector", variable=self.current_tool, value="black")
        self.black_selector.pack(side=tk.LEFT)
        
        # threshold for foreground/background separation
        self.fg_bg_threshold = tk.IntVar(value=128)
        self.fg_bg_threshold_scale = Scale(self.main_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                                           label="Foreground/Background Threshold", variable=self.fg_bg_threshold)
        self.fg_bg_threshold_scale.pack()

    def update_cursor(self, *args):
        if self.current_tool.get() == "green":
            self.canvas.config(cursor="crosshair")
        else:
            self.canvas.config(cursor="cross")

    def load_last_file_path(self):
            if os.path.exists(self.last_file_path):
                with open(self.last_file_path, 'r') as f:
                    data = json.load(f)
                    self.image_path = data.get('last_file', None)

    def save_last_file_path(self):
        with open(self.last_file_path, 'w') as f:
            json.dump({'last_file': self.image_path}, f)

    def load_image(self):
        initial_dir = os.path.dirname(self.image_path) if self.image_path else "/"
        self.image_path = filedialog.askopenfilename(initialdir=initial_dir)
        if self.image_path:
            self.original_image = Image.open(self.image_path)
            self.image = self.resize_to_fit(self.original_image, 1024)
            self.update_canvas()
            self.save_last_file_path()

    def resize_to_fit(self, img, max_size):
        ratio = max_size / max(img.width, img.height)
        if ratio < 1:  # Only resize if the image is larger than max_size
            new_size = (int(img.width * ratio), int(img.height * ratio))
            return img.resize(new_size, Image.LANCZOS)
        return img

    def update_canvas(self):
        if self.image:
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def resize_image(self, value):
        if self.original_image:
            scale = int(value) / 100
            new_size = (int(self.original_image.width * scale), int(self.original_image.height * scale))
            self.image = self.original_image.copy()
            self.image = self.image.resize(new_size, Image.LANCZOS)
            self.rotate_image(self.rotation_angle)  # Apply rotation after resize

    def rotate_image(self, angle):
        if self.image:
            self.rotation_angle = (self.rotation_angle + angle) % 360
            rotated_image = self.image.rotate(self.rotation_angle, expand=True)
            self.image = rotated_image
            self.update_canvas()

    def get_selector_color(self):
        if self.current_tool.get() == "green":
            return 'green'
        elif self.current_tool.get() == "purple":
            return 'purple'
        elif self.current_tool.get() == "blue":
            return 'blue'
        else:
            return 'red'
    
    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        color = self.get_selector_color()
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline=color)

    def on_mouse_drag(self, event):
        curX, curY = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)
        color = self.get_selector_color()
        self.canvas.itemconfig(self.rect, outline=color)

    def on_button_release(self, event):
        end_x, end_y = (event.x, event.y)
        bbox = f"{self.start_x} {self.start_y} {end_x} {end_y}"
        
        if self.current_tool.get() == "red":
            new_data = simpledialog.askstring("Input", "Enter the new data for this field:")
            if new_data:
                avg_color = self.modify_image(self.start_x, self.start_y, end_x, end_y, new_data)
                self.modifications[bbox] = {'text': new_data, 'color': avg_color}
                self.update_tree()
        elif self.current_tool.get() == "green":
            image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")])
            if image_path:
                self.apply_image(self.start_x, self.start_y, end_x, end_y, image_path)
        
        elif self.current_tool.get() == "purple":
            self.apply_random_image(self.start_x, self.start_y, end_x, end_y)
        
        elif self.current_tool.get() == "blue":
            self.replace_black_pixels(self.start_x, self.start_y, end_x, end_y)
        
        elif self.current_tool.get() == "black":
            self.split_and_fill_foreground(self.start_x, self.start_y, end_x, end_y)
        
        self.rect = None

    def replace_black_pixels(self, start_x, start_y, end_x, end_y):
        # Convert PIL Image to numpy array
        img_array = np.array(self.image)
        
        # Define the region of interest
        roi = img_array[start_y:end_y, start_x:end_x]
        
        print(f"DEBUG: ROI shape = {roi.shape}")
        
        # Create a mask for black or near-black pixels
        black_threshold = self.black_threshold.get()
        is_black = np.all(roi < black_threshold, axis=2)
        
        print(f"DEBUG: is_black shape = {is_black.shape}")
        print(f"DEBUG: Number of black pixels = {np.sum(is_black)}")
        
        # Find non-black pixels
        non_black = np.logical_not(is_black)
        
        print(f"DEBUG: non_black shape = {non_black.shape}")
        print(f"DEBUG: Number of non-black pixels = {np.sum(non_black)}")
        
        if np.sum(non_black) == 0:
            print("DEBUG: No non-black pixels found in the selection")
        
        # Find coordinates of black pixels
        y_black, x_black = np.where(is_black)
        
        # For each black pixel, find the nearest non-black pixel
        for y, x in zip(y_black, x_black):
            # Define a search area around the pixel
            R = 10
            y_start, y_end = max(0, y-R), min(roi.shape[0], y+R)
            x_start, x_end = max(0, x-R), min(roi.shape[1], x+R)
            
            for i in range(y_end - 1, y_start - 1, -1):
                for j in range(x_end - 1, x_start - 1, -1):
                    t = 150
                    if roi[i, j][0] > t and roi[i, j][1] > t and roi[i, j][2] > t:
                        roi[y, x] = roi[i, j]
                        break
                
        img_array[start_y:end_y, start_x:end_x] = roi
        self.image = Image.fromarray(img_array)
        self.update_canvas()
        
        print("DEBUG: Pixel replacement completed")
        
    def apply_random_image(self, start_x, start_y, end_x, end_y):
        if os.path.exists(self.selection_folder) and os.path.isdir(self.selection_folder):
            image_files = [f for f in os.listdir(self.selection_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            if image_files:
                random_image = random.choice(image_files)
                image_path = os.path.join(self.selection_folder, random_image)
                self.apply_image(start_x, start_y, end_x, end_y, image_path)
            else:
                tk.messagebox.showwarning("No Images", "No image files found in the 'selection' folder.")
        else:
            tk.messagebox.showwarning("Folder Not Found", "The 'selection' folder does not exist.")

    # make this work properly for transparent images : right now it puts a black background on them, but there should be no back
    def apply_image(self, start_x, start_y, end_x, end_y, image_path):
        applied_image = Image.open(image_path)
        box_width = end_x - start_x
        box_height = end_y - start_y
        applied_image = applied_image.resize((box_width, box_height), Image.LANCZOS)
        
        self.image.paste(applied_image, (start_x, start_y))
        self.update_canvas()

    def modify_image(self, start_x, start_y, end_x, end_y, new_data):
        # First, replace black pixels in the selected area
        self.replace_black_pixels(start_x, start_y, end_x, end_y)

        # Calculate the font size
        box_height = end_y - start_y
        font_size = min(box_height, (end_x - start_x) // (len(new_data) + 1))
        font_size*=FONT_SIZE_RATIO
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)

        # Create a temporary transparent image for the text
        text_image = Image.new("RGBA", (end_x - start_x, end_y - start_y), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_image)

        # Get the bounding box of the text
        left, top, right, bottom = draw.textbbox((0, 0), new_data, font=font)
        text_width = right - left
        text_height = bottom - top

        # Center the text both horizontally and vertically
        text_x = (text_image.width - text_width) // 2
        text_y = (text_image.height - text_height) // 2

        # Draw the text with black color
        draw.text((text_x, text_y), new_data, fill="black", font=font)

        # Paste the text image onto the original image, using the text as a mask
        self.image.paste(text_image, (start_x, start_y), text_image)

        # Update the canvas with the new image
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Calculate average color for the modifications dictionary
        box = (start_x, start_y, end_x, end_y)
        cropped_image = self.image.crop(box)
        avg_color = np.array(cropped_image).mean(axis=(0, 1)).astype(int)
        avg_color = tuple(avg_color[:3])  # Convert to a tuple of RGB values

        # Return the average color
        return '#{:02x}{:02x}{:02x}'.format(*avg_color)

    def update_tree(self):
        self.tree.delete(*self.tree.get_children())
        for bbox, data in self.modifications.items():
            self.tree.insert('', 'end', values=(data['text'], data['color']), tags=(bbox,))

    def edit_selected(self):
        selected_item = self.tree.selection()
        if selected_item:
            item = self.tree.item(selected_item[0])
            bbox = self.tree.item(selected_item[0])['tags'][0]
            print("DEBUG: bbox =", bbox)  # Keep this line for debugging
            old_text = item['values'][0]
            new_text = simpledialog.askstring("Edit Text", "Enter new text:", initialvalue=old_text)
            if new_text:
                # Convert bbox to a tuple of integers
                bbox_values = tuple(map(int, bbox.split()))
                start_x, start_y, end_x, end_y = bbox_values
                
                self.modify_image(start_x, start_y, end_x, end_y, new_text)
                self.modifications[bbox]['text'] = new_text
                self.update_tree()

    def generate_synthetic_data(self):
        if self.image:
            synthetic_image_path = filedialog.asksaveasfilename(defaultextension=".png")
            if synthetic_image_path:
                self.image.save(synthetic_image_path)

    def split_and_fill_foreground(self, start_x, start_y, end_x, end_y):
        
        
        # Convert PIL Image to numpy array
        img_array = np.array(self.image)
        
        # Define the region of interest
        roi = img_array[start_y:end_y, start_x:end_x]
        
        # Remove the background
        output_image = remove(roi)
    
        # Save the result
        # Convert output_image to PIL Image
        output_pil = Image.fromarray(output_image)
        # Convert output_image to numpy array
        output_array = np.array(output_pil)
        
        # Create a mask for non-fully transparent pixels
        alpha_mask = output_array[:,:,3] > 0
        
        # Create a black image of the same size
        black_image = np.zeros_like(output_array)
        
        # Set non-fully transparent pixels to black
        black_image[alpha_mask] = [0, 0, 0, 255]
        
        # Convert back to PIL Image
        output_pil = Image.fromarray(black_image.astype(np.uint8))
        
        # Initialize counters
        alpha_counter = 0
        non_alpha_counter = 0

        # Get the dimensions of the image
        width, height = output_pil.size

        pixels_to_fill = []
        avg = [0, 0, 0]
        # Loop over all pixels
        for x in range(width):
            for y in range(height):
                pixel = output_pil.getpixel((x, y))
                
                # Check if the pixel is pure alpha (fully transparent)
                if pixel[3] == 0:
                    p = img_array[y,x]
                    
                    avg[0] += p[0]
                    avg[1] += p[1]
                    avg[2] += p[2]
                    non_alpha_counter += 1
                    
                else:
                    pixels_to_fill.append((start_y+y, start_x+x))
                    
                    print(f"Non-pure alpha pixel at ({x}, {y}): {pixel}")

        avg[0] /= (non_alpha_counter)
        avg[1] /= (non_alpha_counter)
        avg[2] /= (non_alpha_counter)
        for pixel in pixels_to_fill:
            img_array[pixel[0], pixel[1]][0] = avg[0]
            img_array[pixel[0], pixel[1]][1] = avg[1]
            img_array[pixel[0], pixel[1]][2] = avg[2]
                        
        # Update the image and canvas
        self.image = Image.fromarray(img_array)
        self.update_canvas()

if __name__ == "__main__":
    root = tk.Tk()
    app = SyntheticIDGenerator(root)
    root.mainloop()