import tkinter as tk
from tkinter import Label
from tkinter import filedialog
from PIL import Image, ImageTk

def ImageUploader():
    """
    Uploads an image to the image label
    """
    # filters file dialog to show only image files
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    # opens the file selection dialog and returns the selected path
    path = tk.filedialog.askopenfilename(filetypes=fileTypes)

    # if the path is not empty
    if path:
        # format the image to be used by tkinter
        img = Image.open(path)
        img = img.resize((200, 200))
        pic = ImageTk.PhotoImage(img)

        # display the image
        app.geometry("560x300")
        imageLabel.config(image=pic)
        imageLabel.image = pic
    else:
        print("No file is choosen!")

# check if the code is executed directly
if __name__ == "__main__":
    # defining thinker object
    app = tk.Tk()

    app.title("Is it Cat or Dog?")
    app.geometry("560x270")

    # add the image label
    imageLabel = tk.Label(app)
    imageLabel.pack(pady=10)

    # add the upload button
    uploadButton = tk.Button(app, text="Choose image", command=ImageUploader)
    uploadButton.pack(side=tk.BOTTOM, pady=20)

    app.mainloop()
