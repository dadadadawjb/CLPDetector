import tkinter as tk

from GUI import CLPDetectorApp

if __name__ == '__main__':
    root = tk.Tk()
    root.title("CLPDetector")

    my_app = CLPDetectorApp(root)

    root.mainloop()
