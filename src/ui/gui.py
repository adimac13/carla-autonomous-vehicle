import customtkinter as ctk

class CarlaControlPanel(ctk.CTk):
    def __init__(self):
        super().__init__()
        font_name = "TimesNewRoman"

        #Window config
        self.geometry("600x500")
        self.title("Carla control panel")
        self.resizable(False, False)


        self.label_title = ctk.CTkLabel(self, text="Route Configuration", font = (font_name, 20, "bold"))
        self.label_title.pack(pady=20)

        self.frame_controls = ctk.CTkFrame(self)
        self.frame_controls.pack(pady=40, padx=20)

        self.button_start = ctk.CTkButton(self.frame_controls, text="START", fg_color="green")
        self.button_start.pack(pady=10)

        self.button_stop = ctk.CTkButton(self.frame_controls, text="STOP",  fg_color="red")
        self.button_stop.pack(pady=10)

if __name__ == "__main__":
    app = CarlaControlPanel()
    app.mainloop()


