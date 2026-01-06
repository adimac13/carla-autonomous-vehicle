import customtkinter as ctk

TOTAL_SPAWN_POINTS = 80

class CarlaControlPanel(ctk.CTk):
    def __init__(self):
        super().__init__()
        font_name = "TimesNewRoman"

        #Window config
        self.geometry("600x500")
        self.title("Carla control panel")
        self.resizable(False, False)

        #Title
        self.label_title = ctk.CTkLabel(self, text="Route Configuration", font = (font_name, 20, "bold"))
        self.label_title.pack(pady=20)

        #Config for slider
        self.label_slider = ctk.CTkLabel(self, text = "Spawn point ID: 0 ", font = (font_name, 15, "bold"))
        self.label_slider.pack(pady = 10)

        #Spawn slider for spawn points
        self.spawn_slider = ctk.CTkSlider(master = self, from_ = 0, to = TOTAL_SPAWN_POINTS, number_of_steps=TOTAL_SPAWN_POINTS, command = self.spawn_slider_update_value, width = 500)
        self.spawn_slider.set(0)
        self.spawn_slider.pack(pady = 10)

        self.spawn_slider_control = ctk.CTkFrame(self)
        self.spawn_slider_control.pack(pady=20, padx=20)
        self.button_start = ctk.CTkButton(self.spawn_slider_control, text="SPAWN A CAR", fg_color="PURPLE")
        self.button_start.pack(pady=0)


        #Config for buttons
        self.frame_controls = ctk.CTkFrame(self)
        self.frame_controls.pack(pady=40, padx=20)

        #Spawn start button
        self.button_start = ctk.CTkButton(self.frame_controls, text="START", fg_color="green")
        self.button_start.pack(pady=10)

        #Spawn stop button
        self.button_stop = ctk.CTkButton(self.frame_controls, text="STOP",  fg_color="red")
        self.button_stop.pack(pady=10)

    def spawn_slider_update_value(self, value):
        self.label_slider.configure(text = f"Spawn point ID: {value}")

if __name__ == "__main__":
    app = CarlaControlPanel()
    app.mainloop()


