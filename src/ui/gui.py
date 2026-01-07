import customtkinter as ctk
import carla
import threading
import random
import queue

TOTAL_SPAWN_POINTS = 80
SLIDER_WIDTH = 700

class CarlaControlPanel(ctk.CTk):
    def __init__(self):
        super().__init__()
        font_name = "TimesNewRoman"

        self.carla_running = False
        self.spawn_point = True
        self.spawn_point_change = True
        self.dest_point = False

        #Window config
        self.geometry("800x700")
        self.title("Carla control panel")
        self.resizable(False, False)

        #Title
        self.label_title = ctk.CTkLabel(self, text="Route Configuration in CARLA", font = (font_name, 20, "bold"))
        self.label_title.pack(pady=20)

        #Config for slider
        self.label_slider_spawn = ctk.CTkLabel(self, text = "Spawn point ID: 0 ", font = (font_name, 15, "bold"))
        self.label_slider_spawn.pack(pady = 10)

        #Spawn slider for spawn points with button
        self.spawn_slider= ctk.CTkSlider(master = self, from_ = 0, to = TOTAL_SPAWN_POINTS, number_of_steps=TOTAL_SPAWN_POINTS, command = self.spawn_slider_update_value, width = SLIDER_WIDTH)
        self.spawn_slider.set(0)
        self.spawn_slider.pack(pady = 10)
        self.spawn_slider_control = ctk.CTkFrame(self)
        self.spawn_slider_control.pack(pady=20)
        self.button_spawn = ctk.CTkButton(self.spawn_slider_control, text="SET SPAWN POINT", fg_color="PURPLE", command = self.spawn_car)
        self.button_spawn.pack(pady=0)
        self.spawn_point_id = 0


        #Config for slider
        self.label_slider_dest = ctk.CTkLabel(self, text = "Destination point ID: 0 ", font = (font_name, 15, "bold"))
        self.label_slider_dest.pack(pady = 10)

        #Spawn slider for destination points with button
        self.dest_slider = ctk.CTkSlider(master = self, from_ = 0, to = TOTAL_SPAWN_POINTS, number_of_steps=TOTAL_SPAWN_POINTS, command = self.dest_slider_update_value, width = SLIDER_WIDTH)
        self.dest_slider.set(0)
        self.dest_slider.pack(pady = 10)
        self.dest_slider_control = ctk.CTkFrame(self)
        self.dest_slider_control.pack(pady=20)
        self.button_set_dest = ctk.CTkButton(self.dest_slider_control, text="SET DESTINATION", fg_color="PURPLE")
        self.button_set_dest.pack(pady=0)


        #Config for buttons
        self.frame_controls = ctk.CTkFrame(self)
        self.frame_controls.pack(pady=40, padx=20)

        #Start button
        self.button_start = ctk.CTkButton(self.frame_controls, text="START A CAR", fg_color="green")
        self.button_start.pack(pady=10)

        #Stop button
        self.button_stop = ctk.CTkButton(self.frame_controls, text="STOP A CAR",  fg_color="red")
        self.button_stop.pack(pady=10)

        #Turn on sim button
        self.button_sim = ctk.CTkButton(self.frame_controls, text="START CARLA",  fg_color="blue", command = self.start_simulation)
        self.button_sim.pack(pady=10)

    def spawn_slider_update_value(self, value):
        self.label_slider_spawn.configure(text = f"Spawn point ID: {int(value)}")
        self.spawn_point_id = int(value)
        self.spawn_point_change = True

    def dest_slider_update_value(self, value):
        self.label_slider_dest.configure(text = f"Destination point ID: {int(value)}")

    def start_simulation(self):
        if not self.carla_running:
            self.carla_running = True
            t = threading.Thread(target = self.carla_thread, daemon = True)
            t.start()

    def spawn_car(self):
        self.spawn_point = True

    def carla_thread(self, model_nn = "dave2_const_v_b"):
        actor_list = []
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(10.0)

            #CONNECTING TO WORLD
            self.world = client.load_world('Town02')
            settings = self.world.get_settings()
            spectator = self.world.get_spectator()
            spectator.set_transform(carla.Transform(carla.Location(x=100, y=204, z=203.0),
                                                    carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)))

            #APPLYING SYNCHRONOUS MODE
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)

            blueprint_library =  self.world.get_blueprint_library()
            model_3 = blueprint_library.filter("model3")[0]

            #CONFIGURING SPAWN POINTS
            self.all_spawn_points =  self.world.get_map().get_spawn_points()

            while self.carla_running:
                self.world.tick()
               #Drawing spawn point
                if self.spawn_point:
                    self.draw_spawn_point()



        finally:
            settings =  self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            for actor in actor_list:
                actor.destroy()

    def draw_spawn_point(self):
        loc = self.all_spawn_points[int(self.spawn_point_id)].location
        if self.spawn_point_change:
            loc.z += 1
            self.spawn_point_change = False
        self.world.debug.draw_string(
            loc,
            'Spawn point',
            color=carla.Color(r=255, g=0, b=0),
            life_time=0.1,
        )

if __name__ == "__main__":
    app = CarlaControlPanel()
    app.mainloop()


