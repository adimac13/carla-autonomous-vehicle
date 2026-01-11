import customtkinter as ctk
import carla
import threading
import queue
from pathlib import Path
import torch
from h5py.h5ds import set_label
from torchvision import transforms
from src.models.model_dave2_const_v_b import DrivingModel as dave2_const_v_b_model
from src.models.model_dave2_const_v_b_CIL import DrivingModel as dave2_const_v_b_CIL_model
from src.data.collect_data_dave2_const_v_b import IMAGE_WIDTH, IMAGE_HEIGHT, FOV, raw_data_process
from src.driving.load_model_dave2_const_v_b_to_carla import process_frame
from agents.navigation.behavior_agent import BehaviorAgent
import numpy as np
import math

TOTAL_SPAWN_POINTS = 100
SLIDER_WIDTH = 700

class CarlaControlPanel(ctk.CTk):
    def __init__(self):
        super().__init__()
        font_name = "TimesNewRoman"
        #Flags
        self.carla_running = False      #Controls the main simulation loop execution
        self.spawn_point_draw = True    #Controls visualization of the spawn point
        self.dest_point_draw = True     #Controls visualization of the destination point
        self.spawn_car = False          #Controls spawning a car
        self.first_frame = True         #Flag for init in the first frame
        self.nn_car = False             #Activates neural network driving model
        self.car_go = False             #Controls vehicle movement
        self.reset_flag = False         #Indicates whether to reset

        self.geometry("800x700")
        self.title("Carla control panel")

        self.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)
        self.grid_columnconfigure((0, 1), weight=1)

        #Label for title
        self.label_title_frame = ctk.CTkFrame(self)
        self.label_title_frame.grid(row=0, column=0, columnspan=2, sticky="n")
        self.label_title = ctk.CTkLabel(self.label_title_frame, text="Route Configuration in CARLA",
                                        font=(font_name, 20, "bold"))
        self.label_title.pack(pady=0)

        #Label for slider
        self.label_slider_spawn = ctk.CTkLabel(self, text="Spawn point ID: 0 ", font=(font_name, 15, "bold"))
        self.label_slider_spawn.grid(row=1, column=0, columnspan=2, sticky="ns" )

        #Slider
        self.spawn_slider = ctk.CTkSlider(self, from_=0, to=TOTAL_SPAWN_POINTS,
                                          number_of_steps=TOTAL_SPAWN_POINTS, command=self.spawn_slider_update_value,
                                          width=SLIDER_WIDTH)
        self.spawn_slider.set(0)
        self.spawn_point_id = 0
        self.spawn_slider.grid(row=2, column=0, columnspan=2, sticky="n")

        #Label for slider
        self.label_slider_dest = ctk.CTkLabel(self, text="Destination point ID: 0 ", font=(font_name, 15, "bold"))
        self.label_slider_dest.grid(row=3, column=0, columnspan=2, sticky="ns")

        self.dest_slider = ctk.CTkSlider(self, from_=0, to=TOTAL_SPAWN_POINTS,
                                         number_of_steps=TOTAL_SPAWN_POINTS, command=self.dest_slider_update_value,
                                         width=SLIDER_WIDTH)
        self.dest_slider.set(0)
        self.dest_point_id = 0
        self.dest_slider.grid(row=4, column=0, columnspan=2, sticky="n")

        #Buttons split into two columns
        self.frame_left = ctk.CTkFrame(self)
        self.frame_left.grid(row=5, column=0, sticky="n")

        self.frame_right = ctk.CTkFrame(self)
        self.frame_right.grid(row=5, column=1, sticky="n")

        #Buttons on left for environment
        self.label_left_column = ctk.CTkLabel(self.frame_left, text="Environment", font=(font_name, 15, "bold"))
        self.label_left_column.pack(pady=10)

        self.button_sim_on = ctk.CTkButton(self.frame_left, text="Set up", fg_color="blue", command=self.start_simulation,
                                           font=(font_name, 15, "bold"))
        self.button_sim_on.pack(pady=10)

        self.button_sim_off = ctk.CTkButton(self.frame_left, text="Destroy", fg_color="blue", command=self.stop_simulation
                                            , font=(font_name, 15, "bold"))
        self.button_sim_off.pack(pady=10)

        self.label_left_status = ctk.CTkLabel(self.frame_left, text="", font=(font_name, 15, "bold"))
        self.label_left_status.pack(pady=10)


        #Buttons on right for car
        self.label_right_column = ctk.CTkLabel(self.frame_right, text="Car", font=(font_name, 15, "bold"))
        self.label_right_column.pack(pady=10)

        self.button_spawn = ctk.CTkButton(self.frame_right, text="Spawn a car", fg_color="darkgreen", command=self.spawn_car_on_road
                                          , font=(font_name, 15, "bold"))
        self.button_spawn.pack(pady=10)

        self.button_start = ctk.CTkButton(self.frame_right, text="Start a car", fg_color="green", command = self.start_car
                                         , font=(font_name, 15, "bold"))
        self.button_start.pack(pady=10)

        self.button_stop = ctk.CTkButton(self.frame_right, text="Stop a car", fg_color="red", command = self.stop_car
                                         , font=(font_name, 15, "bold"))
        self.button_stop.pack(pady=10)

        self.button_delete = ctk.CTkButton(self.frame_right, text="Reset", fg_color="darkred", command = self.reset
                                         , font=(font_name, 15, "bold"))
        self.button_delete.pack(pady=10)

        self.label_right_status = ctk.CTkLabel(self.frame_right, text="", font=(font_name, 15, "bold"))
        self.label_right_status.pack(pady=10)


    def set_label_left_status(self, status):
        self.label_left_status.configure(text=status)

    def set_label_right_status(self, status):
        self.label_right_status.configure(text=status)

    def spawn_slider_update_value(self, value):
        self.label_slider_spawn.configure(text=f"Spawn point ID: {int(value)}")
        self.spawn_point_id = int(value)
        self.spawn_point_draw = True

    def dest_slider_update_value(self, value):
        self.label_slider_dest.configure(text=f"Destination point ID: {int(value)}")
        self.dest_point_id = int(value)
        self.dest_point_draw = True

    def start_simulation(self):
        self.set_label_left_status("In progress...")

        if hasattr(self, 't') and self.t is not None and self.t.is_alive():
            self.set_label_left_status("CARLA is running")
            return

        if not self.carla_running:
            self.carla_running = True
            self.t = threading.Thread(target=self.carla_thread, daemon=True)
            self.t.start()

    def stop_simulation(self):
        self.carla_running = False
        self.set_label_left_status("In progress...")

    def spawn_car_on_road(self):
        self.spawn_point_draw = False
        self.dest_point_draw = False
        self.nn_car = True
        self.first_frame = True
        self.set_label_right_status("Spawning a car...")

    def start_car(self):
        self.car_go = True
        self.set_label_right_status("Driving")

    def stop_car(self):
        self.car_go = False
        self.set_label_right_status("Stopped")

    def reset(self):
        self.reset_flag = True
        self.set_label_right_status("Reset in progress...")

    def reset_handle(self):
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
        self.actor_list = []
        #Cleaning elements
        self.center_queue = None
        #Cleaning flags
        self.car_go = False
        self.nn_car = False
        self.first_frame = True
        self.spawn_point_draw = True
        self.dest_point_draw = True
        self.spawn_point_id = 0
        self.dest_point_id = 0
        self.spawn_slider.set(0)
        self.dest_slider.set(0)
        self.set_label_right_status("Reset done")

    def carla_thread(self, model_nn="dave2_const_v_b_CIL"):
        self.actor_list = []
        try:
            self.setup_carla_world()
            self.set_label_left_status("Connected")
            self.frame_number = 0
            while self.carla_running:
                self.world.tick()
                if not self.reset_flag:
                    if self.nn_car:
                        self.car_drive(model_nn)
                    else:
                        if self.spawn_point_draw:
                            self.draw_spawn_point()
                        if self.dest_point_draw:
                            self.draw_dest_point()
                else:
                    self.reset_handle()
                    self.reset_flag = False
        finally:
            self.cleanup_carla()
            self.set_label_left_status("Done")

    def setup_carla_world(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        self.world = client.load_world('Town02')
        settings = self.world.get_settings()
        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(carla.Location(x=100, y=204, z=203.0),carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)))
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.all_spawn_points = self.world.get_map().get_spawn_points()
        def key_to_sort(spawn_point):
            location = spawn_point.location
            return round(location.x) , round(location.y)
        self.all_spawn_points.sort(key = key_to_sort)

    def cleanup_carla(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        self.reset_handle()

    def draw_spawn_point(self):
        loc_raw = self.all_spawn_points[int(self.spawn_point_id)].location
        self.start_location = carla.Location(loc_raw)
        loc = carla.Location(loc_raw)
        loc.z += 1
        self.world.debug.draw_string(loc, 'Spawn point', color=carla.Color(r=255, g=0, b=0), life_time=0.1)

    def draw_dest_point(self):
        loc_raw = self.all_spawn_points[int(self.dest_point_id)].location
        self.destination = carla.Location(loc_raw)
        loc = carla.Location(loc_raw)
        loc.z += 1
        self.world.debug.draw_string(loc, 'Destination', color=carla.Color(r=255, g=0, b=0), life_time=0.1)

    def car_drive(self, model_nn):
        if self.first_frame:
            self.setup_first_frame(model_nn)
            self.set_label_right_status("Done")

        if model_nn == "dave2_const_v_b":
            self.execute_dave2_const_v_b()
        elif model_nn == "dave2_const_v_b_CIL":
            self.execute_dave2_const_v_b_CIL()

    def setup_first_frame(self, model_nn):
        self.first_frame = False
        self.vehicle = self.world.spawn_actor(self.model_3, self.all_spawn_points[int(self.spawn_point_id)])
        self.actor_list.append(self.vehicle)

        if model_nn == "dave2_const_v_b":
            self.setup_dave2_sensor()
            log_path = Path("../../logs")
            agent_path = Path("agent_dave2_const_v_b")
            version = 15
            checkpoint_path = log_path / agent_path / Path(f"version_{str(version)}/checkpoints")
            model_path = next(checkpoint_path.glob("*ckpt"))
            self.model = dave2_const_v_b_model.load_from_checkpoint(model_path, train_flag=False, version=version)
        elif model_nn == "dave2_const_v_b_CIL":
            self.setup_dave2_sensor()
            log_path = Path("../../logs")
            agent_path = Path("agent_dave2_const_v_b_CIL")
            version = 3
            checkpoint_path = log_path / agent_path / Path(f"version_{str(version)}/checkpoints")
            model_path = next(checkpoint_path.glob("*ckpt"))
            self.model = dave2_const_v_b_CIL_model.load_from_checkpoint(model_path, train_flag=False, version=version)

        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((66, 200)), transforms.ToTensor()])

        self.agent = BehaviorAgent(self.vehicle, behavior='cautious')
        self.agent.ignore_traffic_lights(active=True)
        self.agent.ignore_stop_signs(active=True)
        self.agent.ignore_vehicles(active=True)
        self.agent.set_destination(self.destination, start_location=self.start_location)

        for _ in range(20):
            self.world.tick()

    def setup_dave2_sensor(self):
        rgb_bp = self.blueprint_library.find("sensor.camera.rgb")
        rgb_bp.set_attribute('image_size_x', f"{IMAGE_WIDTH}")
        rgb_bp.set_attribute('image_size_y', f"{IMAGE_HEIGHT}")
        rgb_bp.set_attribute('fov', f"{FOV}")
        camera_center_transform = carla.Transform(carla.Location(x=1.0, z=1.5))
        self.center_sensor = self.world.spawn_actor(rgb_bp, camera_center_transform, attach_to=self.vehicle)
        self.actor_list.append(self.center_sensor)
        self.center_queue = queue.Queue()
        self.center_sensor.listen(self.center_queue.put)

    def execute_dave2_const_v_b(self):
        if self.center_queue is None:
            return
        try:
            center_frame = self.center_queue.get(True, 2.0)
        except queue.Empty:
            return

        self.agent._update_information()
        self.draw_route()

        if not self.car_go:
            control = self.agent.run_step()
            control.throttle = 0.0
            control.brake = 1.0
            self.vehicle.apply_control(control)
            return

        if self.agent.done():
            print("DESTINATION REACHED")

        image_center = raw_data_process(center_frame)
        control = self.agent.run_step()
        current_loc = self.vehicle.get_location()
        command_int = self.agent._local_planner.target_road_option

        steer = process_frame(image_center, command_int, self.transform, self.model, self.device)

        if command_int == 4:
            self.agent.set_destination(self.destination, start_location=current_loc)
        else:
            self.frame_number += 1
            if self.frame_number > 400 and abs(steer) < 0.05:
                self.frame_number = 0
                self.agent.set_destination(self.destination, start_location=current_loc)

        control.steer = float(np.clip(steer, -1.0, 1.0))
        control.throttle = 0.15
        control.brake = 0.0
        self.vehicle.apply_control(control)

    def execute_dave2_const_v_b_CIL(self):
        if self.center_queue is None:
            return
        try:
            center_frame = self.center_queue.get(True, 2.0)
        except queue.Empty:
            return

        self.agent._update_information()
        self.draw_route()

        if not self.car_go:
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.brake = 1.0
            self.vehicle.apply_control(control)
            return

        if self.agent.done():
            print("DESTINATION REACHED")

        image_center = raw_data_process(center_frame)
        control = self.agent.run_step()
        current_loc = self.vehicle.get_location()
        command_int = self.agent._local_planner.target_road_option

        steer = process_frame(image_center, command_int, self.transform, self.model, self.device)

        if command_int == 4:
            self.agent.set_destination(self.destination, start_location=current_loc)
        else:
            self.frame_number += 1
            if self.frame_number > 400 and abs(steer) < 0.05:
                self.frame_number = 0
                self.agent.set_destination(self.destination, start_location=current_loc)

        control.steer = float(np.clip(steer, -1.0, 1.0))

        control.throttle = 0.15
        control.brake = 0.0
        self.vehicle.apply_control(control)

        # velocity = self.vehicle.get_velocity()
        # speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        # print(speed, steer)


    def draw_route(self):
        route_queue = self.agent._local_planner._waypoints_queue
        if len(route_queue) > 0:
            for i, (waypoint, _) in enumerate(route_queue):
                if i > len(route_queue) - 1: break
                loc = waypoint.transform.location
                loc.z += 0.05
                self.world.debug.draw_string(loc, 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0),life_time=0.1)


if __name__ == "__main__":
    app = CarlaControlPanel()
    app.mainloop()