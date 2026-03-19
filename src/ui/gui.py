import customtkinter as ctk
import carla
import threading
import queue
from pathlib import Path
import torch
from torchvision import transforms
from src.models.model_dave2_const_v_b import DrivingModel as dave2_const_v_b_model
from src.models.model_dave2_const_v_b_CIL import DrivingModel as dave2_const_v_b_CIL_model
from src.data.collect_data_dave2_const_v_b import IMAGE_WIDTH, IMAGE_HEIGHT, FOV, raw_data_process
from src.driving.load_model_dave2_const_v_b_to_carla import process_frame
from agents.navigation.behavior_agent import BehaviorAgent
import numpy as np
import math
import cv2
from PIL import Image
import random

TOTAL_SPAWN_POINTS = 100
SLIDER_WIDTH = 700

class PID_controller:
    def __init__(self, Kp = 0.1, Ki = 0.05, Kd = 0.1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0.0
        self.integral = 0.0

    def run_step(self, target_speed, current_speed):
        error = target_speed - current_speed
        P = error * self.Kp

        self.integral += error
        #Anti wind-up
        self.integral = max(min(self.integral, 10), -10)
        I = self.integral * self.Ki

        D = self.Kd * (error - self.prev_error)
        self.prev_error = error

        signal = P + I + D

        if signal > 0:
            throttle = np.clip(signal, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip((-0.1) * signal, 0.0, 1.0)

        return throttle, brake

class ViewFromCar(ctk.CTkToplevel):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        font_name = "TimesNewRoman"
        self.geometry("700x400")

        self.grid_rowconfigure((0,2,3), weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure((0,1,2,3), weight = 1)

        self.label = ctk.CTkLabel(self, text = "View from a car", font=(font_name, 20, "bold"))
        self.label.grid(row = 0, column = 0, columnspan = 4, sticky = "n")

        self.image_label = ctk.CTkLabel(self,text="")
        self.image_label.grid(row = 1, column = 0, columnspan = 4, sticky = "nsew")

        #Viewing basic info
        self.speed_label = ctk.CTkLabel(self, text = "Speed: 0.0", font=(font_name, 20, "bold"))
        self.speed_label.grid(row = 2, column = 0, sticky = "s")

        self.throttle_label = ctk.CTkLabel(self, text = "Throttle: 0.0", font=(font_name, 20, "bold"))
        self.throttle_label.grid(row = 2, column = 1, sticky = "s")

        self.brake_label = ctk.CTkLabel(self, text = "Brake: 0.0", font=(font_name, 20, "bold"))
        self.brake_label.grid(row = 2, column = 2, sticky = "s")

        self.steer_label = ctk.CTkLabel(self, text = "Steer: 0.0", font=(font_name, 20, "bold"))
        self.steer_label.grid(row = 2, column = 3, sticky = "s")

        #Buttons to switch car mode
        self.button_autopilot = ctk.CTkButton(self, text = "Autopilot", fg_color = "purple", command = self.autopilot,
                                              font=(font_name, 15, "bold"))
        self.button_autopilot.grid(row = 3, column = 0, columnspan = 2, sticky = "n")

        self.button_manual = ctk.CTkButton(self, text = "Manual", fg_color = "purple", command = self.manual,
                                              font=(font_name, 15, "bold"))
        self.button_manual.grid(row = 3, column = 2, columnspan = 2, sticky = "n")

        #Manual control config
        self.wsad = {'w': False, 's': False, 'a': False, 'd': False, 'q':False}
        self.bind("<KeyPress>", self.key_pressed)
        self.bind("<KeyRelease>", self.key_released)

    def key_pressed(self, event):
        key = event.char.lower()
        if key in self.wsad:
            self.wsad[key] = True

    def key_released(self, event):
        key = event.char.lower()
        if key in self.wsad:
            self.wsad[key] = False

    def update_frame(self, camera_image):
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(camera_image)
        self.ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(600,400))
        self.image_label.configure(image=self.ctk_image)

    def update_panel_info(self, speed, throttle, brake, steer):
        self.speed_label.configure(text = str(f"Speed: {speed:5.1f}"))
        self.throttle_label.configure(text=str(f"Throttle: {throttle:4.2f}"))
        self.brake_label.configure(text=str(f"Brake: {brake:4.2f}"))
        self.steer_label.configure(text=str(f"Steer: {steer: 5.2f}"))

    def autopilot(self):
        self.parent.autopilot_on = True
        self.parent.pid_controller.prev_error = 0.0
        self.parent.pid_controller.integral = 0.0

        #Calculating new navigation when turning on autopilot
        if self.parent.nn_car and hasattr(self.parent, 'agent') and hasattr(self.parent, 'vehicle'):
            current_loc = self.parent.vehicle.get_location()
            self.parent.agent.set_destination(self.parent.destination, start_location=current_loc)

    def manual(self):
        self.parent.autopilot_on = False

class CarlaControlPanel(ctk.CTk):
    def __init__(self):
        super().__init__()
        font_name = "TimesNewRoman"
        #Flags
        self.carla_running = False          #Controls the main simulation loop execution
        self.spawn_point_draw = True        #Controls visualization of the spawn point
        self.dest_point_draw = True         #Controls visualization of the destination point
        # self.spawn_car = False            #Controls spawning a car
        self.first_frame = True             #Flag for init in the first frame
        self.nn_car = False                 #Activates neural network driving model
        self.car_go = False                 #Controls vehicle movement
        self.reset_flag = False             #Indicates whether to reset
        self.view_window = False            #Indicates whether top level windows exists
        self.autopilot_on = True            #Activates autopilot
        self.obstacle_detect = False        #Checks whether there is an obstacle in front of a car or in past 200 frames
        self.reverse_gear = False           #Toggle reverse
        self.prev_q_state = False           #States whether one frame before "q" was pressed
        self.obstacle_current = False       #Flag is active only when it sees obstacle in current frame

        #Values for panel info
        self.speed = 0.0
        self.throttle_pid = 0.0
        self.brake_pid = 0.0
        self.steer = 0.0

        self.geometry("450x550")
        self.title("Carla control panel")

        self.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7), weight=1)
        self.grid_columnconfigure((0, 1), weight=1)

        #Label for title
        self.label_title_frame = ctk.CTkFrame(self)
        self.label_title_frame.grid(row=0, column=0, columnspan=2, sticky="n")
        self.label_title = ctk.CTkLabel(self.label_title_frame, text="Route Configuration in CARLA",
                                        font=(font_name, 20, "bold"))
        self.label_title.pack(pady=0)

        #Label for slider
        self.label_slider_spawn = ctk.CTkLabel(self, text="Spawn point ID: 0 ", font=(font_name, 15, "bold"), text_color="red")
        self.label_slider_spawn.grid(row=1, column=0, columnspan=2, sticky="ns" )

        #Slider
        self.spawn_slider = ctk.CTkSlider(self, from_=0, to=TOTAL_SPAWN_POINTS,
                                          number_of_steps=TOTAL_SPAWN_POINTS, command=self.spawn_slider_update_value,
                                          width=SLIDER_WIDTH)
        self.spawn_slider.set(0)
        self.spawn_point_id = 0
        self.spawn_slider.grid(row=2, column=0, columnspan=2, sticky="n")

        #Label for slider
        self.label_slider_dest = ctk.CTkLabel(self, text="Destination point ID: 0 ", font=(font_name, 15, "bold"), text_color="lightgreen")
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

        self.button_view = ctk.CTkButton(self.frame_right, text="View", fg_color="orange", command = self.view_from_car
                                         , font=(font_name, 15, "bold"))
        self.button_view.pack(pady=10)

        self.label_right_status = ctk.CTkLabel(self.frame_right, text="", font=(font_name, 15, "bold"))
        self.label_right_status.pack(pady=10)

        #Label for combo box
        self.frame_combo_box = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_combo_box.grid(row=6, column=0, sticky="s")
        self.label_combo_box = ctk.CTkLabel(self.frame_combo_box, text="Choose AI model:",
                                        font=(font_name, 15, "bold"), fg_color="transparent")
        self.label_combo_box.pack(pady=0)

        #Combo box for model to choose
        option_list = ["dave2_const_v_b_CIL","dave2_const_v_b"]
        self.combo_box = ctk.CTkComboBox(self, values = option_list, state = "readonly")
        self.combo_box.grid(row=7,column = 0, sticky = "n")
        self.combo_box.set("dave2_const_v_b_CIL")

        #Button for spawning obstacles
        self.button_obstacles = ctk.CTkButton(self, text="Spawn obstacles", font=(font_name, 15, "bold"), fg_color="purple", command = self.spawn_obstacles)
        self.button_obstacles.grid(row = 7, column = 1, sticky = "n")

        self.toplevel_window = None
        self.current_frame = None
        self.update_top_level_frame()

    def is_road_straight(self, transform, distance = 20.0, tolerance = 5.0):
        """
        Checks if the road is straight
        """

        waypoint = self.world.get_map().get_waypoint(transform.location)
        prev_waypoints = waypoint.previous(distance)
        next_waypoints = waypoint.next(distance)

        if not next_waypoints or not prev_waypoints:
            return False

        prev_waypoint = prev_waypoints[0]
        next_waypoint = next_waypoints[0]

        yaw_1_previous = waypoint.transform.rotation.yaw
        yaw_2_previous = prev_waypoint.transform.rotation.yaw
        diff_previous = abs(yaw_1_previous - yaw_2_previous)
        diff_previous = diff_previous % 360
        if diff_previous > 180:
            diff_previous = 360 - diff_previous

        yaw_1_next = waypoint.transform.rotation.yaw
        yaw_2_next = next_waypoint.transform.rotation.yaw
        diff_next = abs(yaw_1_next - yaw_2_next)
        diff_next = diff_next % 360
        if diff_next > 180:
            diff_next = 360 - diff_next

        #If diff is smaller than the tolerance True is returned
        return diff_next < tolerance and diff_previous < tolerance

    def spawn_obstacles(self, max_obstacles = 10):
        """
        Spawns obstacles only on straight roads; minimum distance between each obstacle is 10.0 m
        """
        if not self.carla_running or self.nn_car:
            return
        possible_obstacles = []

        for spawnpoint in self.all_spawn_points:
            if self.is_road_straight(spawnpoint):

                possible_obstacles.append(spawnpoint)


        # self.obstacles_list = np.random.choice(np.arange(0, len(possible_obstacles),1), 10, replace=False)
        obstacles_num = 0
        self.obstacles_list = []

        while obstacles_num < max_obstacles:
            idx = random.randint(0,len(possible_obstacles) - 1)

            if idx in self.obstacles_list: continue

            flag_dist = False
            for obstacle_idx in self.obstacles_list:
                if possible_obstacles[idx].location.distance(possible_obstacles[obstacle_idx].location) < 30.0:
                    flag_dist = True
                    continue

            if flag_dist: continue

            self.obstacles_list.append(idx)
            obstacles_num += 1


        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()

        for index in self.obstacles_list:
            transform = possible_obstacles[index]
            obstacle = random.choice(self.blueprint_library.filter('vehicle.mini.cooper_s_2021'))
            actor = self.world.try_spawn_actor(obstacle, transform)

            if actor is not None:
                self.actor_list.append(actor)


    def update_top_level_frame(self):
        """
        Updates view from a car
        """
        if self.view_window and self.current_frame is not None and self.toplevel_window is not None:
            self.toplevel_window.update_frame(self.current_frame)
            self.toplevel_window.update_panel_info(self.speed, self.throttle_pid, self.brake_pid, self.steer)
        self.after(30, self.update_top_level_frame)

    def view_from_car(self):
        if not self.nn_car:
            return

        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = ViewFromCar(self)
            self.view_window = True

            #Closing function
            def on_close():
                self.view_window = False
                self.toplevel_window.destroy()
                self.toplevel_window = None

            self.toplevel_window.protocol("WM_DELETE_WINDOW", on_close)
        else:
            self.toplevel_window.focus()

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
            model_nn = self.combo_box.get()
            self.t = threading.Thread(target=self.carla_thread, daemon=True, kwargs = {"model_nn": model_nn})
            self.t.start()

    def stop_simulation(self):
        self.carla_running = False
        self.set_label_left_status("In progress...")

    def spawn_car_on_road(self):
        if not self.carla_running or self.nn_car:
            return
        self.spawn_point_draw = False
        self.dest_point_draw = False
        self.nn_car = True
        self.first_frame = True
        self.set_label_right_status("Spawning a car...")

    def start_car(self):
        if not self.nn_car:
            return
        self.car_go = True
        self.set_label_right_status("Driving")

    def stop_car(self):
        if not self.nn_car:
            return
        self.car_go = False
        self.set_label_right_status("Stopped")

    def reset(self):
        if self.carla_running:
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
        self.autopilot_on = True
        self.spawn_point_id = 0
        self.dest_point_id = 0
        self.spawn_slider.set(0)
        self.dest_slider.set(0)
        self.spawn_slider_update_value(0)
        self.dest_slider_update_value(0)
        self.set_label_right_status("Reset done")

    def carla_thread(self, model_nn):
        self.actor_list = []
        try:
            self.setup_carla_world()
            self.set_label_left_status("Connected")
            self.frame_number = 0
            while self.carla_running:
                self.world.tick()
                if not self.reset_flag:
                    if self.autopilot_on:
                        if self.nn_car:
                            self.car_drive(model_nn)
                    elif not self.autopilot_on:
                        self.car_drive_manual()

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
        loc.z += 0.05
        self.world.debug.draw_string(loc, 'Here', color=carla.Color(r=255, g=0, b=0),life_time=0.1)

    def draw_dest_point(self):
        loc_raw = self.all_spawn_points[int(self.dest_point_id)].location
        self.destination = carla.Location(loc_raw)
        loc = carla.Location(loc_raw)
        loc.z += 0.05
        self.world.debug.draw_string(loc, 'Here', color=carla.Color(r=0, g=255, b=0),life_time=0.1)

    def car_drive(self, model_nn):
        if self.first_frame:
            self.setup_first_frame(model_nn)
            self.set_label_right_status("Done")

        if model_nn == "dave2_const_v_b" or model_nn == "dave2_const_v_b_CIL":
            self.execute_dave2_const_v_b()


    def setup_first_frame(self, model_nn):
        self.first_frame = False
        self.vehicle = self.world.spawn_actor(self.model_3, self.all_spawn_points[int(self.spawn_point_id)])
        self.actor_list.append(self.vehicle)
        self.pid_controller = PID_controller()
        self.setup_lidar_sensor()

        #It is only that lidar doesn't crash in the first frame
        self.command_int = 4

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

    def setup_lidar_sensor(self):
        #Number of frames since last obstacle was detected
        self.frames_with_obstacle_detected = 0
        self.prev_obstacle_state = False
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('horizontal_fov', '50')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '80000')
        lidar_bp.set_attribute('dropoff_general_rate', '0.05')
        lidar_bp.set_attribute('dropoff_intensity_limit', '0.1')

        #Located at the front of the car
        transform = carla.Transform(carla.Location(x=1.8, z=1.5))
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, transform, attach_to=self.vehicle)
        self.actor_list.append(self.lidar_sensor)
        self.lidar_sensor.listen(lambda point_cloud: self.process_lidar_data(point_cloud))

    def process_lidar_data(self, point_cloud_data):
        """
        current_frame_obstacle states whether in current frame obstacle is detected
        self.obstacle_detect states whether in the last 200 frames obstacle was detected
        This ensures stable lane-changing maneuver and prevents too quick return to the original lane
        """
        min_dist = 0.2
        max_dist = 15.0
        vehicle_width = 1.3
        sensor_height = 1.2
        min_points_threshold = 450
        max_frames = 200

        sum_dist = 0
        num_el = 0
        for detection in point_cloud_data:
            x = detection.point.x
            y = detection.point.y
            z = detection.point.z

            if x <= 0: continue
            if abs(y) > vehicle_width: continue
            if z < -sensor_height: continue
            if z > 1.0: continue
            dist = math.sqrt(x ** 2 + y ** 2 + z ** 2)

            if dist > min_dist and dist < max_dist:
                sum_dist += dist
                num_el += 1
        current_frame_obstacle = False

        if num_el >= min_points_threshold:
            mean_dist = sum_dist / num_el
            if mean_dist < 10.0:
                current_frame_obstacle = True

        #When car is on a crossroad it doesn't detect obstacles to avoid changing a lane
        if self.command_int != 4:
            self.obstacle_detect = False
            self.prev_obstacle_state = False
            self.obstacle_current = False
            return

        #When car is avoiding obstacles it doesn't detect other obstacles
        if self.prev_obstacle_state and self.frames_with_obstacle_detected > 0:
            current_frame_obstacle = False

        if current_frame_obstacle:
            self.obstacle_detect = True
            self.obstacle_current = True
            print("Obstacle")
            self.frames_with_obstacle_detected = 0

        elif self.prev_obstacle_state:
            self.frames_with_obstacle_detected += 1
            if self.frames_with_obstacle_detected < max_frames:
                self.obstacle_detect = True
                self.obstacle_current = False
                print("Past obstacle")
            else:
                self.obstacle_current = False
                self.obstacle_detect = False
                self.frames_with_obstacle_detected = 0
        else:
            self.obstacle_current = False
            self.obstacle_detect = False
        self.prev_obstacle_state = self.obstacle_detect
        return

    def execute_dave2_const_v_b(self):
        # Same for const_v_b and const_v_b_CIL
        if self.center_queue is None:
            return
        try:
            center_frame = self.center_queue.get(True, 2.0)
        except queue.Empty:
            return

        self.agent._update_information()
        self.draw_route()

        image_center = raw_data_process(center_frame)
        if self.view_window:
            self.current_frame = image_center

        if not self.car_go:
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.brake = 1.0
            self.vehicle.apply_control(control)
            return

        control = self.agent.run_step()
        current_loc = self.vehicle.get_location()
        self.command_int = self.agent._local_planner.target_road_option
        vehicle_roll = self.vehicle.get_transform().rotation.roll

        # When obstacle is detected car changes the lane
        self.steer = process_frame(image_center, self.command_int, self.transform, self.model, self.device, mirror=self.obstacle_detect)
        if self.obstacle_detect:
            self.steer *= (-1)

        if self.command_int == 4:
            self.agent.set_destination(self.destination, start_location=current_loc)
        else:
            self.frame_number += 1
            if self.frame_number > 200 and abs(self.steer) < 0.05:
                self.frame_number = 0
                self.agent.set_destination(self.destination, start_location=current_loc)

        self.steer = float(np.clip(self.steer, -1.0, 1.0))

        # If car drives on a sidewalk
        if vehicle_roll < -1.0:
            self.steer = -0.2
        elif vehicle_roll > 1.0:
            self.steer = 0.2

        if self.obstacle_current:
            self.steer = -0.8

        control.steer = self.steer

        # Calculating throttle and brake using PID
        v = self.vehicle.get_velocity()
        self.speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        if abs(self.steer) < 0.15 and not self.obstacle_detect:
            # When a car is driving straight
            target_speed = 10.0
        elif not self.obstacle_detect:
            # When a car is turning
            target_speed = 7.0
        elif self.obstacle_current:
            # When an obstacle is detected in front of car
            target_speed = 2.0
        else:
            # When car is driving on the left lane, while avoiding obstacle
            target_speed = 4.0

        self.throttle_pid, self.brake_pid = self.pid_controller.run_step(target_speed=target_speed,
                                                                         current_speed=self.speed)

        # If red light is deteceted => stop
        traffic_light = self.vehicle.get_traffic_light()
        if traffic_light and traffic_light.get_state() == carla.TrafficLightState.Red:
            self.brake_pid = 1.0
            self.throttle_pid = 0.0

        control.throttle = self.throttle_pid
        control.brake = self.brake_pid

        self.vehicle.apply_control(control)

    def car_drive_manual(self):
        if self.center_queue is None or self.toplevel_window is None:
            return
        try:
            center_frame = self.center_queue.get(True, 2.0)
        except queue.Empty:
            return

        image_center = raw_data_process(center_frame)
        if self.view_window:
            self.current_frame = image_center

        if not self.car_go:
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.brake = 1.0
            self.vehicle.apply_control(control)
            return

        control = carla.VehicleControl()

        #Gear changes only when the "q" is being pressed
        current_q = self.toplevel_window.wsad['q']
        if current_q and not self.prev_q_state:
            self.reverse_gear = not self.reverse_gear
        if self.reverse_gear:
            control.reverse = True

        if self.toplevel_window.wsad['w']:
            #When reverse gear, more throttle is required
            if self.reverse_gear:
                self.throttle_pid = 0.3
            else:
                self.throttle_pid = 0.2
        else:
            self.throttle_pid = 0.0

        if self.toplevel_window.wsad['a']:
            self.steer = -0.2
        elif self.toplevel_window.wsad['d']:
            self.steer = 0.2
        else:
            self.steer = 0.0

        if self.toplevel_window.wsad['s']:
            self.brake_pid = 0.5
        else:
            self.brake_pid = 0.0


        v = self.vehicle.get_velocity()
        self.speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        control.steer = self.steer
        control.throttle = self.throttle_pid
        control.brake = self.brake_pid
        self.vehicle.apply_control(control)
        self.prev_q_state = current_q

    def draw_route(self):
        route_queue = self.agent._local_planner._waypoints_queue
        if len(route_queue) > 2:
            for i, (waypoint, _) in enumerate(route_queue):
                if i > len(route_queue) - 1: break
                loc = waypoint.transform.location
                loc.z += 0.05
                self.world.debug.draw_string(loc, 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0),life_time=0.1)
        else:
            #If destination is reached
            self.car_go = False
            self.set_label_right_status("Destination reached")


if __name__ == "__main__":
    app = CarlaControlPanel()
    app.mainloop()