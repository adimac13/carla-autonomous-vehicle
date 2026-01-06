#Model based on NVIDIA's DAVE-2 with constant throttle=0.18 and brake=0.0
import carla
import random
import queue
import sys
import os
import time
import cv2
import numpy as np
import csv
from pathlib import Path
import shutil

CARLA_ROOT = "C:/CARLA_0.9.16/PythonAPI/carla"
sys.path.append(CARLA_ROOT)
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.local_planner import RoadOption

IMAGE_WIDTH = 500
IMAGE_HEIGHT = 300
FOV = 100
STEER_CORRECTION = 0.08
output_dir = Path("../../labels/dave2_const_v_s")

def set_target(all_spawn_points, num_positions_to_spawn, num_obstacles, vehicle, world):
    destination = all_spawn_points[random.randint(0, num_positions_to_spawn) + num_obstacles].location
    current_loc = vehicle.get_location()
    start_waypoint = world.get_map().get_waypoint(current_loc)
    start_loc = start_waypoint.transform.location
    return destination, start_loc

def save_data(image_dir,writer, frame_number, image_left,image_center, image_right, control, command_int, speed):
    #SAVING IMAGE FROM CENTER DEPTH CAMERA
    save_dir = image_dir / f"{frame_number}_center.png"
    cv2.imwrite(str(save_dir), image_center)
    writer.writerow([save_dir, control.steer, control.throttle, control.brake, command_int, speed])

    # SAVING IMAGE FROM LEFT DEPTH CAMERA
    save_dir = image_dir / f"{frame_number}_left.png"
    cv2.imwrite(str(save_dir), image_left)
    steer = min(1, control.steer + STEER_CORRECTION)
    writer.writerow([save_dir, steer, control.throttle, control.brake, command_int, speed])

    # SAVING IMAGE FROM RIGHT DEPTH CAMERA
    save_dir = image_dir / f"{frame_number}_right.png"
    cv2.imwrite(str(save_dir), image_right)
    steer = max(-1, control.steer - STEER_CORRECTION)
    writer.writerow([save_dir, steer, control.throttle, control.brake, command_int, speed])

def raw_data_process(frame):
    im = np.array(frame.raw_data)
    im = np.reshape(im, (frame.height, frame.width, 4))
    im = im[:, :, :3]
    return im.copy()


def main():
    actor_list = []
    try:
        weather_presets = [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.CloudyNoon,
            carla.WeatherParameters.WetNoon,
            carla.WeatherParameters.HardRainNoon,
            carla.WeatherParameters.ClearSunset,
            carla.WeatherParameters.WetCloudySunset
        ]

        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        #CONNECTING TO WORLD
        world = client.load_world('Town02')
        settings = world.get_settings()
        spectator = world.get_spectator()

        #APPLYING SYNCHRONOUS MODE
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        model_3 = blueprint_library.filter("model3")[0]

        #ADDING RGB CAMERAS
        rgb_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_bp.set_attribute('image_size_x', f"{IMAGE_WIDTH}")
        rgb_bp.set_attribute('image_size_y', f"{IMAGE_HEIGHT}")
        rgb_bp.set_attribute('fov', f"{FOV}")


        #SPAWNING OBSTACLES
        all_spawn_points = world.get_map().get_spawn_points()
        random.shuffle(all_spawn_points)
        num_obstacles = 20
        num_positions_to_spawn = len(all_spawn_points) - num_obstacles - 1

        #SPAWNING VEHICLE
        car_rand = random.randint(0, num_positions_to_spawn) + num_obstacles
        start_position = all_spawn_points [car_rand]
        vehicle = world.spawn_actor(model_3, start_position)
        actor_list.append(vehicle)

        #SPAWNING RGB CAMERAS
        camera_x_offset = 1.0
        camera_y_offset = 1.0
        camera_z_offset = 1.5
        camera_center_transform = carla.Transform(carla.Location(x = camera_x_offset, z = camera_z_offset))
        center_sensor = world.spawn_actor(rgb_bp, camera_center_transform, attach_to=vehicle)
        actor_list.append(center_sensor)

        camera_left_transform = carla.Transform(carla.Location(x = camera_x_offset,y = -camera_y_offset, z = camera_z_offset))
        left_sensor = world.spawn_actor(rgb_bp, camera_left_transform, attach_to=vehicle)
        actor_list.append(left_sensor)

        camera_right_transform = carla.Transform(carla.Location(x = camera_x_offset,y = camera_y_offset, z = camera_z_offset))
        right_sensor = world.spawn_actor(rgb_bp, camera_right_transform, attach_to=vehicle)
        actor_list.append(right_sensor)

        #QUEUE FOR PHOTOS
        center_queue = queue.Queue()
        center_sensor.listen(center_queue.put)
        left_queue = queue.Queue()
        left_sensor.listen(left_queue.put)
        right_queue = queue.Queue()
        right_sensor.listen(right_queue.put)

        #ADDING AGENT
        agent = BehaviorAgent(vehicle, behavior='cautious')
        agent.ignore_traffic_lights(active = True)
        agent.ignore_stop_signs(active=True)
        agent.ignore_vehicles(active = True)

        #LETTING CAR SPAWN
        for _ in range(20):
            world.tick()

        #CHOOSING DESTINATION
        destination, start_loc = set_target(all_spawn_points, num_positions_to_spawn, num_obstacles, vehicle, world)
        agent.set_destination(destination, start_location = start_loc)

        #SETTING SPECTATOR ON THE TOP LOOKING DOWN
        spectator.set_transform(carla.Transform(carla.Location(x = 100, y = 204, z = 203.0),
                                                carla.Rotation(pitch = -90.0, yaw = 0.0, roll = 0.0)))

        #CATALOG PREPARATION
        image_dir = output_dir / 'images'
        # print(f"Saving to {str(image_dir)}")

        #Deletes images if necessary
        if image_dir.exists():
            shutil.rmtree(image_dir)
        image_dir.mkdir(parents = True, exist_ok = True)

        data_dir = output_dir / 'annotations.csv'
        csv_file = open(str(data_dir), 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(['image', 'steer', 'throttle', 'brake', 'command', 'speed'])

        frame_number = 0
        simulation_step = 0
        command4_step = 0
        print("START")
        while True:
            world.tick()
            try:
                center_frame = center_queue.get(True, 2.0)
                left_frame = left_queue.get(True, 2.0)
                right_frame = right_queue.get(True, 2.0)
            except queue.Empty:
                continue

            agent._update_information()

            #DRAWING FOUND ROUTE
            route_queue = agent._local_planner._waypoints_queue
            if len(route_queue) > 0:
                for i, (waypoint, _) in enumerate(route_queue):
                    if i > len(route_queue)-1: break
                    loc = waypoint.transform.location
                    loc.z += 1.0
                    world.debug.draw_string(
                        loc,
                        'O',
                        draw_shadow=False,
                        color=carla.Color(r=0, g=255, b=0),
                        life_time=0.1,
                        persistent_lines=True
                    )

            if agent.done():
                print("DESTINATION REACHED")
                destination, start_loc = set_target(all_spawn_points, num_positions_to_spawn, num_obstacles, vehicle, world)
                agent.set_destination(destination, start_location=start_loc)

            #APPLYING CONTROL TO AGENT
            control = agent.run_step()
            #Const throttle and brake
            control.throttle = 0.18
            control.brake = 0.0
            vehicle.apply_control(control)


            #CONVERTING DATA FROM DEPTH CAMERA
            image_left = raw_data_process(left_frame)
            image_center = raw_data_process(center_frame)
            image_right = raw_data_process(right_frame)

            #GATHERING INFORMATION AND SAVING DATA
            v = vehicle.get_velocity()
            v = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
            command_int = agent._local_planner.target_road_option

            #If car goes straight save 1 in 5 images
            if command_int == 4:
                if command4_step % 6 == 0:
                    save_data(image_dir, writer, frame_number, image_left, image_center, image_right, control, command_int, v)
                command4_step += 1
                frame_number += 1

            else:
                save_data(image_dir, writer, frame_number, image_left,image_center, image_right, control, command_int, v)
                frame_number += 1


            # if not(simulation_step % 1000):
            #     selected_weather = random.choice(weather_presets)
            #     world.set_weather(selected_weather)
            frame_number += 1

            simulation_step += 1
            #SHOWING IMAGE FROM RGB CAMERA
            # cv2.putText(image_center, f"{v:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)
            # cv2.imshow("left",image_left)
            # cv2.imshow("right", image_right)
            # cv2.imshow("center", image_center)
            # cv2.waitKey(1)


    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        for actor in actor_list:
            actor.destroy()
        cv2.destroyAllWindows()
        csv_file.close()
        print("SUCCESSFULLY EXECUTED")


if __name__ == "__main__":
    main()