import carla
import random
import queue
import sys
import os
import time
import cv2
import numpy as np
from carla import ColorConverter as cc
import csv
from pathlib import Path

CARLA_ROOT = "C:/CARLA_0.9.16/PythonAPI/carla"
sys.path.append(CARLA_ROOT)
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.local_planner import RoadOption


IMAGE_WIDTH = 450
IMAGE_HEIGHT = 250
FOV = 120
script_location = Path(__file__).resolve().parent
output_dir = script_location.parent / 'labels'

def obstacle_detection(vehicle, obstacle_list, dist_threshold = 5.0):
    v_loc = vehicle.get_location()
    v_trans = vehicle.get_transform()
    v_fwd = v_trans.get_forward_vector()

    for obs in obstacle_list:
        obs_loc = obs.get_location()
        dist = v_loc.distance(obs_loc)

        if dist < dist_threshold:
            vec_to_obs = carla.Vector3D(obs_loc.x - v_loc.x, obs_loc.y - v_loc.y, obs_loc.z - v_loc.z)
            length = np.sqrt(vec_to_obs.x ** 2 + vec_to_obs.y ** 2 + vec_to_obs.z ** 2)
            vec_to_obs.x /= length
            vec_to_obs.y /= length

            #ASSUMING Z IS CONST
            dot = v_fwd.x * vec_to_obs.x + v_fwd.y * vec_to_obs.y

            if dot > 0.9:
                return True, obs
    return False, None

def set_target(all_spawn_points, num_positions_to_spawn, num_obstacles, vehicle, world):
    destination = all_spawn_points[random.randint(0, num_positions_to_spawn) + num_obstacles].location
    current_loc = vehicle.get_location()
    start_waypoint = world.get_map().get_waypoint(current_loc)
    start_loc = start_waypoint.transform.location
    return destination, start_loc

def save_data(image_dir,writer, frame_number, image_array, control, command_int, speed):
    #SAVING IMAGE FROM DEPTH CAMERA
    save_dir = image_dir / f"{frame_number}.png"
    cv2.imwrite(str(save_dir), image_array)

    #ADDING DATA TO CSV
    writer.writerow([save_dir, control.steer, control.throttle, control.brake, command_int, speed])


def main():
    actor_list = []
    try:
        obstacle_list = []

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

        #ADDING DEPTH CAMERA
        depth_bp = blueprint_library.find("sensor.camera.depth")
        depth_bp.set_attribute('image_size_x', f"{IMAGE_WIDTH}")
        depth_bp.set_attribute('image_size_y', f"{IMAGE_HEIGHT}")
        depth_bp.set_attribute('fov', f"{FOV}")


        #SPAWNING OBSTACLES
        all_spawn_points = world.get_map().get_spawn_points()
        random.shuffle(all_spawn_points)
        num_obstacles = 20
        num_positions_to_spawn = len(all_spawn_points) - num_obstacles - 1

        # for i in range(num_obstacles):
        #     obs_position = all_spawn_points[i]
        #     obs = world.spawn_actor(model_3, obs_position)
        #     obs.set_simulate_physics(True)
        #     actor_list.append(obs)
        #     obstacle_list.append(obs)

        #SPAWNING VEHICLE
        car_rand = random.randint(0, num_positions_to_spawn) + num_obstacles
        start_position = all_spawn_points [car_rand]
        vehicle = world.spawn_actor(model_3, start_position)
        actor_list.append(vehicle)
        # start_position = random.choice(all_spawn_points)
        # vehicle = world.spawn_actor(model_3, start_position)
        # actor_list.append(vehicle)


        #SPAWNING DEPTH CAMERA
        camera_transform = carla.Transform(carla.Location(x = 1.5, z = 1.5))
        depth_sensor = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
        actor_list.append(depth_sensor)

        #QUEUE FOR PHOTOS
        sensor_queue = queue.Queue()
        depth_sensor.listen(sensor_queue.put)

        #ADDING AGENT
        agent = BehaviorAgent(vehicle, behavior='cautious')
        agent.ignore_traffic_lights(active = True)
        agent.ignore_stop_signs(active=True)
        agent.ignore_vehicles(active = True)

        #LETTIN CAR SPAWN
        for _ in range(20):
            world.tick()

        #CHOOSING DESTINATION
        destination, start_loc = set_target(all_spawn_points, num_positions_to_spawn, num_obstacles, vehicle, world)
        agent.set_destination(destination, start_location = start_loc)

        #SETTING SPECTATOR ON THE TOP LOOKING DOWN
        spectator.set_transform(carla.Transform(carla.Location(x = 100, y = 204, z = 203.0),
                                                carla.Rotation(pitch = -90.0, yaw = 0.0, roll = 0.0)))

        is_changing_lane = False

        #CATALOG PREPARATION
        image_dir = output_dir / 'images'
        image_dir.mkdir(parents = True, exist_ok = True)
        data_dir = output_dir / 'data.csv'
        csv_file = open(str(data_dir), 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(['image', 'steer', 'throttle', 'brake', 'command', 'speed'])

        frame_number = 0
        simulation_step = 0
        print("START")
        while True:
            # print(spectator.get_transform())
            world.tick()
            try:
                s_frame = sensor_queue.get(True, 1.0)
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

            if agent.done() and not is_changing_lane:
                print("DESTINATION REACHED")
                destination, start_loc = set_target(all_spawn_points, num_positions_to_spawn, num_obstacles, vehicle, world)
                agent.set_destination(destination, start_location=start_loc)
            elif agent.done() and is_changing_lane:
                is_changing_lane = False
                agent.set_destination(destination, start_location=start_loc)


            # detection, obstacle = obstacle_detection(vehicle, obstacle_list)
            # if detection and not is_changing_lane:
            #     is_changing_lane = True
                # agent.lane_change('right')
            # print(agent._local_planner.target_road_option)


            #APPLYING CONTROL TO AGENT
            control = agent.run_step()
            vehicle.apply_control(control)

            #CONVERTING DATA FROM DEPTH CAMERA
            s_frame.convert(cc.LogarithmicDepth)
            array = np.frombuffer(s_frame.raw_data, dtype = np.dtype("uint8"))
            array = np.reshape(array, (s_frame.height, s_frame.width, 4))
            array = array [:, :, :3]
            # array = array.copy() #TO PUT TEXT IN CV2

            #GATHERING INFORMATION AND SAVING DATA
            v = vehicle.get_velocity()
            v = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
            command_int = agent._local_planner.target_road_option
            if simulation_step % 5 == 0:
                save_data(image_dir, writer, frame_number, array, control, command_int, v)
                frame_number += 1
            simulation_step += 1

            #SHOWING IMAGE FROM DEPTH CAMERA
            # v = vehicle.get_velocity()
            # v_kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
            # cv2.putText(array, f"{v_kmh:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)
            # cv2.imshow("",array)
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