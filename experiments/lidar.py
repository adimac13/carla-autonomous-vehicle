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
import math

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


def process_lidar_data(point_cloud_data):
    min_dist= 0.2
    max_dist = 10.0
    vehicle_width = 1.0
    sensor_height = 1.2

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
    if num_el < 20:
        return False

    mean_dist = sum_dist / num_el

    if mean_dist  < 5.0:
        print(f"Obstacle!! {mean_dist}")
        return True


    return False



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

        #SPAWNING OBSTACLES
        all_spawn_points = world.get_map().get_spawn_points()
        random.shuffle(all_spawn_points)
        num_obstacles = 20
        num_positions_to_spawn = len(all_spawn_points) - num_obstacles - 1

        for i in range(num_obstacles):
            obs_position = all_spawn_points[i]
            obs = world.spawn_actor(model_3, obs_position)
            obs.set_simulate_physics(True)
            actor_list.append(obs)
            obstacle_list.append(obs)

        #SPAWNING VEHICLE
        car_rand = random.randint(0, num_positions_to_spawn) + num_obstacles
        start_position = all_spawn_points [car_rand]
        vehicle = world.spawn_actor(model_3, start_position)
        actor_list.append(vehicle)

        lidar_blueprint = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_blueprint.set_attribute('horizontal_fov', '60')
        lidar_blueprint.set_attribute('range', '20')
        lidar_blueprint.set_attribute('channels', '16')
        lidar_blueprint.set_attribute('points_per_second', '20000')

        # Montaż z przodu na masce (z=1.0)
        transform = carla.Transform(carla.Location(x=1.8, z=1.5))

        sensor = world.spawn_actor(lidar_blueprint, transform, attach_to=vehicle)
        sensor.listen(lambda point_cloud: process_lidar_data(point_cloud))

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

        while True:
            world.tick()
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
                destination, start_loc = set_target(all_spawn_points, num_positions_to_spawn, num_obstacles, vehicle, world)
                agent.set_destination(destination, start_location=start_loc)

            #APPLYING CONTROL TO AGENT
            control = agent.run_step()
            vehicle.apply_control(control)


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
        # csv_file.close()
        print("SUCCESSFULLY EXECUTED")


if __name__ == "__main__":
    main()