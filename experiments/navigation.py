import cv2
import torch
from torchvision import transforms
from pathlib import Path
import sys

root = "../.."
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.models.model_dave2_const_v_b import DrivingModel
from src.data.collect_data_dave2_const_v_b import set_target, IMAGE_WIDTH, IMAGE_HEIGHT, FOV, raw_data_process
import carla
import random
import queue
import numpy as np
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.local_planner import RoadOption


def main():
    try:
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

        #CONFIGURING SPAWN POINTS
        map = world.get_map()
        all_spawn_points = map.get_spawn_points()

        # SETTING SPECTATOR ON THE TOP LOOKING DOWN
        spectator.set_transform(carla.Transform(carla.Location(x=100, y=204, z=203.0),
                                                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)))

        waypoint_tuple_list = map.get_topology()
        while True:
            world.tick()

            for nod in waypoint_tuple_list:
                p1 = nod [0]
                p2 = nod [1]
                draw_pos_1 = p1.transform.location + carla.Location(z=0.5)
                draw_pos_2 = p2.transform.location + carla.Location(z=0.5)

                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)

                world.debug.draw_line(
                    draw_pos_1,
                    draw_pos_2,
                    thickness=0.1,
                    color=carla.Color(r, g, b),
                    life_time=0
                )

    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        print("SUCCESSFULLY EXECUTED")

if __name__ == "__main__":
    main()