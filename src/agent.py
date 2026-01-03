import carla
import random
import queue
import sys
import os
import time
import cv2
import numpy as np
from carla import ColorConverter as cc

CARLA_ROOT = "C:/CARLA_0.9.16/PythonAPI/carla"
sys.path.append(CARLA_ROOT)
from agents.navigation.behavior_agent import BehaviorAgent


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FOV = 120

def obstacle_detection(vehicle, obstacle_list, dist_threshold = 10.0):
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

            #assuming z is const
            dot = v_fwd.x * vec_to_obs.x + v_fwd.y * vec_to_obs.y

            if dot > 0.7:
                return True, obs
    return False, None

def set_target(all_spawn_points, num_positions_to_spawn, num_obstacles, vehicle, world):
    destination = all_spawn_points[random.randint(0, num_positions_to_spawn) + num_obstacles].location
    current_loc = vehicle.get_location()
    start_waypoint = world.get_map().get_waypoint(current_loc)
    start_loc = start_waypoint.transform.location
    return destination, start_loc


def main():
    actor_list = []
    try:
        obstacle_list = []

        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        #connecting to world
        world = client.get_world()
        settings = world.get_settings()
        spectator = world.get_spectator()

        #applying synchronous mode
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.02
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        model_3 = blueprint_library.filter("model3")[0]

        #adding depth camera
        depth_bp = blueprint_library.find("sensor.camera.depth")
        depth_bp.set_attribute('image_size_x', f"{IMAGE_WIDTH}")
        depth_bp.set_attribute('image_size_y', f"{IMAGE_HEIGHT}")
        depth_bp.set_attribute('fov', f"{FOV}")


        #spawning obstacles
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

        #spawning vehicle
        car_rand = random.randint(0, num_positions_to_spawn) + num_obstacles
        start_position = all_spawn_points [car_rand]
        vehicle = world.spawn_actor(model_3, start_position)
        actor_list.append(vehicle)
        # start_position = random.choice(all_spawn_points)
        # vehicle = world.spawn_actor(model_3, start_position)
        # actor_list.append(vehicle)


        #spawning depth camera
        camera_transform = carla.Transform(carla.Location(x = 1.5, z = 1.5))
        depth_sensor = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
        actor_list.append(depth_sensor)

        #queue for photos
        sensor_queue = queue.Queue()
        depth_sensor.listen(sensor_queue.put)

        #adding agent
        agent = BehaviorAgent(vehicle, behavior='normal')
        agent.ignore_traffic_lights(active = True)
        agent.ignore_stop_signs(active=True)
        agent.ignore_vehicles(active = True)

        #lettin car spawn
        for _ in range(20):
            world.tick()
        print("START")

        #choosing destination
        agent.set_destination(*set_target(all_spawn_points, num_positions_to_spawn, num_obstacles, vehicle, world))

        #setting spectator on the top looking down
        spectator.set_transform(carla.Transform(carla.Location(x = 6.0, y = 14.5, z = 219.0),
                                                carla.Rotation(pitch = -90.0, yaw = 90.0, roll = 0.0)))

        while True:
            world.tick()
            try:
                s_frame = sensor_queue.get(True, 1.0)
            except queue.Empty:
                continue

            agent._update_information()

            #drawing found route
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
                agent.set_destination(*set_target(all_spawn_points, num_positions_to_spawn, num_obstacles, vehicle, world))



            #applying control to agent
            control = agent.run_step()
            vehicle.apply_control(control)

            #converting data from Depth Camera
            s_frame.convert(cc.LogarithmicDepth)
            array = np.frombuffer(s_frame.raw_data, dtype = np.dtype("uint8"))
            array = np.reshape(array, (s_frame.height, s_frame.width, 4))
            array = array [:, :, :3]
            array = array.copy() #to put text in cv2

            v = vehicle.get_velocity()
            v_kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
            cv2.putText(array, f"{v_kmh:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)
            cv2.imshow("",array)
            cv2.waitKey(1)

    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        for actor in actor_list:
            actor.destroy()
        cv2.destroyAllWindows()
        print("SUCCESSFULLY EXECUTED")


if __name__ == "__main__":
    main()