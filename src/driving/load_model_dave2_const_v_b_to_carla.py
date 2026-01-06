#Model based on NVIDIA's DAVE-2 with constant throttle=0.18 and brake=0.0
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
from carla import ColorConverter as cc


def process_frame(camera_image, command):
    camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2YUV)
    image_tensor = transform(camera_image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    command_tensor = torch.tensor(command, dtype = torch.long).to(device)

    with torch.no_grad():
        outputs = model(image_tensor, command_tensor)
        #Model returns => steer

    steer = outputs.item()

    return steer

def world_setup():
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

        #CONFIGURING SPAWN POINTS
        all_spawn_points = world.get_map().get_spawn_points()
        random.shuffle(all_spawn_points)
        num_obstacles = 20
        num_positions_to_spawn = len(all_spawn_points) - num_obstacles - 1

        # SPAWNING VEHICLE
        car_rand = random.randint(0, num_positions_to_spawn) + num_obstacles
        start_position = all_spawn_points[car_rand]
        vehicle = world.spawn_actor(model_3, start_position)
        actor_list.append(vehicle)

        #ADDING RGB CAMERAS
        rgb_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_bp.set_attribute('image_size_x', f"{IMAGE_WIDTH}")
        rgb_bp.set_attribute('image_size_y', f"{IMAGE_HEIGHT}")
        rgb_bp.set_attribute('fov', f"{FOV}")

        #SPAWNING RGB CAMERAS
        camera_x_offset = 1.0
        camera_z_offset = 1.5

        camera_center_transform = carla.Transform(carla.Location(x = camera_x_offset, z = camera_z_offset))
        center_sensor = world.spawn_actor(rgb_bp, camera_center_transform, attach_to=vehicle)
        actor_list.append(center_sensor)

        #QUEUE FOR PHOTOS
        center_queue = queue.Queue()
        center_sensor.listen(center_queue.put)

        #ADDING AGENT
        agent = BehaviorAgent(vehicle, behavior='cautious')
        agent.ignore_traffic_lights(active = True)
        agent.ignore_stop_signs(active=True)
        agent.ignore_vehicles(active = True)

        # LETTING CAR SPAWN
        for _ in range(20):
            world.tick()

        # CHOOSING DESTINATION
        destination, start_loc = set_target(all_spawn_points, num_positions_to_spawn, num_obstacles, vehicle, world)
        agent.set_destination(destination, start_location=start_loc)

        # SETTING SPECTATOR ON THE TOP LOOKING DOWN
        spectator.set_transform(carla.Transform(carla.Location(x=100, y=204, z=203.0),
                                                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)))

        print("START")

        frame_number = 0
        while True:
            world.tick()
            try:
                center_frame = center_queue.get(True, 2.0)
            except queue.Empty:
                continue

            agent._update_information()

            # DRAWING FOUND ROUTE
            route_queue = agent._local_planner._waypoints_queue
            if len(route_queue) > 0:
                for i, (waypoint, _) in enumerate(route_queue):
                    if i > len(route_queue) - 1: break
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
                # destination, start_loc = set_target(all_spawn_points, num_positions_to_spawn, num_obstacles, vehicle,
                #                                     world)
                # agent.set_destination(destination, start_location=start_loc)

            image_center = raw_data_process(center_frame)

            #TO GET NAVIGATION
            control = agent.run_step()

            current_loc = vehicle.get_location()
            agent.set_destination(destination, start_location=current_loc)

            command_int = agent._local_planner.target_road_option

            steer = process_frame(image_center, command_int)

            control.steer = float(np.clip(steer, -1.0, 1.0))
            control.throttle = 0.15
            control.brake = 0.0

            vehicle.apply_control(control)

            print(command_int.value, control.steer)


    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        for actor in actor_list:
            actor.destroy()
        print("SUCCESSFULLY EXECUTED")

if __name__ == "__main__":
    #Uplodaing model from .ckpt file
    log_path = Path("../../logs")
    agent_path = Path("agent_dave2_const_v_b")
    version = 9

    checkpoint_path = log_path / agent_path / Path(f"version_{str(version)}/checkpoints")

    try:
        model_path = next(checkpoint_path.glob("*ckpt"))
    except StopIteration:
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")

    #Lodaing model to gpu
    model = DrivingModel.load_from_checkpoint(model_path, method = 0)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((66,200)), transforms.ToTensor()])

    world_setup()