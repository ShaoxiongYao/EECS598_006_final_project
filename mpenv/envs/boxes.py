import os
import numpy as np
import sys
from gym import spaces
import pinocchio as pin
import hppfcl

from mpenv.envs.base import Base
from mpenv.envs import utils as envs_utils
from mpenv.envs.utils import ROBOTS_PROPS
from mpenv.core import utils
from mpenv.core.geometry import Geometries
from mpenv.core.mesh import Mesh
from mpenv.core.model import ConfigurationWrapper

from mpenv.observers.robot_links import RobotLinksObserver
from mpenv.observers.point_cloud import PointCloudObserver
from mpenv.observers.ray_tracing import RayTracingObserver

import open3d as o3d

np.set_printoptions(threshold=sys.maxsize)

class Boxes(Base):
    def __init__(
        self,
        robot_name,
        has_boxes,
        cube_bounds=True,
        obstacles_type="boxes",
        # obstacles_type="shapes",
        # obstacles_type="ycb",
        dynamic_obstacles=False,
    ):
        super().__init__(robot_name)

        self.has_boxes = has_boxes
        self.geoms = None
        self.cube_bounds = cube_bounds
        self.obstacles_type = obstacles_type
        self.dynamic_obstacles = dynamic_obstacles

        self.robot_props = ROBOTS_PROPS[self.robot_name]
        self._set_obstacles_props()
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.robot_props["action_dim"],), dtype=np.float32
        )

        self.normalizer_local = {"mean": 0, "std": 0.4}
        self.normalizer_global = {"mean": 0, "std": 0.5}

    def _set_obstacles_props(self):
        bound_range = 0.6
        if self.robot_name == "s_shape":
            self.freeflyer_bounds = np.array(
                [
                    [-bound_range, -bound_range, -bound_range, -np.inf, -np.inf, -np.inf, -np.inf],
                    [bound_range, bound_range, bound_range, np.inf, np.inf, np.inf, np.inf],
                ]
            )
        else:
            self.freeflyer_bounds = np.array([[-bound_range, -bound_range, -bound_range], 
                                              [bound_range, bound_range, bound_range]])

        # constrain box xyz position to not be on boundary to get a denser obstacle set
        self.center_bounds = self.freeflyer_bounds[:, :3].copy()
        self.center_bounds[0, :3] += np.array([0.15, 0.15, 0.15])
        self.center_bounds[1, :3] -= np.array([0.15, 0.15, 0.15])
        self.center_bounds
        self.size_bounds = [0.15, 0.5]

    def _reset(self, start=None, goal=None):
        self.geoms = self.get_obstacles_geoms()
        self.robot = self.add_robot(self.robot_name, self.freeflyer_bounds)

        for geom_obj in self.geoms.geom_objs:
            self.add_obstacle(geom_obj, static=True)
        self.model_wrapper.create_data()

        if start is not None:
            if not isinstance(start, ConfigurationWrapper):
                start = ConfigurationWrapper(self.model_wrapper, start)
            self.state = start
        else:
            self.state = self.random_configuration()
        if goal is not None:
            if not isinstance(goal, ConfigurationWrapper):
                goal = ConfigurationWrapper(self.model_wrapper, goal)
            self.goal_state = goal
        else:
            self.goal_state = self.random_configuration()
        
        # print("start q:", self.state.q)
        # print("goal q:", self.goal_state.q)

        return self.observation()

    def get_obstacles_geoms(self):
        if not self.has_boxes:
            return Geometries()
        
        min_num_obs, max_num_obs = 3, 10
        # boxes
        if self.n_obstacles is None:
            n_obstacles = self._np_random.randint(min_num_obs, max_num_obs)
        else:
            n_obstacles = self.n_obstacles

        geom_objs, placement_tuple = generate_geom_objs(
            self._np_random,
            self.freeflyer_bounds,
            self.center_bounds,
            self.size_bounds,
            self.cube_bounds,
            n_obstacles,
            self.obstacles_type,
            self.dynamic_obstacles,
            self.obstacles_color,
            self.obstacles_alpha,
            o3d_viz=self.o3d_viz
        )
        self.se3_obst_tuple = placement_tuple
        geoms = Geometries(geom_objs)
        return geoms

    def compute_surface_pcd(self, n_pts):
        return obstacles_to_surface_pcd(self.geoms, n_pts, self.freeflyer_bounds)

    def compute_volume_pcd(self, n_pts):
        return self.geoms.compute_volume_pcd(n_pts)

    def set_eval(self):
        pass


def sample_box_parameters(np_random, center_bounds, size_bounds):
    box_size = np_random.uniform(size_bounds[0], size_bounds[1], size=3)
    pos_box = np_random.uniform(center_bounds[0], center_bounds[1], 3)
    rand_se3 = pin.SE3.Identity()
    pin.SE3.setRandom(rand_se3)
    rand_se3.translation = pos_box
    return rand_se3, box_size


def sample_geom(np_random, obst_type, size, index=None):
    geom = None
    path = ""
    scale = np.ones(3)
    if obst_type != "boxes":
        size /= 2

    if obst_type == "boxes":
        geom = hppfcl.Box(*size)
    elif obst_type == "shapes":
        j = np.random.randint(3)
        if j == 0:
            r = 1.3 * size[0]
            geom = hppfcl.Sphere(r)
        elif j == 1:
            geom = hppfcl.Cylinder(size[0] / 1.5, size[1] * 3)
        elif j == 2:
            # geom = hppfcl.Cone(size[0] * 1.5, 3 * size[1])
            geom = hppfcl.Capsule(size[0], 2 * size[1])
    elif obst_type == "ycb":
        # dataset_path = "YCB_PATH"
        current_dir = os.path.dirname(__file__)
        dataset_path = "../assets/ycb_data"
        files = os.listdir(os.path.realpath(os.path.join(current_dir, dataset_path)))
        idx = np.random.randint(len(files))
        path = os.path.join(dataset_path, files[idx])
        scale = 3.5 * np.ones(3)
    return geom, path, scale


def generate_geom_objs(
    np_random,
    freeflyer_bounds,
    center_bounds,
    size_bounds,
    cube_bounds,
    n_obstacles,
    obstacles_type,
    dynamic_obstacles,
    obstacles_color,
    obstacles_alpha,
    handcraft=False,
    o3d_viz=None
):
    colors = np_random.uniform(0, 1, (n_obstacles, 4))
    colors[:, 3] = obstacles_alpha
    if obstacles_color is not None:
        colors = [obstacles_color for _ in range(n_obstacles)]
    name = "box{}"
    geom_objs = []
    placement_tuple = []
    # obstacles
    for i in range(n_obstacles):
        if not handcraft:
            rand_se3_init, obst_size = sample_box_parameters(
                np_random, center_bounds, size_bounds
            )
        else:
            rand_se3_init, obst_size = add_handcraft_obs(i)
        
        if not handcraft:
            rand_se3_target, obst_size = sample_box_parameters(
                np_random, center_bounds, size_bounds
            )
        else:
            rand_se3_target, obst_size = add_handcraft_obs(i)

        placement_tuple.append((rand_se3_init, rand_se3_target))
        geom, path, scale = sample_geom(np_random, obstacles_type, obst_size, i)
        mesh = Mesh(
            name=name.format(i),
            geometry=geom,
            placement=rand_se3_init,
            color=colors[i],
            geometry_path=path,
            scale=scale,
        )
        geom_obj_obstacle = mesh.geom_obj()
        geom_objs.append(geom_obj_obstacle)
        # uncomment below for visualization
        # if o3d_viz is not None:
        #     geom_mesh = o3d.geometry.TriangleMesh.create_box(width=0.95*obst_size[0], height=0.95*obst_size[1], depth=0.95*obst_size[2])
        #     lb_to_center = np.eye(4)
        #     lb_to_center[0, 3]-=0.5*obst_size[0]
        #     lb_to_center[1, 3]-=0.5*obst_size[1]
        #     lb_to_center[2, 3]-=0.5*obst_size[2]
        #     geom_mesh.transform(lb_to_center)
        #     geom_mesh.transform(rand_se3_init)
        #     if o3d_viz.viz is not None:
        #         o3d_viz.viz.add_geometry(geom_mesh)
        #     else:
        #         o3d_viz._create_viz()
        #         o3d_viz.viz.add_geometry(geom_mesh)
    if cube_bounds:
        geom_objs_bounds = envs_utils.get_bounds_geom_objs(freeflyer_bounds[:, :3])
        geom_objs += geom_objs_bounds
    return geom_objs, placement_tuple

def add_handcraft_obs(idx):

    # parameter set 1
    # translation_matrix = np.array([[0.3, -0.3, 0], [-0.3, -0.3, 0], [0, 0.3, 0], 
    #                                [0.35, 0.3, 0], [-0.35, 0.3, 0]])
    # obst_size_matrix = np.array([[0.35, 0.5, 0.1], [0.35, 0.5, 0.1], [0.05, 0.5, 0.1], 
    #                              [0.2, 0.5, 0.1], [0.2, 0.5, 0.1]])

    # parameter set 2
    # translation_matrix = np.array([[0.35, 0, 0], [-0.35, 0, 0]])
    # obst_size_matrix = np.array([[0.45, 1.2, 0.1], [0.45, 1.2, 0.1]])

    #  parameter set 3
    translation_matrix = np.array([[0, -0.3, 0], [0.35, 0.3, 0], [-0.35, 0.3, 0]])
    obst_size_matrix = np.array([[0.8, 0.5, 0.1], [0.6, 0.5, 0.1], [0.6, 0.5, 0.1]])

    se3 = pin.SE3.Identity()
    se3.rotation = np.eye(3)
    if idx < translation_matrix.shape[0]:
        se3.translation = translation_matrix[idx, :]
        obst_size = obst_size_matrix[idx, :]
    else:
        se3.translation = translation_matrix[0, :]
        obst_size = obst_size_matrix[0, :]

    return se3, obst_size

def obstacles_to_surface_pcd(geoms, n_pts, bounds):
    n_valid=0
    valid_points=np.zeros([n_pts, 3])
    valid_normals=np.zeros([n_pts, 3]) 
    while n_valid < n_pts:
        points, normals = geoms.compute_surface_pcd(n_pts)
        for i in range(n_pts):
            if ((normals[i] == np.array([1, 0, 0])).all() and points[i, 0] > bounds[1, 0]) or \
                ((normals[i] == np.array([-1, 0, 0])).all() and points[i, 0] < bounds[0, 0]) or \
                ((normals[i] == np.array([0, 1, 0])).all() and points[i, 1] > bounds[1, 1]) or \
                ((normals[i] == np.array([0, -1, 0])).all() and points[i, 1] < bounds[0, 1]) or \
                ((normals[i] == np.array([0, 0, 1])).all() and points[i, 2] > bounds[1, 2]) or \
                ((normals[i] == np.array([0, 0, -1])).all() and points[i, 2] < bounds[0, 2]):
                continue
            else:
                valid_points[n_valid] = points[i]
                valid_normals[n_valid] = normals[i]
                n_valid+=1
                if n_valid >= n_pts:
                    break
    return valid_points, valid_normals


def boxes_noobst(robot_name):
    env = Boxes(robot_name, has_boxes=False, cube_bounds=True)
    # coordinate_frame = "global"
    coordinate_frame = "local"
    env = RobotLinksObserver(env, coordinate_frame)
    return env


def boxes_pointcloud(robot_name, n_samples, on_surface, add_normals):
    env = Boxes(robot_name, has_boxes=True, cube_bounds=True, dynamic_obstacles=False)
    coordinate_frame = "local"
    env = PointCloudObserver(env, n_samples, coordinate_frame, on_surface, add_normals)
    env = RobotLinksObserver(env, coordinate_frame)
    return env


def boxes_raytracing(robot_name, n_samples, n_rays):
    env = Boxes(robot_name, has_boxes=True, cube_bounds=True, dynamic_obstacles=False)
    visibility_radius = 0.7
    memory_distance = 0.06
    env = RayTracingObserver(env, n_samples, n_rays, visibility_radius, memory_distance)
    coordinate_frame = "local"
    env = RobotLinksObserver(env, coordinate_frame)
    return env
