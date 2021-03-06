# Learning Obstacle Representations for Neural Motion Planning

Robin Strudel, Ricardo Garcia, Justin Carpentier, Jean-Paul Laumond, Ivan Laptev, Cordelia Schmid\
CoRL 2020


![](images/overview.png)

- [Project Page](https://www.di.ens.fr/willow/research/nmp_repr/)
- [Paper](https://arxiv.org/abs/2008.11174)



### Table of Content

- [Cite](#cite)
- [Setup](#setup)
- [Training](#training)
  - [Train on narrow passages](#narrow-passages)
  - [Train on 3D environments](#3d-environments)
  - [Monitor experiments](#monitor)
- [Run](#run)
- [Logging](#logging)

## Cite

Please cite our work if you use our code or compare to our approach
```
@inproceedings{strudelnmp2020,
title={Learning Obstacle Representations for Neural Motion Planning},
author={R. {Strudel} and R. {Garcia} and J. {Carpentier} and J.P. {Laumond} and I. {Laptev} and C. {Schmid}},
journal={Proceedings of Conference on Robot Learning (CoRL)},
year={2020}
}
```

## Setup

Download the code
```
git clone https://github.com/rstrudel/nmprepr
cd nmprepr
```

To create a new conda environment containing dependencies
```
conda env create -f environment.yml
conda activate nmprepr
```

To update a conda environment with dependencies
```
conda env update -f environment.yml
```

## Train

### Narrow passages

To train a planning policy on 2D environments with narrow passages
```
python -m nmp.train Narrow-64Pts-LocalSurfaceNormals-v0 narrow --horizon 50 --seed 0
```

### 3D environments

To train a planning policy for the Sphere
```
python -m nmp.train Sphere-Boxes-64Pts-Rays-v0 sphere_boxes --horizon 80 --seed 0
```

To train planning policies for the S-Shape
```
python -m nmp.train SShape-Boxes-64Pts-Rays-v0 sshape_boxes --horizon 80 --seed 0
```

### Monitor

You can monitor experiments with
```
tensorboard --logdir=/path/to/experiment
```

## Run

Launch gepetto-gui in a separate terminal
```
gepetto-gui
```

Run a planning policy for the S-Shape and visualize it with gepetto-gui
```
python -m nmp.run SShape-Boxes-64Pts-Rays-v0 --exp-name log_dir/params.pkl --seed 100 --horizon 80
```
       
<img src="images/sphere_boxes.gif" width="400">

Evaluate the success rate of a policy on 100 episodes
```
python -m nmp.run SShape-Boxes-64Pts-Rays-v0 --exp-name log_dir/params.pkl --seed 100 --horizon 80 --episodes 100
```


## Logging

By default the checkpointing will be in your home directory. You can change it by defining a `CHECKPOINT` environment variable. Add the following to your `.bashrc` file to change the logging directory.
```
export CHECKPOINT=/path/to/checkpoints
```

## Documentation
- Class `gym.Env`: basically the whole environment defined by the user, including operating robots and obstacles
- Class `ConfigurationWrapper`: configuration, which corresponds to q in the paper
- Class `ModelWrapper`: model of objects for collision checking and so on
- Colission Checker: `m.collision(q)` - m is ModelWrapper instance and q is ConfigurationWrapper intance

Code execution structure:
1. call run.py in `nmp`, with arguments
2. initialize environment and policy
3. use `multitask_rollout` function to rollout and collection results
  - in `render` mode, will rollout run infinitely many times
  - in `evaluate` mode, will run finite time according to specified number of episodes


System dynamics forward:

- in `env`, the function `step` takes in action

  - function `step` calls `move` and use `model_wrapper` to apply action

        next_state = model_wrapper.integrate(
              state, velocity, self.cartesian_integration
          )
    
    the model is pure kinematics
  
  - need to check what action is actually applied, also the action needs to be formatted before calling `move` function 

Bidirection RRT is implemented in `planning`, `env` has `solve_rrt` to call planning, will return planning results. 

To change sampling startegy, can change `solve.solve` state sampling distribution

To integrate poliy as local planner, needs to modify the extend function, but not sure whether to use `rollout` function from `rlkit`.

### Environment structure

`env = gym.make(env_name)` returns environment `mpenv.observers.robot_links.RobotLinksObserver`

`env.step()` is equal to `env.env.step()`, which is equal to `env.env.env.step()` (refer to `gym/core.py`)

`env.env` type `mpenv.observers.ray_tracing.RayTracingObserver`

`env.env.env` type `mpenv.envs.boxes.Boxes`

### RRT?

Command:

    python -m nmp.test_run --cpu SShape-Boxes-64Pts-Rays-v0 --exp-name log_dir/params.pkl --seed 100 --horizon 80 --episodes 0 > test_run_output

### Visualization

Display robot:

    env.env.env.viz.display(ConfigurationWrapper(env.env.env.model_wrapper,q))

Draw edge:

    # see base.py under envs
    previous_oMg = ConfigurationWrapper.q_oM[2]
    current_oMg = ConfigurationWrapper.q_oM[2]
    previous_ee = env.env.env.robot.get_ee(previous_oMg).translation
    current_ee = env.env.env.robot.get_ee(current_oMg).translation
    env.env.env.viz.add_edge_to_roadmap("path", previous_ee, current_ee) # path is the node name, which can be modified
  
## Evaluation metric

Time, success rate, path quality

## Environment parameters
mpenv/envs/boxes.py line 26-28: 
    
    obstacles_type="boxes" or "shapes" "ycb"

`YCB` models are stored in `mpenv/assets/ycb_data`

mpenv/envs/boxes.py line 49: 

    bound_range = 1.0

mpenv/envs/boxes.py line 98: 

    min_num_obs, max_num_obs = 15, 20
  
Use handcraft narrow environment:

    Function generate_geom_objs, set parameter handcraft=True

Example seed: 10

Choose one set of parameters from 2, in function 

    add_handcraft_obs, set parameters set 1 or parameter set 2
    decide variable translation_matrix and obst_size_matrix 

## Change sampling function

In `solve.py`, 

    def solve(env, delta_growth, iterations, simplify, nmp_input=None, sampler="Full"):

Change parameter sampler:

    if sampler == "Full":
        sample_fn = sample_full_fn
    elif sampler == "Free":
        sample_fn = sample_free_fn
    elif sampler == "Bridge":
        sample_fn = sample_bridge_fn
    elif sampler == "NearSurface":
        sample_fn = sample_near_surface_fn

## Set render as an option

Use `--render` or not in the command:

        python -m nmp.test_run --cpu SShape-Boxes-64Pts-Rays-v0 --exp-name log_dir/params.pkl --seed 100 --horizon 5 --render --solver_type Normal_RRT

## Debug straight line connection case

    LOG_DIR=/home/yaosx/Desktop/EECS598_006_final_project/sshape_boxes_global_1024/seed0
    ENV_NAME=SShape-Boxes-1024Pts-SurfaceNormals-v0
    python -m nmp.test_run --cpu $ENV_NAME --exp-name $LOG_DIR/params.pkl --seed 200 --render --horizon 5 --episodes 0 --solver_type RL_RRT^C
