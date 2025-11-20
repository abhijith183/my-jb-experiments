## Experimenting with VLA models

This directory now includes a minimal runner for evaluating OpenVLA checkpoints
on LIBERO tasks in the robosuite simulator.

### Running the LIBERO + OpenVLA example (script)

1. Install dependencies (examples):
   ```bash
   pip install robosuite libero openvla torch
   ```
2. Launch a task:
   ```bash
   python VLAs/openvla_libero_runner.py \
       --benchmark libero_spatial \
       --task-index 0 \
       --model-id openvla-openai/7b \
       --camera-name agentview \
       --horizon 200
   ```

The script will load the specified task, render camera observations, query the
OpenVLA policy, and print the cumulative reward at the end of the episode.

### Running in a notebook (with GUI viewer)

1. Open `VLAs/openvla_libero_runner.ipynb` in Jupyter or VS Code.
2. Edit the configuration cell (benchmark, task index, model id, and camera name).
3. Run the notebook cells. The robosuite GUI viewer should pop up while the
   episode executes, and the off-screen frames will feed the OpenVLA policy.

Make sure robosuite is installed with mujoco GUI support so the renderer window
can appear.
