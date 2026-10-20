from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.config import *

# custom configs when adding new envs. can scale to any type and any difficulty.


def medium_terrains_cfg() -> TerrainGeneratorCfg:
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=12,  # difficulty resolution
    num_cols=20,  # ignored when curriculum=True
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    add_lights=True,
    sub_terrains={
      "pyramid_stairs": pyramid_stairs(
        proportion=0.25, step_height_range=(0.0, 0.18), step_width=0.32
      ),
      "pyramid_stairs_inv": pyramid_stairs_inv(
        proportion=0.25, step_height_range=(0.0, 0.18), step_width=0.32
      ),
      "random_spread_boxes": random_spread_boxes(
        proportion=0.10,
        num_boxes=80,
        box_width_range=(0.1, 0.84),
        box_length_range=(0.05, 1.0),
        box_yaw_range=(0.0, 360.0),
        add_floor=True,
        platform_width=1,
        border_width=0.25,
      ),
    },
  )


# general traversal set for teachers
def teacher_terrains_cfg() -> TerrainGeneratorCfg:
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=15,  # difficulty resolution
    num_cols=20,  # ignored when curriculum=True
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    add_lights=True,
    sub_terrains={
      "flat": flat(proportion=0.05),
      "pyramid_stairs": pyramid_stairs(
        proportion=0.15, step_height_range=(0.0, 0.20), step_width=0.32
      ),
      "pyramid_stairs_inv": pyramid_stairs_inv(
        proportion=0.15, step_height_range=(0.0, 0.20), step_width=0.32
      ),
      "hf_pyramid_slope": hf_pyramid_slope(proportion=0.10, slope_range=(0.0, 0.5)),
      "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
        proportion=0.10, slope_range=(0.0, 0.5)
      ),
      # flat at difficulty 0, up to 8 cm bumps at 1
      "random_rough": random_rough(
        proportion=0.20, noise_range=(0.02, 0.08), scale_with_difficulty=True
      ),
      "wave_terrain": wave_terrain(proportion=0.05, amplitude_range=(0.0, 0.15)),
      "random_spread_boxes": random_spread_boxes(
        proportion=0.15,
        num_boxes=80,
        box_width_range=(0.1, 0.84),
        box_length_range=(0.05, 1.0),
        box_height_range=(0.05, 0.2),
        box_yaw_range=(0.0, 360.0),
        add_floor=True,
        platform_width=1,
        border_width=0.25,
      ),
      "tilted_grid": tilted_grid(
        proportion=0.05,
        grid_width=1,
        tilt_range_deg=12,
        height_range=0.15,
        floor_depth=2,
      ),
    },
  )


def hf_terrains_cfg() -> TerrainGeneratorCfg:
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=15,  # difficulty resolution
    num_cols=20,  # ignored when curriculum=True
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    add_lights=True,
    sub_terrains={
      "perlin_noise": perlin_noise(),
      "discrete_obstacles": discrete_obstacles(),
      "random_rough": random_rough(),
    },
  )
