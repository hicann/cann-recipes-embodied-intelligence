# This file is adapted from AgibotTech/EnerVerse-AC.
# Original project: https://github.com/AgibotTech/EnerVerse-AC
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.
#
# Modifications were made to integrate the utility into this sample.

import matplotlib.cm as cm


ColorMapLeft = cm.Greens
ColorMapRight = cm.Reds
ColorListLeft = [(0, 0, 255), (255, 255, 0), (0, 255, 255)]
ColorListRight = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]



EndEffectorPts = [
    [0, 0, 0, 1],
    [0.1, 0, 0, 1],
    [0, 0.1, 0, 1],
    [0, 0, 0.1, 1]
]

Gripper2EEFCvt = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0.23],
    [0, 0, 0, 1]
]

# End-effector to camera rotation offsets [roll, pitch, yaw], unit: radians (±0.5236 ≈ ±30°)
EEF2CamLeft = [0, 0, -0.5236]
EEF2CamRight = [0, 0, 0.5236]
