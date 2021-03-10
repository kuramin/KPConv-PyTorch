import numpy as np
from os.path import join
from utils.ply import read_ply, write_ply
import colorsys


def hsv2rgb(h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))


if __name__ == '__main__':
    radius = 1.5
    kp_extent = 1.2

    # print(hsv2rgb(0, 1, 1))
    # print(hsv2rgb(0.2, 1, 1))
    # print(hsv2rgb(0.4, 1, 1))
    # print(hsv2rgb(0.6, 1, 1))
    # print(hsv2rgb(0.8, 1, 1))
    # print(hsv2rgb(1, 1, 1))
    # print(hsv2rgb(1, 1, 1))
    # print(hsv2rgb(1, 1, 1))

    kernel_dir = 'kernels/dispositions'
    kernel_file = join(kernel_dir, 'k_{:03d}_{:s}_{:d}D.ply'.format(15, 'center', 3))
    data = read_ply(kernel_file)
    kernel_points = np.vstack((data['x'], data['y'], data['z'])).T
    kernel_points = radius * kernel_points
    #print(kernel_points)
    ball_center = kernel_points[0]

    diap_end = 2 * radius
    diap_step = 0.01  # 0.01
    cube_points = []
    for x in np.arange(0, diap_end + diap_step, diap_step):
        for y in np.arange(0, diap_end + diap_step, diap_step):
            for z in np.arange(0, diap_end + diap_step, diap_step):
                x_p = x - diap_end / 2
                y_p = y - diap_end / 2
                z_p = z - diap_end / 2
                cube_points.append([x_p, y_p, z_p])

    # with open('dense_cube.ply', "w") as file:
    #     stri = 'ply\nformat ascii 1.0\nelement vertex ' + str(int(((diap_end-diap_start) / diap_step + 1)**3)) + '\nproperty float32 x\nproperty float32 y\nproperty float32 z\nend_header\n'
    #     file.write(stri)
    #     for x in np.arange(diap_start, diap_end + diap_step, diap_step):
    #         for y in np.arange(diap_start, diap_end + diap_step, diap_step):
    #             for z in np.arange(diap_start, diap_end + diap_step, diap_step):
    #                 x_p = x - diap_end / 2
    #                 y_p = y - diap_end / 2
    #                 z_p = z - diap_end / 2
    #                 file.write('{:2.6f} {:2.6f} {:2.6f}\n'.format(x_p, y_p, z_p))
    #                 cube_points.append([x_p, y_p, z_p])

    cube_points = np.array(cube_points)
    write_ply('dense_cube.ply',
              [cube_points],
              ['x', 'y', 'z'])

    # print(cube_points.shape)
    # with open('dense_ball.ply', "w") as file:
    #     stri = 'ply\nformat ascii 1.0\nelement vertex ' + str(cube_points.shape[0]) + '\nproperty float32 x\nproperty float32 y\nproperty float32 z\nend_header\n'
    #     file.write(stri)
    #     for point in cube_points:
    #         if (point[0]-ball_center[0])**2 + (point[1]-ball_center[1])**2 + (point[2]-ball_center[2])**2 < radius**2:
    #             file.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')

    ball_points = []
    ball_colors = []
    for point in cube_points:
        if (point[0]-ball_center[0])**2 + (point[1]-ball_center[1])**2 + (point[2]-ball_center[2])**2 < radius**2:
            ball_points.append([point[0], point[1], point[2]])
            grey_component = ((point[0]-ball_center[0])**2 + (point[1]-ball_center[1])**2 + (point[2]-ball_center[2])**2) / radius**2
            ball_colors.append([255 * grey_component, 255 * grey_component, 255 * grey_component])

    ball_points = np.array(ball_points)
    ball_colors = np.array(ball_colors, dtype=np.uint8)
    write_ply('dense_ball.ply',
              [ball_points, ball_colors],
              ['x', 'y', 'z'] + ['red', 'green', 'blue'])

    # with open('dense_ball.ply', "w") as file:
    #     stri = 'ply\nformat ascii 1.0\nelement vertex ' + str(ball_points.shape[0]) + '\nproperty float32 x\nproperty float32 y\nproperty float32 z\nend_header\n'
    #     file.write(stri)
    #     for point in ball_points:
    #         file.write('{:2.6f} {:2.6f} {:2.6f}\n'.format(point[0], point[1], point[2]))

    rigid_balls_points = []
    rigid_ball_colors = []
    for point in ball_points:
        for ind_kp, kernel_point in enumerate(kernel_points):
            hue = ind_kp / len(kernel_points)
            if (point[0]-kernel_point[0])**2 + (point[1]-kernel_point[1])**2 + (point[2]-kernel_point[2])**2 < kp_extent**2:
                rigid_balls_points.append([point[0], point[1], point[2]])
                saturation = ((point[0]-kernel_point[0])**2 + (point[1]-kernel_point[1])**2 + (point[2]-kernel_point[2])**2) / kp_extent**2
                rgb_from_hsv = hsv2rgb(hue, saturation, 1)
                # print([rgb_from_hsv[0], rgb_from_hsv[1], rgb_from_hsv[2]])
                rigid_ball_colors.append([rgb_from_hsv[0], rgb_from_hsv[1], rgb_from_hsv[2]])
                # grey_component = ((point[0]-kernel_point[0])**2 + (point[1]-kernel_point[1])**2 + (point[2]-kernel_point[2])**2) / kp_extent**2
                # rigid_ball_colors.append([255 * grey_component, 255 * grey_component, 255 * grey_component])

    rigid_balls_points = np.array(rigid_balls_points)
    rigid_ball_colors = np.array(rigid_ball_colors, dtype=np.uint8)

    # with open('rigid_balls.ply', "w") as file:
    #     stri = 'ply\nformat ascii 1.0\nelement vertex ' + str(rigid_balls_points.shape[0]) + '\nproperty float32 x\nproperty float32 y\nproperty float32 z\nend_header\n'
    #     file.write(stri)
    #     for point in rigid_balls_points:
    #         file.write('{:2.6f} {:2.6f} {:2.6f}\n'.format(point[0], point[1], point[2]))

    write_ply('rigid_balls.ply',
              [rigid_balls_points, rigid_ball_colors],
              ['x', 'y', 'z'] + ['red', 'green', 'blue'])
