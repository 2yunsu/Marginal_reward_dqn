import numpy as np
import cv2


message_draw_config = {
    "tile_width": 30,
    "outline": 2,
}


def draw_msg(msg_len, n_agents, message):
    global message_draw_config
    canvas_width = message_draw_config["outline"] + msg_len * \
        (message_draw_config["outline"] + message_draw_config["tile_width"])
    canvas_height = message_draw_config["outline"] + n_agents * \
        (message_draw_config["outline"] + message_draw_config["tile_width"])

    canvas = np.zeros((canvas_width, canvas_height, 3), np.uint8)
    canvas[:, :, :] = 255

    for y in range(n_agents + 1):
        top_y = y * (message_draw_config["outline"] + message_draw_config["tile_width"])
        bottom_y = top_y + message_draw_config["outline"]
        canvas[:, top_y : bottom_y, :] = 0

    for x in range(msg_len + 1):
        left_x = x * (message_draw_config["outline"] + message_draw_config["tile_width"])
        right_x = left_x + message_draw_config["outline"]
        canvas[left_x : right_x, :, :] = 0

    for y in range(n_agents):
        top_y = message_draw_config["outline"] + y * \
            (message_draw_config["outline"] + message_draw_config["tile_width"])
        bottom_y = top_y + message_draw_config["tile_width"] - 1

        for x in range(msg_len):
            left_x = x * (message_draw_config["outline"] + message_draw_config["tile_width"])
            right_x = left_x + message_draw_config["outline"]
            canvas[left_x : right_x, :, :] = 0

            left_x = message_draw_config["outline"] + x * \
                (message_draw_config["outline"] + message_draw_config["tile_width"])
            right_x = left_x + message_draw_config["tile_width"] - 1

            colour = int(message[y, x] * 255)
            cv2.rectangle(canvas, (top_y, left_x), (bottom_y, right_x), (colour, colour, colour), -1)

    canvas = cv2.transpose(canvas)

    return canvas