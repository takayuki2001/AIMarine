import cv2
import numpy as np

from inc.CustomMarineModel import GameSys

canvas = np.full((GameSys.HEIGHT, GameSys.WIDTH, 3), 0, dtype=np.uint8)
'''
Canvas
'''

'''
self.ship = self.canvas.create_polygon(5, 0,  # P1
                                       5, GameRender.COLL_H_HERF * 2,  # P2
                                       GameRender.SHIP_F_R, GameRender.COLL_H_HERF,  # P3
                                       fill='red')
'''
cv2.circle(canvas,
           center=(100, 150),
           radius=60,
           color=(0, 255, 0),
           thickness=3,
           lineType=cv2.LINE_4,
           shift=0)
cv2.imshow('test.png', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
