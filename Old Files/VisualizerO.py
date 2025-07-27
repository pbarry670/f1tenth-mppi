import numpy as np
import cv2

class Visualizer:
    def __init__(self, origin=(512, 512)):
        self.width = 1024
        self.height = 1024 #Keep square image to avoid things breaking
        self.img = np.zeros((self.width, self.height, 3), np.uint8)
        self.img.fill(255)
        self.origin = origin
        self.w = 5  # m

        self.resolution = self.img.shape[0] / self.w

    def get_img(self):
        return self.img

    def draw_frame(self):
        cv2.arrowedLine(
            self.img, self.origin, (self.origin[0], self.origin[1] - 40), (0, 0, 0), 1
        )
        cv2.arrowedLine(
            self.img, self.origin, (self.origin[0] - 40, self.origin[1]), (0, 0, 0), 1
        )

    def clear_img(self):
        self.img.fill(255)

        # Draw origin and coordinate frame
        cv2.circle(self.img, self.origin, radius=4, color=(0, 0, 255), thickness=1)
        self.draw_frame()

    def draw_circle(self, x, y, radius, red, green, blue, thickness):
        cv2.circle(self.img, (x,y), radius, (red, green, blue), thickness)
        #img, center, radius, BGR, thickness

    def draw_car(self):
        car = np.array([[-0.15, -0.1], [-0.15, 0.1], [0.15, 0.1], [0.15, -0.1]])
        car = car*self.resolution
        car = car + self.width/2
        car = car.astype(np.int32)
        
        cv2.fillPoly(self.img, [car], (255, 0, 0))
                     
    def draw_polylines(self, pts):
        pts = np.array(pts)
        pts[:, 0] = pts[:, 0] + 0.15
        pts = pts*self.resolution
        pts = pts + self.width/2
        pts = pts.astype(np.int32)
        print(pts)
        cv2.polylines(self.img, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

    def draw_pointcloud(self, pts, radius, red, green, blue, thickness):
        num_pts = int(len(pts)//2)
        pts = self.convert_resolution(pts)
        pts_x = np.array(pts[:num_pts])
        pts_y = np.array(pts[num_pts:])

        pts_x = (pts_x + self.width/2).astype(np.int32)
        pts_y = (pts_y + self.height/2).astype(np.int32)

        for i in range(num_pts):
            self.draw_circle(pts_x[i], pts_y[i], radius, red, green, blue, thickness)

    def convert_resolution(self, x):
        num_entries = len(x)
        for i in range(num_entries):
            x[i] = x[i]*int(self.resolution)
        return x 
    
