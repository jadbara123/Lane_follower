import pygame
import math
import os
import cv2
import numpy as np
import hard_code_LK
pygame.init()

WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
RED = (255, 0, 0)

class Carlo:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Top-Down Car Game")

        self.car_image = pygame.image.load('triangle.png').convert()
        self.car_image = pygame.transform.scale(self.car_image, (30, 20))
        self.car_image.set_colorkey(WHITE)
        self.car_rect = self.car_image.get_rect()
        self.car_speed = 0  
        self.car_x, self.car_y = WIDTH / 2, HEIGHT / 2
        self.car_angle = 0
        self.acceleration = 0.6
        self.current_action = 0
        self.max_speed = 1.2
        self.rect_x = 0
        self.rect_y = 0
        self.rect_width = 800
        self.rect_height = 600

        # Check if "track.png" exists in the current directory
        self.track_background = None
        if os.path.exists("track1.png"):
            self.track_background = pygame.image.load("track1.png")
            self.track_background = pygame.transform.scale(self.track_background, (WIDTH, HEIGHT))

        self.running = True
        self.clock = pygame.time.Clock()

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype = "float32")

        s = pts.sum(axis = 1)
        rect[0] = pts[0]
        rect[2] = pts[2]

        diff = np.diff(pts, axis = 1)
        rect[1] = pts[1]
        rect[3] = pts[3]
        # return the ordered coordinates
        return rect

    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self.order_points(pts)
        (tl, tr, br, bl) = (pts[0], pts[1], pts[2], pts[3])
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        # return the warped image
        return warped
    
    def get_state(self):
        warped = self.warped
        warped_r = self.control_sag()
        warped_l = self.control_sol()
        # X type setup
        state_vector = [int(np.nanmean(warped_r[0:8, 0:8]/150)), int(np.nanmean(warped_r[0:8, 8:16]/150)),
                        int(np.nanmean(warped_r[11:19, 4:12]/150)), int(np.nanmean(warped_r[22:30, 0:8]/150)),
                        int(np.nanmean(warped_r[22:30, 8:16]/150)),

                        int(np.nanmean(warped_l[0:8, 0:8]/150)), int(np.nanmean(warped_l[0:8, 8:16]/150)),
                        int(np.nanmean(warped_l[11:19, 4:12]/150)), int(np.nanmean(warped_l[22:30, 0:8]/150)),
                        int(np.nanmean(warped_l[22:30, 8:16]/150)), 
                        
                        int(np.nanmean(warped[0:6, 0:6]/150)),   int(np.nanmean(warped[0:6, 6:12]/150)),
                        int(np.nanmean(warped[0:6, 12:18]/150)), int(np.nanmean(warped[11:17, 12:16]/150)),
                        int(np.nanmean(warped[11:17, 16:20]/150)), int(np.nanmean(warped[11:17, 20:24]/150)),
                        int(np.nanmean(warped[22:28, 24:28]/150)), int(np.nanmean(warped[22:28, 28:32]/150)),
                        int(np.nanmean(warped[22:28, 32:36]/150)), int(np.nanmean(warped[33:39, 30:36]/150)),
                        int(np.nanmean(warped[33:39, 36:42]/150)), int(np.nanmean(warped[33:39, 42:48]/150)),
                        int(np.nanmean(warped[33:39, 0:6]/150)), int(np.nanmean(warped[33:39, 6:12]/150)),
                        int(np.nanmean(warped[33:39, 12:18]/150)), int(np.nanmean(warped[22:28, 12:16]/150)),
                        int(np.nanmean(warped[22:28, 16:20]/150)), int(np.nanmean(warped[22:28, 20:24]/150)),
                        int(np.nanmean(warped[11:17, 24:28]/150)), int(np.nanmean(warped[11:17, 28:32]/150)),
                        int(np.nanmean(warped[11:17, 32:36]/150)), int(np.nanmean(warped[0:6, 30:36]/150)),
                        int(np.nanmean(warped[0:6, 36:42]/150)), int(np.nanmean(warped[0:6, 42:48]/150)),
                        ]
        for i in range(len(state_vector)):
            if state_vector[i] > 0.9: state_vector[i] = 1
        return state_vector
    
    def get_action(self, train = True):
        if train:
            return self.current_action
        else:
            self.lane = hard_code_LK.lane_keep()
            action = self.lane.main(self.control_sag(), self.control_sol())
            print(action)
            return action

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cv2.destroyAllWindows()
                exit()

        keys = pygame.key.get_pressed()
        
        self.current_action = 0
        if keys[pygame.K_UP]:
            self.car_speed += self.acceleration
            if self.car_speed > self.max_speed:
                self.car_speed = self.max_speed
            self.current_action = 5
        else:
            self.car_speed -= self.acceleration
            if self.car_speed < 0:
                self.car_speed = 0

        if keys[pygame.K_DOWN]:
            self.car_speed -= self.acceleration
            if self.car_speed < -self.max_speed:
                self.car_speed = -self.max_speed
        if keys[pygame.K_LEFT]:
            self.car_angle -= 1+self.car_speed
            self.current_action = 1
        if keys[pygame.K_RIGHT]:
            self.car_angle += 1+self.car_speed
            self.current_action = 2
        if keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
            self.car_speed += self.acceleration
            if self.car_speed > self.max_speed:
                self.car_speed = self.max_speed
            self.car_angle += 1+self.car_speed
            self.current_action = 3
        if keys[pygame.K_UP] and keys[pygame.K_LEFT]:
            self.car_speed += self.acceleration
            if self.car_speed > self.max_speed:
                self.car_speed = self.max_speed
            self.car_angle -= 1+self.car_speed
            self.current_action = 4

        self.car_x += self.car_speed * math.cos(math.radians(self.car_angle))
        self.car_y += self.car_speed * math.sin(math.radians(self.car_angle))

        self.car_x = max(self.rect_x, min(self.car_x, self.rect_x + self.rect_width))
        self.car_y = max(self.rect_y, min(self.car_y, self.rect_y + self.rect_height))

    def update(self):
        self.handle_events()
    
    def act(self, action):
        #print("DEBUG1")
        """if str(action) == "0":
            self.car_speed += 0.02
            if self.car_speed > self.max_speed:
                self.car_speed = self.max_speed-0.03"""

        if str(action) == str(5):
            self.car_speed += self.acceleration
            if self.car_speed > self.max_speed:
                self.car_speed = self.max_speed
        
        if str(action) == "0":
            self.car_speed -= 0.008
            if self.car_speed < 0.4:
                self.car_speed = 0.4

        if str(action) == str(1):
            self.car_angle -= 1+self.car_speed

        if str(action) == str(2):
            self.car_angle += 1+self.car_speed
            
        if str(action) == str(3):
            self.car_speed += self.acceleration
            if self.car_speed > self.max_speed:
                self.car_speed = self.max_speed
            self.car_angle += 1+self.car_speed

        if str(action) == str(4):
            self.car_speed += self.acceleration
            if self.car_speed > self.max_speed:
                self.car_speed = self.max_speed
            self.car_angle -= 1+self.car_speed

        self.car_x += self.car_speed * math.cos(math.radians(self.car_angle))
        self.car_y += self.car_speed * math.sin(math.radians(self.car_angle))

        self.car_x = max(self.rect_x, min(self.car_x, self.rect_x + self.rect_width))
        self.car_y = max(self.rect_y, min(self.car_y, self.rect_y + self.rect_height))

        print(action)

    def render(self):
        self.screen.fill(WHITE)
        if self.track_background:
            self.screen.blit(self.track_background, (0, 0))

        pygame.draw.rect(self.screen, RED, (self.rect_x, self.rect_y, self.rect_width, self.rect_height), 2)

        rotated_car = pygame.transform.rotate(self.car_image, -self.car_angle)
        self.car_rect = rotated_car.get_rect(center=(self.car_x, self.car_y))
        self.screen.blit(rotated_car, self.car_rect.topleft)

        dot_distance = 15  
        dot_positions_relative = [
            (dot_distance+40, -24),  
            (dot_distance+40, 24),
            (dot_distance, 24),
            (dot_distance, -24)
        ]

        rotated_dot_positions = []
        for dot_x, dot_y in dot_positions_relative:
            dot_x_rotated = self.car_x + dot_x * math.cos(math.radians(self.car_angle)) - dot_y * math.sin(math.radians(self.car_angle))
            dot_y_rotated = self.car_y + dot_x * math.sin(math.radians(self.car_angle)) + dot_y * math.cos(math.radians(self.car_angle))
            rotated_dot_positions.append((int(dot_y_rotated), int(dot_x_rotated)))

        pygame_surface = pygame.display.get_surface()
        pygame_image = pygame.surfarray.array3d(pygame_surface)
        self.opencv_image = cv2.cvtColor(pygame_image, cv2.COLOR_RGB2BGR)
        cv2.rotate(self.opencv_image, 90)

        roi_coordinates = []
        for dot_position in rotated_dot_positions:
            cv2.circle(self.opencv_image, dot_position, 3, (255, 255, 255), -1)
            roi_coordinates.append(dot_position)

        self.warped = self.four_point_transform(self.opencv_image, np.array(roi_coordinates))
        self.warped = cv2.cvtColor(self.warped, cv2.COLOR_BGR2GRAY)
        ret, self.warped = cv2.threshold(self.warped, 127, 255, cv2.THRESH_BINARY)

        cv2.circle(self.opencv_image, (50,50), 5, (0,0,255), -1)
        cropped = self.warped[0:8, 0:8]

        cv2.imshow("warprd", self.warped)
        cv2.imshow("cropped", cropped)
        cv2.imshow("original", self.opencv_image)

        pygame.display.flip()
        self.clock.tick(60)

    def run(self):
        self.update()
        self.render()
        return self.warped
    
    def control_sol(self):
        dot_distance = 15  
        dot_positions_relative = [
            (dot_distance+15, -8),
            (dot_distance+15, -24),
            (dot_distance-15, -24),
            (dot_distance-15, -8)
        ]

        rotated_dot_positions = []
        for dot_x, dot_y in dot_positions_relative:
            dot_x_rotated = self.car_x + dot_x * math.cos(math.radians(self.car_angle)) - dot_y * math.sin(math.radians(self.car_angle))
            dot_y_rotated = self.car_y + dot_x * math.sin(math.radians(self.car_angle)) + dot_y * math.cos(math.radians(self.car_angle))
            rotated_dot_positions.append((int(dot_y_rotated), int(dot_x_rotated)))

        pygame_surface = pygame.display.get_surface()
        pygame_image = pygame.surfarray.array3d(pygame_surface)
        self.opencv_image = cv2.cvtColor(pygame_image, cv2.COLOR_RGB2BGR)
        cv2.rotate(self.opencv_image, 90)

        roi_coordinates = []
        for dot_position in rotated_dot_positions:
            cv2.circle(self.opencv_image, dot_position, 3, (255, 255, 255), -1)
            roi_coordinates.append(dot_position)

        warped_l = self.four_point_transform(self.opencv_image, np.array(roi_coordinates))
        warped_l = cv2.cvtColor(warped_l, cv2.COLOR_BGR2GRAY)
        ret, warped_l = cv2.threshold(warped_l, 127, 255, cv2.THRESH_BINARY)

        return warped_l
    
    def control_sag(self):
        dot_distance = 15  
        dot_positions_relative = [
            (dot_distance+15, +8),  
            (dot_distance+15, +24),
            (dot_distance-15, +24),
            (dot_distance-15, +8)
        ]

        rotated_dot_positions = []
        for dot_x, dot_y in dot_positions_relative:
            dot_x_rotated = self.car_x + dot_x * math.cos(math.radians(self.car_angle)) - dot_y * math.sin(math.radians(self.car_angle))
            dot_y_rotated = self.car_y + dot_x * math.sin(math.radians(self.car_angle)) + dot_y * math.cos(math.radians(self.car_angle))
            rotated_dot_positions.append((int(dot_y_rotated), int(dot_x_rotated)))

        pygame_surface = pygame.display.get_surface()
        pygame_image = pygame.surfarray.array3d(pygame_surface)
        self.opencv_image = cv2.cvtColor(pygame_image, cv2.COLOR_RGB2BGR)
        cv2.rotate(self.opencv_image, 90)

        roi_coordinates = []
        for dot_position in rotated_dot_positions:
            cv2.circle(self.opencv_image, dot_position, 3, (255, 255, 255), -1)
            roi_coordinates.append(dot_position)
        
        warped_r = self.four_point_transform(self.opencv_image, np.array(roi_coordinates))
        warped_r = cv2.cvtColor(warped_r, cv2.COLOR_BGR2GRAY)
        ret, warped_r = cv2.threshold(warped_r, 127, 255, cv2.THRESH_BINARY)
        

        return warped_r
    
    def yeni_alan(self):
        dot_distance = 15  
        dot_positions_relative = [
            (dot_distance+40, -24),  
            (dot_distance+40, 24),
            (dot_distance, 24),
            (dot_distance, -24)
        ]

        rotated_dot_positions = []
        for dot_x, dot_y in dot_positions_relative:
            dot_x_rotated = self.car_x + dot_x * math.cos(math.radians(self.car_angle)) - dot_y * math.sin(math.radians(self.car_angle))
            dot_y_rotated = self.car_y + dot_x * math.sin(math.radians(self.car_angle)) + dot_y * math.cos(math.radians(self.car_angle))
            rotated_dot_positions.append((int(dot_y_rotated), int(dot_x_rotated)))

        pygame_surface = pygame.display.get_surface()
        pygame_image = pygame.surfarray.array3d(pygame_surface)
        self.opencv_image = cv2.cvtColor(pygame_image, cv2.COLOR_RGB2BGR)
        cv2.rotate(self.opencv_image, 90)

        roi_coordinates = []
        for dot_position in rotated_dot_positions:
            cv2.circle(self.opencv_image, dot_position, 3, (255, 255, 255), -1)
            roi_coordinates.append(dot_position)
        #orijinal resimle uğraşma alanı
        

        self.warped = self.four_point_transform(self.opencv_image, np.array(roi_coordinates))
        return self.warped
    
    def is_terminal(self):
        warped_sol = self.control_sol()
        warped_sag = self.control_sag()
        sol_serit = np.mean(warped_sol) #bellow 60 over 170
        sag_serit = np.mean(warped_sag) #bellow 60 over 170
        sol_serit_flag = 170 < sol_serit or sol_serit < 50
        sag_serit_flag = 170 < sag_serit or sag_serit < 30

        if sol_serit_flag: return True, 1
        elif sag_serit_flag: return True, 1
        else: return False, 0
    


"""if __name__ == "__main__":
    carlo = Carlo()
    while 1:
        carlo.run()
        carlo.rl_view()
    pygame.quit()
    cv2.destroyAllWindows()
"""