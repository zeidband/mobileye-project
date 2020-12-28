import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from phase_IV.frame import Frame
from phase_I.run_attention import find_tfl_lights
from phase_III.frame_container import FrameContainer
import phase_III.SFM as SFM

start_frame = 23


class TFL_manager:
    def __init__(self, pkl_data):
        self.data = pkl_data
        self.prev_frame = None

    def get_EM(self, curr_iteration: int, prev_iteration: int):
        EM = np.eye(4)

        for i in range(prev_iteration + start_frame, curr_iteration + start_frame - 1):
            EM = np.dot(self.data['egomotion_' + str(i) + '-' + str(i + 1)], EM)

        return EM

    def find_lights(self, curr_frame: Frame) -> [list, list]:
        red_x, red_y, green_x, green_y = find_tfl_lights(np.array(Image.open(curr_frame.path)))
        auxiliary = []
        candidates = []

        for x, y in zip(green_x, green_y):
            candidates.append((x, y))
            auxiliary.append('g')

        for x, y in zip(red_x, red_y):
            candidates.append((x, y))
            auxiliary.append('r')

        return candidates, auxiliary

    def evaluate_candidates(self, model, img, candidate_arr, aux, crop_size=81) -> (list, list):
        # Evaluate a list of traffic light candidates using the trained model
        # Create crops of each candidate and stack them as a batch
        candidate_batch = np.zeros((len(candidate_arr), crop_size, crop_size, 3), dtype=np.uint8)

        for i in range(len(candidate_arr)):
            candidate_batch[i, :, :, :] = img.crop((candidate_arr[i][0] - 41, candidate_arr[i][1] - 41,
                                                    candidate_arr[i][0] + 40, candidate_arr[i][1] + 40))
        # Run inference on batch
        results = model.predict(candidate_batch)[:, 1] > .85
        # Return candidates that the net judged as traffic lights
        approved_mask = results.astype(np.bool)

        return list(np.array(candidate_arr)[approved_mask]), list(np.array(aux)[approved_mask])

    def get_tfl(self, curr_frame: Frame, candidates: list, auxiliary: list) -> [list, list]:
        loaded_model = load_model("model.h5")
        image = Image.open(curr_frame.path)
        tfl_candidates, tfl_auxiliary = self.evaluate_candidates(loaded_model, image, candidates, auxiliary)

        return tfl_candidates, tfl_auxiliary

    def find_distance_of_tfl(self, curr_frame: Frame) -> list:
        prev_container = FrameContainer(self.prev_frame.path)
        curr_container = FrameContainer(curr_frame.path)
        focal = self.data['flx']
        pp = self.data['principle_point']
        prev_container.traffic_light = np.array(self.prev_frame.tfl_candidates)
        curr_container.traffic_light = np.array(curr_frame.tfl_candidates)
        curr_container.EM = self.get_EM(curr_frame.iteration + 23, self.prev_frame.iteration + 23)

        return SFM.calc_tfl_dist(prev_container, curr_container, focal, pp)

    # def get_pp(self):
    #     return self.data['principle_point']
    #
    # def get_focal(self):
    #     return self.data['flx']

    def on_frame(self, curr_frame: Frame) -> list:
        # part 1
        lights_candidates, lights_auxiliary = self.find_lights(curr_frame)
        assert len(lights_candidates) == len(lights_auxiliary)

        # part 2
        curr_frame.tfl_candidates, curr_frame.tfl_auxiliary = self.get_tfl(curr_frame, lights_candidates,
                                                                           lights_auxiliary)
        assert len(curr_frame.tfl_candidates) == len(curr_frame.tfl_auxiliary)
        assert len(curr_frame.tfl_candidates) <= len(lights_candidates)

        # part 3
        tfl_distance = None

        if self.prev_frame is not None and len(curr_frame.tfl_candidates) and len(self.prev_frame.tfl_candidates):
            tfl_distance = self.find_distance_of_tfl(curr_frame)
            assert len(tfl_distance.traffic_lights_3d_location) == len(curr_frame.tfl_candidates)

        self.visualize(curr_frame, lights_candidates, lights_auxiliary, tfl_distance)
        self.prev_frame = curr_frame

        return curr_frame.tfl_candidates

    def visualize(self, curr_frame: Frame, lights_candidates, lights_auxiliary, tfl_distance) -> None:
        fig, (candidate, traffic_light, dis) = plt.subplots(1, 3, figsize=(12, 6))
        candidate.set_title('candidates')
        candidate.imshow(Image.open(curr_frame.path))
        for i in range(len(lights_candidates)):
            candidate.plot(lights_candidates[i][0], lights_candidates[i][1], lights_auxiliary[i] + "+")

        traffic_light.set_title('traffic_lights')
        traffic_light.imshow(Image.open(curr_frame.path))
        for i in range(len(curr_frame.tfl_candidates)):
            traffic_light.plot(curr_frame.tfl_candidates[i][0], curr_frame.tfl_candidates[i][1],
                               curr_frame.tfl_auxiliary[i] + "*")

        dis.set_title('distance')
        dis.imshow(Image.open(curr_frame.path))
        if tfl_distance is not None:
            for i in range(len(tfl_distance.traffic_lights_3d_location)):
                dis.text(curr_frame.tfl_candidates[i][0], curr_frame.tfl_candidates[i][1],
                         r'{0:.1f}'.format(tfl_distance.traffic_lights_3d_location[i, 2]), color='y')

        plt.show()
