from Parameters import *
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog




class FacialDetector:
    def __init__(self, params):
        self.params = params
        self.best_model = None

    def get_positive_descriptors(self):
        files = []
        images_path = os.path.join(self.params.dir_pos_examples, '*jpg')
        files += glob.glob(images_path)

        num_images = len(files)
        positive_descriptors = []
        print('Calculam descriptorii pt %d imagini pozitive...' % num_images)
        for i in range(num_images):
            print('Procesam exemplul pozitiv numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(self.params.block_size, self.params.block_size), feature_vector=True)

            # new_width = int(hog_image.shape[1] * 3)
            # new_height = int(hog_image.shape[0] * 3)
            # scaled_image = cv.resize(hog_image, (new_width, new_height))
            # cv.imshow('image', np.uint8(scaled_image))
            # cv.waitKey(0)

            positive_descriptors.append(features)

            if self.params.use_flip_images:
                features = hog(np.fliplr(img), pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                               cells_per_block=(self.params.block_size, self.params.block_size), feature_vector=True)

                positive_descriptors.append(features)

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors

    def get_negative_descriptors(self):
        files = []
        images_path = os.path.join(self.params.dir_neg_examples, '*jpg')
        files += glob.glob(images_path)

        num_images = len(files)
        num_negative_per_image = self.params.number_negative_examples // num_images
        negative_descriptors = []
        print('Calculam descriptorii pt %d imagini negative' % num_images)
        for i in range(num_images):
            print('Procesam exemplul negativ numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            num_rows = img.shape[0]
            num_cols = img.shape[1]
            x = np.random.randint(low=0, high=num_cols - self.params.dim_window, size=num_negative_per_image)
            y = np.random.randint(low=0, high=num_rows - self.params.dim_window, size=num_negative_per_image)

            for idx in range(len(y)):
                patch = img[y[idx]: y[idx] + self.params.dim_window, x[idx]: x[idx] + self.params.dim_window]
                descr = hog(patch, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                            cells_per_block=(self.params.block_size, self.params.block_size), feature_vector=False)


                negative_descriptors.append(descr.flatten())

        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors

    def train_classifier(self, training_examples, train_labels):
        svm_file_name = os.path.join(self.params.dir_save_files, f'best_model_%d_%d_%d_%d - {self.params.name}' %
                                     (self.params.dim_hog_cell, self.params.block_size,
                                      self.params.number_negative_examples, self.params.dim_window))
        if os.path.exists(svm_file_name):
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            print(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        scores = best_model.decision_function(training_examples)
        self.best_model = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]

        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title(f'Distributia scorurilor clasificatorului pe exemplele de antrenare - {self.params.name}')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

    def run(self):
        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        detections = None
        scores = np.array([])
        file_names = np.array([])
        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0]
        num_test_images = len(test_files)
        descriptors_to_return = []
        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i, num_test_images))
            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)

            for scale in [1.2, 1, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
                new_width = int(img.shape[1] * scale)
                new_height = int(img.shape[0] * scale)
                scaled_image = cv.resize(img, (new_width, new_height))

                image_scores = []
                image_detections = []

                # for y in range(0, scaled_image.shape[0] - self.params.dim_window+1, self.params.dim_hog_cell):
                #     for x in range(0, scaled_image.shape[1] - self.params.dim_window+1, self.params.dim_hog_cell):

                h_window= int(self.params.ratio * self.params.dim_window)                                                           #adaugat acum
                w_window= self.params.dim_window                                                               #adaugat acum
                for y in range(0, scaled_image.shape[0] - h_window + 1, self.params.dim_hog_cell):                  #adaugat acum
                    for x in range(0, scaled_image.shape[1] - w_window + 1, self.params.dim_hog_cell):          #adaugat acum

                        window = scaled_image[y:y+h_window, x:x+w_window]
                        window = cv.resize(window, (self.params.dim_window, self.params.dim_window))       # adaugat acum

                        descr =  hog(window, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                            cells_per_block=(self.params.block_size, self.params.block_size), feature_vector=True)

                        score = np.dot(descr, w)[0] + bias
                        if score > self.params.threshold:
                            # x_min = int(x * self.params.dim_hog_cell)
                            # y_min = int(y * self.params.dim_hog_cell)
                            # x_max = int(x * self.params.dim_hog_cell + self.params.dim_window)
                            # y_max = int(y * self.params.dim_hog_cell + self.params.dim_window)
                            # image_detections.append([x_min, y_min, x_max, y_max])
                            image_detections.append([x, y, x+self.params.dim_window, y+self.params.dim_window])
                            image_scores.append(score)
                if len(image_scores) > 0:
                    image_detections, image_scores = self.non_maximal_suppression(np.array(image_detections),
                                                                                  np.array(image_scores), scaled_image.shape)
                if len(image_scores) > 0:
                    scaled_back_detections = []
                    for detection in image_detections:
                        x_min, y_min, x_max, y_max = detection
                        scaled_x_min = int(x_min // scale)
                        scaled_y_min = int(y_min // scale)
                        scaled_x_max = int(x_max // scale)
                        scaled_y_max = int(y_max // scale)
                        scaled_back_detections.append([scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max])

                    if detections is None:
                        detections = scaled_back_detections
                    else:
                        detections = np.concatenate((detections, scaled_back_detections))
                    scores = np.append(scores, image_scores)
                    short_name = ntpath.basename(test_files[i])
                    image_names = [short_name for ww in range(len(image_scores))]
                    file_names = np.append(file_names, image_names)

                end_time = timeit.default_timer()
                print('Timpul de procesarea al imaginii de testare %d/%d - scale %f este %f sec.'
                      % (i, num_test_images, scale, end_time - start_time))

        return detections, scores, file_names

    def compute_average_precision(self, rec, prec):
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], int)

        num_gt_detections = len(ground_truth_detections)
        gt_exists_detection = np.zeros(num_gt_detections)
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]


            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Average precision %.3f - {self.params.name}' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, f'precizie_medie_{self.params.name}.png'))
        plt.show()
