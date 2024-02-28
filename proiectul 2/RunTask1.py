from Parameters import *
from FacialDetector import *
import pdb
from Visualize import *
import os

params: Parameters = Parameters()
params.dim_hog_cell = 6
params.number_negative_examples = 90000
params.threshold = 0
params.block_size = 3params.dim_window = 64
params.ratio = 1

params.path_annotations = os.path.join(params.base_dir, 'test/task1_gt_validare.txt')

if not os.path.exists(params.dir_save_files):
    os.makedirs(params.dir_save_files)
    print('directory created: {} '.format(params.dir_save_files))
else:
    print('directory {} exists '.format(params.dir_save_files))



facial_detector: FacialDetector = FacialDetector(params)


params.number_positive_examples = 0
positive_features = []
for name in ["barney", "betty", "fred", "wilma", "unknown"]:
    params.dir_pos_examples = os.path.join(params.dir_examples_directories, name)
    files = [f for f in os.listdir(params.dir_pos_examples) if os.path.isfile(os.path.join(params.dir_pos_examples, f))]
    params.number_positive_examples += len(files)

    positive_features_path = os.path.join(params.dir_save_files, 'descriptoriExemplePozitive_' + os.path.basename(
        params.dir_examples_directories) + '_' + str(params.dim_hog_cell) + '_' + str(
        params.block_size) + '_' + name + '.npy')

    if os.path.exists(positive_features_path):
        current_features = np.load(positive_features_path)
        positive_features.append(current_features)
        print(f'Am incarcat descriptorii pentru exemplele pozitive - {name}')
    else:
        print(f'Construim descriptorii pentru exemplele pozitive - {name}')
        current_features = facial_detector.get_positive_descriptors()
        positive_features.append(current_features)
        np.save(positive_features_path, current_features)
        print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_features_path)


if params.use_flip_images:
    params.number_positive_examples *= 2

negative_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegative_' + str(params.dim_hog_cell) + '_'  + str(params.block_size) + '_' + str(params.number_negative_examples) + '_' + str(params.dim_window) + '.npy')
if os.path.exists(negative_features_path):
    negative_features = np.load(negative_features_path)
    print('Am incarcat descriptorii pentru exemplele negative')
else:
    print('Construim descriptorii pentru exemplele negative:')
    negative_features = facial_detector.get_negative_descriptors()
    np.save(negative_features_path, negative_features)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)


flat_pos_features = []
for character_features in positive_features:
    for feature in character_features:
        flat_pos_features.append(feature)
flat_pos_features = np.array(flat_pos_features)


training_examples = np.concatenate((np.squeeze(flat_pos_features), np.squeeze(negative_features)), axis=0)
train_labels = np.concatenate((np.ones(params.number_positive_examples), np.zeros(negative_features.shape[0])))
facial_detector.train_classifier(training_examples, train_labels)


detections, scores, file_names = facial_detector.run()

np.save(os.path.join(params.dir_save_files, '352_Cioclov_Maria', 'task1', 'detections_all_faces.npy'), detections)
np.save(os.path.join(params.dir_save_files, '352_Cioclov_Maria', 'task1', 'file_names_all_faces.npy'), file_names)
np.save(os.path.join(params.dir_save_files, '352_Cioclov_Maria', 'task1', 'scores_all_faces.npy'), scores)

# if params.has_annotations:
#     facial_detector.eval_detections(detections, scores, file_names)
#     show_detections_with_ground_truth(detections, scores, file_names, params)
# else:
#     show_detections_without_ground_truth(detections, scores, file_names, params)