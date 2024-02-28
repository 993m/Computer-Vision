from Parameters import *
from FacialDetector import *
import pdb
from Visualize import *
import os

params: Parameters = Parameters()
params.dim_hog_cell = 6
params.threshold = 0
params.dim_window = 64

def run_for_character(name):
    params.name = name

    params.dir_pos_examples = os.path.join(params.dir_examples_directories, name)

    params.path_annotations = os.path.join(params.base_dir, f'test/task2_{name}_gt_validare.txt')

    if not os.path.exists(params.dir_save_files):
        os.makedirs(params.dir_save_files)
        print('directory created: {} '.format(params.dir_save_files))
    else:
        print('directory {} exists '.format(params.dir_save_files))


    facial_detector: FacialDetector = FacialDetector(params)

    positive_features_path = os.path.join(params.dir_save_files, 'descriptoriExemplePozitive_' + os.path.basename(params.dir_examples_directories) + '_' + str(params.dim_hog_cell) + '_' + str(params.block_size) + f'_{params.name}' + '.npy')

    if os.path.exists(positive_features_path):
        positive_features = np.load(positive_features_path)
        print(f'Am incarcat descriptorii pentru exemplele pozitive - {name}')
    else:
        print(f'Construim descriptorii pentru exemplele pozitive - {name}:')
        positive_features = facial_detector.get_positive_descriptors()
        np.save(positive_features_path, positive_features)
        print(f'Am salvat descriptorii pentru exemplele pozitive in fisierul %s - {name}' % positive_features_path)


    negative_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegative_' + str(params.dim_hog_cell) + '_'  + str(params.block_size) + '_' + str(params.number_negative_examples) + '_' + str(params.dim_window) + '.npy')
    if os.path.exists(negative_features_path):
        negative_features = np.load(negative_features_path)
        print('Am incarcat descriptorii pentru exemplele negative')
    else:
        print('Construim descriptorii pentru exemplele negative:')
        negative_features = facial_detector.get_negative_descriptors()
        np.save(negative_features_path, negative_features)
        print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)

    # ADAUG CELELALTE PERSONAJE LA EXEMPLE NEGATIVE
    # for character in ["barney", "betty", "fred", "wilma", "unknown"]:
    #     if character != name:
    #         params.dir_pos_examples = os.path.join(params.dir_examples_directories, character)
    #         files = [f for f in os.listdir(params.dir_pos_examples) if
    #                  os.path.isfile(os.path.join(params.dir_pos_examples, f))]
    #         params.number_negative_examples += len(files)
    #
    #         negative_features_path = os.path.join(params.dir_save_files, 'descriptoriExemplePozitive_' + os.path.basename(
    #             params.dir_examples_directories) + '_' + str(params.dim_hog_cell) + '_' + str(
    #             params.block_size) + '_' + character + '.npy')
    #
    #         if os.path.exists(negative_features_path):
    #             negative_features = np.concatenate((negative_features, np.load(negative_features_path)))
    #             print(f'Am incarcat descriptorii pentru exemplele negative - {character}')
    #         else:
    #             print(f'Construim descriptorii pentru exemplele negative - {character}')
    #             current_features = facial_detector.get_positive_descriptors()
    #             negative_features = np.concatenate((negative_features, current_features))
    #             np.save(negative_features_path, current_features)
    #             print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)

    training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
    train_labels = np.concatenate((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0])))
    facial_detector.train_classifier(training_examples, train_labels)

    detections, scores, file_names = facial_detector.run()

    np.save(os.path.join(params.dir_save_files, '352_Cioclov_Maria', 'task2', f'detections_{name}.npy'), detections)
    np.save(os.path.join(params.dir_save_files, '352_Cioclov_Maria', 'task2', f'file_names_{name}.npy'), file_names)
    np.save(os.path.join(params.dir_save_files, '352_Cioclov_Maria', 'task2', f'scores_{name}.npy'), scores)

    # if params.has_annotations:
    #     facial_detector.eval_detections(detections, scores, file_names)
    #     show_detections_with_ground_truth(detections, scores, file_names, params)
    # else:
    #     show_detections_without_ground_truth(detections, scores, file_names, params)



# FRED #
params.number_negative_examples = 20000
params.block_size = 3
params.ratio = 1.05
run_for_character("fred")

# WILMA #
params.number_negative_examples = 50000
params.block_size = 4
params.ratio = 1.13
run_for_character("wilma")

# BARNEY #
params.number_negative_examples = 50000
params.block_size = 3
params.ratio = 0.91
run_for_character("barney")

# BETTY #
params.number_negative_examples = 50000
params.block_size = 4
params.ratio = 1.0469
run_for_character("betty")
