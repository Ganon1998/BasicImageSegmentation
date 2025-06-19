from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
from segmentation import k_means_segment, train_model, default_convergence, cluster, train_model_improved
from sklearn.metrics import silhouette_score


def find_optimal_k(image, max_k=5):
    scores = []
    for k in range(2, max_k+1):
        segmented = k_means_segment(image, k=k)
        score = silhouette_score(image.reshape(-1, 3), segmented.reshape(-1, 3))
        scores.append(score)
    return np.argmax(scores) + 2


def main(filepath):
    image = Image.open(filepath)
    flattened_image = np.array(image)
    h, w, c = flattened_image.shape

    k = find_optimal_k(flattened_image)
    segmented_image = k_means_segment(flattened_image, k=k)

    # show image as a result from k means clustering
    plt.imshow(segmented_image.astype('uint8'))
    plt.show()

    MU, SIGMA, PI, responsibility = train_model(flattened_image, k, default_convergence)
    clustered_pixels = cluster(responsibility)
    segmented_image = MU[clustered_pixels].reshape((h, w, c))

    # show image as a result from GMM
    plt.imshow(segmented_image.astype('uint8'))
    plt.show()

    MU, SIGMA, PI, responsibility = train_model_improved(flattened_image, k, default_convergence)
    clustered_pixels = cluster(responsibility)
    segmented_image = MU[clustered_pixels].reshape((h, w, c))

    # show image as a result of GMM after 900 rounds of training
    plt.imshow(segmented_image.astype('uint8'))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a file.")
    parser.add_argument("file_path", type=str, help="The path to the file to process")

    args = parser.parse_args()
    main(args.file_path)