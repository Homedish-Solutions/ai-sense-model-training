import os
import zipfile
import matplotlib.pyplot as plt

 

def unzip_dataset(zip_path: str, extract_to: str = None):
    extract_to = extract_to or os.path.dirname(zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to    

def get_labels(image_path:str) -> list[str]:
    labels = []
    for i in os.listdir(image_path):
        if os.path.isdir(os.path.join(image_path, i)):
            labels.append(i)
    return labels    

def show_examples(image_path: str, labels: list[str], num_examples: int = 5):
    for label in labels:
        label_dir = os.path.join(image_path, label)
        example_filenames = os.listdir(label_dir)
        num_images = min(num_examples, len(example_filenames))
        fig, axs = plt.subplots(1, num_images, figsize=(10, 2))
        for i in range(num_images):
            axs[i].imshow(plt.imread(os.path.join(label_dir, example_filenames[i])))
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
        fig.suptitle(f'Showing {num_images} example(s) for "{label}"')
    plt.show()
