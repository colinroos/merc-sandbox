from util.image_pipeline import tile_image_from_annotation
from util.files import find_files_glob

annotations = find_files_glob('./data-cache/raw/**/', '*.xml')
images = find_files_glob('./data-cache/raw/**/', '*.jpg')

for annote, image in zip(annotations, images):
    tile_image_from_annotation(annote, image)

