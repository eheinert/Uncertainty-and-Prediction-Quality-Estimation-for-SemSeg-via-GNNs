import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image
from skimage.segmentation import find_boundaries
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path

def metaseg_rgb_color(value):
    """original MetaSeg RGBA colors"""
    r = 1 - 0.5 * value
    g = value
    b = 0.3 + 0.35 * value
    return r, g, b, 1

class Style:
    metaseg_colors = np.array([metaseg_rgb_color(value) for value in np.linspace(0, 1, 128)])
    color_map = ListedColormap(metaseg_colors, name='MetaSeg')

def plot_segments(segments, segment_values, save_path=None, image_path = None, inf_blend = 0):
    if segment_values.dtype == "bool":
        segment_values = segment_values.astype(np.float32)
    img_array = Style.color_map(segment_values[segments - 1])
    img_array = (img_array[..., :3] * 255).astype(np.uint8)
    img_array *= ~find_boundaries(segments + 1, connectivity=2)[..., None]
    """if inf_type == photo blend with original Image"""       
    img = Image.fromarray(img_array)
    if inf_blend > 0:
        photo = Image.open(image_path).convert('RGB')
        img  = Image.blend(photo, img, inf_blend)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)
    else:
        return img

def visualize_and_save_meta_prediction(metaseg_data, meta_prediction, save_dir, inf_blend = 0, image_paths = None, worker=None):
    print("Visualize and save meta prediction masks here: {}".format(save_dir))
    if worker is not None and worker.parallel:
        num_cpus = worker.num_cpus if "num_cpus" in worker else 1
        chunksize = worker.chunksize if "chunksize" in worker else 1
        pool_args = [(metaseg_data[i].segments,
                      meta_prediction[i],
                      save_dir / f"{metaseg_data[i].basename}_meta_prediction.png",
                      image_paths[i],
                      inf_blend
                      ) for i in range(len(metaseg_data))]
        with Pool(num_cpus) as pool:
            y_hat = pool.starmap(plot_segments, tqdm(pool_args, total=len(pool_args)), chunksize=chunksize)
    return y_hat