import supervisely as sly
import os
import numpy as np
from dataset_tools.convert import unpack_if_archive
import src.settings as s
from urllib.parse import unquote, urlparse
from supervisely.io.fs import get_file_name, file_exists
import cv2

from tqdm import tqdm

def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:        
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path
    
def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count
    
def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    train_data_path = os.path.join("RuCode Hand Segmentation","train","train")
    test_images_path = os.path.join("RuCode Hand Segmentation","test","test", "images")
    batch_size = 30
    images_folder = "images"
    masks_folder = "segmentation"

    images_paths_train = []
    images_paths_test = [os.path.join(test_images_path,file) for file in os.listdir(test_images_path)]

    for r,d,f in os.walk(train_data_path):
        for dir in d:
            if "images" in dir:
                img_list = os.listdir(os.path.join(r,dir))
                index = os.path.basename(r)
                for file in img_list:
                    old_path = os.path.join(r,dir,file)
                    # new_path = os.path.join(r,dir,f"{index}{file}")
                    # os.rename(old_path,new_path)     
                    images_paths_train.append(old_path)  #renamed  
    project_dict = {"train": images_paths_train, "test": images_paths_test}


    def get_key(dict, value):
        for k, v in dict.items():
            if v == value:
                return k
            
    def create_ann(image_path):
        labels = []
        tags = []
        path_to_file, image_name = os.path.split(image_path)
        path_to_images, _ = os.path.split(path_to_file)
        if "train" in image_path:
            tag_value = image_name[0]
            tag = sly.Tag(tag_meta, value=tag_value)
            tags.append(tag)
        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        mask_path = os.path.join(path_to_images, masks_folder, image_name[1:])
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        if file_exists(mask_path):
            mask_np = sly.imaging.image.read(mask_path)
            if len(np.unique(mask_np)) != 1:
                for color in class_names.values():
                    obj_class = meta.get_obj_class(get_key(class_names, color))
                    mask = np.all(mask_np == color, axis=2)
                    curr_bitmap = sly.Bitmap(mask)
                    curr_label = sly.Label(curr_bitmap, obj_class)
                    labels.append(curr_label)
        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)
    
    class_names = {
        "palm":(255,0,0),
        "thumb_finger":(255,0,255),
        "pointer_finger":(0,255,255),
        "middle_finger":(255,255,0),
        "fourth_finger":(0,0,255),
        "little_finger":(128,255,128)
        }
    obj_classes = [sly.ObjClass(name, sly.Bitmap, class_names[name]) for name in class_names]
    tag_meta = sly.TagMeta("letter", sly.TagValueType.ANY_STRING)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=obj_classes, tag_metas=[tag_meta])
    api.project.update_meta(project.id, meta.to_json())

    dataset = dataset_test = api.dataset.create(project.id, "test", change_name_if_conflict=True)
    dataset_train = api.dataset.create(project.id, "train", change_name_if_conflict=True)

    progress = sly.Progress(
        "Create dataset {}".format(dataset.name),
        len(project_dict["train"]) + len(project_dict["test"]),
    )

    for ds in project_dict:
        if ds == "train":
            dataset = dataset_train
        else:
            dataset = dataset_test
        image_paths = project_dict[ds]
        for img_pathes_batch in sly.batched(image_paths, batch_size=batch_size):
            img_names_batch = [
                os.path.basename(img_path) for img_path in img_pathes_batch
            ]
            img_infos = api.image.upload_paths(
                dataset.id, img_names_batch, img_pathes_batch
            )
            img_ids = [im_info.id for im_info in img_infos]
            anns_batch = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)
            progress.iters_done_report(len(img_names_batch))

    return project
