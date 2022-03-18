import os
import shutil
#from vision.torchvision.datasets.video_utils import VideoClips
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo

class Kinetics(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
        root: str,
        frames_per_clip: int,
        num_classes: str = "400",
        split: str = "train",
        frame_rate: Optional[int] = None,
        step_between_clips: int = 1,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ("avi", "mp4"),
        download: bool = False,
        num_download_workers: int = 1,
        num_workers: int = 1,
        _precomputed_metadata: Optional[Dict[str, Any]] = None,
        _video_width: int = 0,
        _video_height: int = 0,
        _video_min_dimension: int = 0,
        _audio_samples: int = 0,
        _audio_channels: int = 0,
        _legacy: bool = False,
    	):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.num_classes = num_classes
        self.extensions = extensions
        self.num_download_workers = num_download_workers

        self.root = root
        self._legacy = _legacy
        if _legacy:
            print("Using legacy structure")
            self.split_folder = root
            self.split = "unknown"
            if download:
                raise ValueError("Cannot download the videos using legacy_structure.")
        else:
            self.split_folder = path.join(root, split)
            self.split = split


        self.classes, class_to_idx = find_classes(self.split_folder)
        self.samples = make_dataset(self.split_folder, class_to_idx, extensions, is_valid_file=None)
        self.video_list = [x[0] for x in self.samples]
        self.frames_per_clip = frames_per_clip
        """self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
            _audio_channels=_audio_channels,
        )"""
        #self.transform = transform

    def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
	    """Finds the class folders in a dataset.
	    See :class:`DatasetFolder` for details.
	    """
	    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
	    if not classes:
	        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

	    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
	    return classes, class_to_idx
    def make_dataset(directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).
        See :class:`DatasetFolder` for details.
        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        return instances
	    """if empty_classes:
	        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
	        if extensions is not None:
	            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
	        raise FileNotFoundError(msg)"""
        

    def __len__(self) -> int:
        return len(self.video_list)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:

        video_path = self.video_list[idx];
        video = EncodedVideo.from_path(video_path)
        video_tense=video.get_clip(0, int(video.duration))['video']
        video_relax=video_tense.unsqueeze(0)

        numparts = 5
        numframes = self.frames_per_clip
        perpartframes = int(numframes/numparts)
        timesamp = np.random.uniform(0, 1)
        section = int(numparts*timesamp)
        sampled_part = video_relax[:, :, section*perpartframes:(section+1)*perpartframes, :, :]

        #video, audio, info, video_idx = self.video_clips.get_clip(idx)
        if not self._legacy:
            # [T,H,W,C] --> [T,C,H,W]
            video = video.permute(0, 3, 1, 2)
        label = self.samples[idx][1]

        #if self.transform is not None:
        #    video = self.transform(video)

        return sampled_part, label


if __name__ == "__main__":
		"""dir_name = "dataset"
		fake_dir = "fake_dst"
		files = os.listdir(dir_name+'/train/');
		for subdir, dirs, files in os.walk(dir_name+'/train/'):
			for d in dirs:
				if(not os.path.isdir(fake_dir+'/train/'+d)):
					os.mkdir(fake_dir+'/train/'+d)
				for f in os.listdir(dir_name+'/train/'+d):
					shutil.copy(dir_name+'/train/'+d+'/'+f, fake_dir+'/train/'+d+'/'+f)
					break;"""


		# if(not os.path.isdir(dir_name+'/train/')):os.mkdir(dir_name+'/train/')
		#for f in files:
		#		folder_name = f.split('_')[0]
		#		if os.path.isdir(dir_name+'/train/'+folder_name) == False:
		#				os.mkdir(dir_name+'/train/'+folder_name)
		#		shutil.move(dir_name+'/'+f, dir_name+'/train/'+folder_name+'/'+f)

		dataloader = Kinetics(root='fake_dset', frames_per_clip='10', num_classes='400', num_workers=8)