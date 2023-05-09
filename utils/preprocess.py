import h5py
from pathlib import Path
from tqdm import tqdm
import numpy as np


def preprocess(pid: str, dataset_dir: Path, output_dir: Path) -> None:
    with h5py.File(dataset_dir / f'{pid}.mat', 'r') as f_input:
        images = f_input.get('Data/data')[()]
        labels = f_input.get('Data/label')[()]
        gazes = labels[:, :2]
    images = images.transpose(0, 2, 3, 1).astype(np.uint8)

    person_output_dir = output_dir / pid
    person_output_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(person_output_dir / 'images.npy'), images)
    np.save(str(person_output_dir / 'gazes.npy'), gazes)


def main():
    output_dir = Path("./../data")
    output_dir.mkdir(exist_ok=True, parents=True)
    dataset_dir = Path("./../MPIIFaceGaze_normalized/MPIIFaceGaze_normalizad")

    for pid in tqdm(range(15)):
        pid_str = f'p{pid:02}'
        preprocess(pid_str, dataset_dir, output_dir)


if __name__ == '__main__':
    main()
