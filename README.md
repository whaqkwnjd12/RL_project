
## Usage

* code directory로 들어가준다.
### 1. conda 가상 환경 생성
```bash
  conda env create -f environment.yml
  conda activate RL
  # torch install
  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  # or (for cpu)
  pip install torch torchvision
```
### 2. Data 구축
```bash
  python data_download.py --data_dir [DATA_DIREC] --history_length [HISTORY_LENGTH] --seed [SEED]
  # e.g.
  # python data_download.py --data_dir ../dataset --history_length 10 --seed 67
```
--data_dir (default: "./dataset")
* signal_data 디렉토리와 clinical_data.csv가 위치해있는 상위 dataset 디렉토리

--history_length (default: 30)
* 각 subject의 time step의 길이에 관여하는 요소
* 결과 data 길이 = history_length + 1

--seed (default: 67) int
* train & valid set을 나눌 떄 사용되는 seed

### 3. Training
```bash
  python main.py --data_dir [DATA_DIREC] --save_dir [SAVE_DIREC] --history_length [HISTORY_LENGTH] --policy_type [POLICY_TYPE]
  # e.g.
  # python main.py --data_dir ../dataset --save_dir ../results --history_length 10 --policy_type transformer
```
--data_dir (default: "./dataset")
* signal_data 디렉토리가 위치해있는 상위 dataset 디렉토리

--save_dir (default: "./results")
* training 결과들이 저장될 디렉토리

--history_length (default: 30)
* 각 subject의 time step의 길이에 관여하는 요소

--policy_type (default: "transformer")
* Policy Model들 ('lstm', 'mog', 'transformer')

### 4. Pretrained-Model Load
```bash
  python model_load.py --model_direc [MODEL DIREC]
```

--data_dir (required)
* model 정보가 위치해있는 디렉토리

## Pretrained Results Link
* [Google Drive](https://drive.google.com/drive/folders/1mrZ9lBG4S8q9K_01zBXXAVjSnT0U5fAO?usp=sharing)