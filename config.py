_configuration = {
    'BATCH_SIZE': 64,
    'IMAGE_SIZE': [41, 41],
    'EPOCHS': 100,
    'TRAIN_SCALES':[2, 3, 4],
    'VALID_SCALES': [4],
    'DATA_PATH': '',
    'VALID_PATH': '',
    'MODEL_PATH': ''
}

BATCH_SIZE = _configuration['BATCH_SIZE']
IMAGE_SIZE = _configuration['IMAGE_SIZE']
EPOCHS = _configuration['EPOCHS']
TRAIN_SCALES = _configuration['TRAIN_SCALES']
VAL_SCALES = _configuration['VALID_SCALES']

data_path = _configuration['DATA_PATH']
validation_path = _configuration['VALID_PATH']
model_path = _configuration['MODEL_PATH']