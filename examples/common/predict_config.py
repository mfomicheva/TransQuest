from multiprocessing import cpu_count

MODEL_TYPE = 'xlmroberta'
MODEL_NAME = 'xlm-roberta-base'


predict_config = {
    'best_model_dir': 'models/best_model',
    'max_seq_length': 128,
    'eval_batch_size': 8,
    'regression': True,
    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    'silent': False,
    'manual_seed': 777,
    'encoding': None,
}