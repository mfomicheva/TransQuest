from multiprocessing import cpu_count

MODEL_TYPE = 'xlmroberta'
MODEL_NAME = 'xlm-roberta-base'


predict_config = {
    'max_seq_length': 128,
    'eval_batch_size': 8,
    'regression': True,
    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    'silent': False,
    'manual_seed': 777,
    'encoding': None,
    'do_lower_case': False,
    'wandb_project': None,
    'wandb_kwargs': {},
    'no_cache': False,
    'overwrite_output_dir': True,
    'reprocess_input_data': True,
    'use_cached_eval_features': False,
}