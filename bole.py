from recbole.quick_start import run_recbole

# run_recbole(model='KGNNLS', dataset='Amazon_Books', config_file_list=["config/KGNNLS.yaml"])

run_recbole(model='KGIN', dataset='ml-20m', config_file_list=["config/KGIN.yaml"])

# ssh -p 28415 root@124.71.96.66