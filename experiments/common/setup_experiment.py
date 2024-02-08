import os, sys, json
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from stable_baselines3.common.logger import configure
import hydra


# def setup_experiment(exp_name, options: dataclass):
#     """Parse params, setup a directory, and map std outputs to files"""
#     parser = ArgumentParser()
#     parser.add_argument('--log_dir', default=f"logs/{exp_name}", type=str, help="Output Folder path")
#     parser.add_argument('--add_date_suffix', default=False, type=bool)
#     parser.add_arguments(options, 'params')
#     args = parser.parse_args()
#     args = parser.parse_args()

#     if args.add_date_suffix:
#         timestr = datetime.now().strftime("%Y%m%d.%H%M%S.%f")
#         args.log_dir = f"{args.log_dir}_{timestr}"
    
#     log_dir = args.log_dir
#     os.makedirs(log_dir, exist_ok=True)
#     # save cmd args
#     with open(f'{log_dir}/commandline_args.txt', 'w') as f:
#         class EnhancedJSONEncoder(json.JSONEncoder):
#             def default(self, o):
#                 if dataclasses.is_dataclass(o):
#                     return dataclasses.asdict(o)
#                 return super().default(o)
#         data = {**args.__dict__, "experiment_name": exp_name}
#         json.dump(data, f, indent=2, cls=EnhancedJSONEncoder)

#     sys.stdout = open(log_dir+"/log.txt", "w")
#     sys.stderr = open(log_dir+"/error_log.txt", "w")
#     return args

def get_log_dir():
    return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

def get_value_logger():
    logger_folder = get_log_dir()
    return configure(logger_folder, ["stdout", "csv", "tensorboard", "log"])

    
def flush_logs():
    sys.stdout.flush()