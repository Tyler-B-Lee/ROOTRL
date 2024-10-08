import os
import sys
import time
import numpy as np
import torch

from shutil import rmtree, copyfile
from stable_baselines3 import PPO

from register import get_network_arch
import config

import logging
# logging.basicConfig(
#     filename='file.log',
#     format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
#     filemode='w',
#     # level=logging.DEBUG
# )
logger = logging.getLogger(__name__)


def load_model(model_type:str, model_name:str, zoo_dir:str, env_obj=None):
    """
    Load a PPO model from the given model directory, with the given model name.
    If the model does not exist, create a new model with the given name and save it.

    Returns the loaded PPO model object.

    :param model_type: The type of model environment to load the model for.
    :param model_name: The name of the model file to load.
    :param env_obj: The environment object to load the model for.

    :return: The loaded PPO model object.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"> Loading on device: {device}")

    filename = os.path.join(zoo_dir, model_type, model_name)
    if os.path.exists(filename):
        print(f'Loading model: {model_name} for environment {model_type}, object {env_obj}')
        cont = True
        while cont:
            try:
                ppo_model = PPO.load(filename, env=env_obj, device=device)
                cont = False
            except Exception as e:
                time.sleep(5)
                print(e)

    # no base model found
    elif model_name == 'base.pt':
        logger.info(f"\tNo 'base' model found for environment {model_type}")
        if any(s in model_type for s in ['Hates','Helps','Main_Arena']):
            # We take the previous best model in the main_algo environment
            # as the base model for the new environment
            if 'Main_Arena' in model_type:
                if 'Marquise' in model_type:
                    prev_env_name = "MarquiseMain_Algo"
                elif 'Eyrie' in model_type:
                    prev_env_name = "EyrieMain_Algo"
                elif 'Alliance' in model_type:
                    prev_env_name = "AllianceMain_Algo"
                elif 'Vagabond' in model_type:
                    prev_env_name = "VagabondMain_Algo"
                else:
                    raise Exception(f"Unknown Environment name: '{model_type}'")
            
            # The base model for the given environment will actually be
            # the previous best model in the base environment
            else:
                if 'Marquise' in model_type:
                    prev_env_name = "MarquiseMain_Base"
                elif 'Eyrie' in model_type:
                    prev_env_name = "EyrieMain_Base"
                elif 'Alliance' in model_type:
                    prev_env_name = "AllianceMain_Base"
                elif 'Vagabond' in model_type:
                    prev_env_name = "VagabondMain_Base"
                else:
                    raise Exception(f"Unknown Environment name: '{model_type}'")
            
            logger.info(f"\tAttempting to copy 'best_model' from environment {prev_env_name}...")
            best_file_to_start_from = os.path.join(zoo_dir, prev_env_name, "best_model.pt")
            copyfile(best_file_to_start_from, filename)
            logger.info(f"\tNew 'base' model created for environment {model_type}")
            ppo_model = PPO.load(filename, env=env_obj, device=device)
        
        else:
            cont = True
            while cont:
                try:
                    ppo_model = PPO(get_network_arch(model_type), env=env_obj, device=device)
                    logger.info(f'Saving new base.pt PPO model...')
                    ppo_model.save(os.path.join(zoo_dir, model_type, 'base.pt'))

                    cont = False
                except IOError as e:
                    sys.exit(f'Check zoo/{model_type}/ exists and read/write permission granted to user')
                except Exception as e:
                    logger.error(e)
                    time.sleep(2)
                
    else:
        raise Exception(f'\n{filename} not found')
    
    return ppo_model

def get_best_model_name(model_type, zoo_dir):
    modellist = [f for f in os.listdir(os.path.join(zoo_dir, model_type)) if f.startswith("_model")]
    
    if len(modellist)==0:
        filename = None
    else:
        modellist.sort()
        filename = modellist[-1]
        
    return filename

def get_model_stats(filename):
    if filename is None:
        generation = 0
        timesteps = 0
        best_rules_based = -np.inf
        best_reward = -np.inf
    else:
        stats = filename.split('_')
        generation = int(stats[2])
        best_rules_based = float(stats[3])
        best_reward = float(stats[4])
        timesteps = int(stats[5])
    return generation, timesteps, best_rules_based, best_reward


def reset_logs(model_dir):
    try:
        filelist = [ f for f in os.listdir(config.LOGDIR) if f not in ['.gitignore']]
        for f in filelist:
            if os.path.isfile(f):  
                os.remove(os.path.join(config.LOGDIR, f))

        for i in range(100):
            if os.path.exists(os.path.join(config.LOGDIR, f'tb_{i}')):
                rmtree(os.path.join(config.LOGDIR, f'tb_{i}'))
        
        open(os.path.join(config.LOGDIR, 'log.txt'), 'a').close()
    
        
    except Exception as e :
        print(e)
        print('Reset logs failed')

def reset_models(model_dir):
    try:
        filelist = [ f for f in os.listdir(model_dir) if f not in ['.gitignore']]
        for f in filelist:
            os.remove(os.path.join(model_dir , f))
    except Exception as e :
        print(e)
        print('Reset models failed')