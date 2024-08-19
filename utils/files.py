import os
import sys
import csv
import time
import numpy as np


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


def write_results(players, game, games, episode_length):
    
    out = {'game': game
    , 'games': games
    , 'episode_length': episode_length
    , 'p1': players[0].name
    , 'p2': players[1].name
    , 'p1_points': players[0].points
    , 'p2_points': np.sum([x.points for x in players[1:]])
    }

    if not os.path.exists(config.RESULTSPATH):
        with open(config.RESULTSPATH,'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=out.keys())
            writer.writeheader()

    with open(config.RESULTSPATH,'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=out.keys())
        writer.writerow(out)


def load_model(env_name:str, model_name:str, env_obj=None):

    filename = os.path.join(config.MODELDIR, env_name, model_name)
    if os.path.exists(filename):
        logger.info(f'Loading model: {model_name} for environment {env_name}, object {env_obj}')
        cont = True
        while cont:
            try:
                ppo_model = PPO.load(filename, env=env_obj)
                cont = False
            except Exception as e:
                time.sleep(5)
                print(e)

    # no base model found
    elif model_name == 'base.zip':
        logger.info(f"\tNo 'base' model found for environment {env_name}")
        if 'MainAlgo' in env_name:
            # The base model for the algo environment will actually be
            # the previous best model in the base environment
            if 'Marquise' in env_name:
                prev_env_name = "MarquiseMainBase"
            elif 'Eyrie' in env_name:
                prev_env_name = "EyrieMainBase"
            elif 'Alliance' in env_name:
                prev_env_name = "AllianceMainBase"
            elif 'Vagabond' in env_name:
                prev_env_name = "VagabondMainBase"
            else:
                raise Exception(f"Unknown MainAlgo Environment name: '{env_name}'")
            
            logger.info(f"\tAttempting to copy 'best_model' from environment {prev_env_name}...")
            best_file_to_start_from = os.path.join(config.MODELDIR, prev_env_name, "best_model.zip")
            copyfile(best_file_to_start_from, filename)
            logger.info(f"\tNew 'base' model created for environment {env_name}")
            ppo_model = PPO.load(filename, env=env_obj)
        else:
            cont = True
            while cont:
                try:
                    ppo_model = PPO(get_network_arch(env_name), env=env_obj)
                    logger.info(f'Saving new base.zip PPO model...')
                    ppo_model.save(os.path.join(config.MODELDIR, env_name, 'base.zip'))

                    cont = False
                except IOError as e:
                    sys.exit(f'Check zoo/{env_name}/ exists and read/write permission granted to user')
                except Exception as e:
                    logger.error(e)
                    time.sleep(2)
                
    else:
        raise Exception(f'\n{filename} not found')
    
    return ppo_model


def load_all_models(env):
    """
    Load all of the opponent models that could be used for opponents in the given model manager environment.

    Returns a list of 4 items: For each player faction id's index (0-3), the item is 'None' if
    the environment is already training that faction or the rules opponents are being used. 
    Otherwise, the opponent models' slot will have a PPO object for that faction. 
    
    The type of opponents that need to be loaded could be based on the environment passed in.
    """
    models = [{} for _ in range(4)]

    # first ensure that the factions we want to load models of
    # for this environment atleast have a "base" model
    base_filename = os.path.join(config.MODELDIR, env.name, "base.zip")
    if not os.path.exists(base_filename):
        load_model(env.name, "base.zip", env)

    if 'MainBase' in env.name or 'MainAlgo' in env.name:
        return models
    
    modellist = [f for f in os.listdir(os.path.join(config.MODELDIR, env.name)) if f.startswith("_model")]
    modellist.sort()
    models = [load_model(env, 'base.zip')]
    for model_name in modellist:
        models.append(load_model(env, name = model_name))
    return models


def get_best_model_name(env_name):
    modellist = [f for f in os.listdir(os.path.join(config.MODELDIR, env_name)) if f.startswith("_model")]
    
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