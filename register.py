

def get_environment(env_name):
    try:
        if "MainBase" in env_name:
            from envs.base_training import MainBaseEnv
            return MainBaseEnv
        else:
            raise Exception(f'No environment found for {env_name}')
    except SyntaxError as e:
        print(e)
        raise Exception(f'Syntax Error for {env_name}!')
    except:
        raise Exception(f'Install the environment first using: \nbash scripts/install_env.sh {env_name}\nAlso ensure the environment is added to /utils/register.py')
    

def get_network_arch(env_name):
    if env_name in ('MarquiseMainBase'):
        from models.marquise_model import CustomPolicy
        return CustomPolicy
    if env_name in ('EyrieMainBase'):
        from models.eyrie_model import CustomPolicy
        return CustomPolicy
    if env_name in ('AllianceMainBase'):
        from models.alliance_model import CustomPolicy
        return CustomPolicy
    if env_name in ('VagabondMainBase'):
        from models.vagabond_model import CustomPolicy
        return CustomPolicy
    else:
        raise Exception(f'No model architectures found for {env_name}')

