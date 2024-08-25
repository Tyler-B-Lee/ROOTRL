

def get_environment(env_name):
    try:
        from envs.algo_training import RootEnv
        return RootEnv
        if "Base" in env_name:
            from envs.base_training import BaseEnv
            return BaseEnv
        elif "Algo" in env_name:
            from envs.algo_training import RootEnv
            return RootEnv
        else:
            raise Exception(f'No environment found for {env_name}')
    except SyntaxError as e:
        print(e)
        raise Exception(f'Syntax Error for {env_name}!')
    except:
        raise Exception(f'Install the environment first using: \nbash scripts/install_env.sh {env_name}\nAlso ensure the environment is added to /utils/register.py')
    

def get_network_arch(env_name):
    if "Marquise" in env_name:
        from models.marquise_model import CustomPolicy
        return CustomPolicy
    if "Eyrie" in env_name:
        from models.eyrie_model import CustomPolicy
        return CustomPolicy
    if "Alliance" in env_name:
        from models.alliance_model import CustomPolicy
        return CustomPolicy
    if "Vagabond" in env_name:
        from models.vagabond_model import CustomPolicy
        return CustomPolicy
    else:
        raise Exception(f'No model architectures found for {env_name}')

