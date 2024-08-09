

def get_environment(env_name):
    try:
        if env_name in ('MarquiseMainBase'):
            from envs.marquise_base_training import MarquiseMainBaseEnv
            return MarquiseMainBaseEnv
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
    else:
        raise Exception(f'No model architectures found for {env_name}')

