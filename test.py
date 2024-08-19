# docker-compose exec app python3 test.py -d -g 1 -a base base human -e butterfly 

import argparse

def main(args):
  import logging

  if args.debug:
    log_level = logging.DEBUG
  else:
    log_level = logging.INFO
  
  logging.basicConfig(
    filename='logs/log.txt',
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
    filemode='w',
    level=log_level
  )
  logger = logging.getLogger(__name__)
  logger.debug("Logging level set at DEBUG or lower!")
  logger.info("Beginning Imports...")

  from stable_baselines3.common.utils import set_random_seed

  from utils.files import load_model, write_results
  from register import get_environment
  from utils.agents import Agent

  import config
    
  #make environment
  model_players = [i for i,a in enumerate(args.agents) if (a != "rules")]
  env = get_environment(args.env_name)(env_name = args.env_name, model_players = model_players, verbose = args.verbose, manual = args.manual)
  set_random_seed(args.seed)

  total_rewards = {}

  # if args.recommend:
  #   ppo_model = load_model(env, 'best_model.zip')
  #   ppo_agent = Agent('best_model', ppo_model)
  # else:
  #   ppo_agent = None
  ENV_NAMES = ["MarquiseMainBase", "EyrieMainBase", "AllianceMainBase", "VagabondMainBase"]

  agents = []

  #load the agents
  if len(args.agents) != env.n_players:
    raise Exception(f'{len(args.agents)} players specified but this is a {env.n_players} player game!')

  for agent, aspace_size, env_name in zip(args.agents, config.ACTION_SPACE_SIZES, ENV_NAMES):
    if agent == 'human':
      agent_obj = Agent('human', aspace_size)
    elif agent == 'rules':
      agent_obj = Agent('rules', aspace_size)
    elif agent == 'base':
      base_model = load_model(env_name, 'base.zip', None)
      agent_obj = Agent('base', aspace_size, base_model)   
    else:
      ppo_model = load_model(env_name, f'{agent}.zip', None)
      agent_obj = Agent(agent, aspace_size, ppo_model)

    agents.append(agent_obj)
    total_rewards[agent_obj.id] = 0
  
  #play games
  logger.info(f'\nPlaying {args.games} games...')
  for game in range(args.games):
    players = agents[:]


    obs, _ = env.reset(seed=args.seed)
    terminated = truncated = False
    
    for i, p in enumerate(players):
      logger.debug(f'Player {i+1} = {p.name}')

    while not (terminated or truncated):

      current_player = players[env.current_player_num]
      env.render()
      logger.debug(f'\nCurrent player name: {current_player.name}')

      # if args.recommend and current_player.name in ['human', 'rules']:
      #   # show recommendation from last loaded model
      #   logger.debug(f'\nRecommendation by {ppo_agent.name}:')
      #   action = ppo_agent.choose_action(env, choose_best_action = True, mask_invalid_actions = True)

      if current_player.name == 'human':
        action = input('\nPlease choose an action: ')
        try:
          # for int actions
          action = int(action)
        except:
          # for MulitDiscrete action input as list TODO
          action = eval(action)
      elif current_player.name == 'rules':
        logger.debug(f'\n{current_player.name} model choices')
        action = current_player.choose_action(env, choose_best_action = False, mask_invalid_actions = True)
      else:
        logger.debug(f'\n{current_player.name} model choices')
        action = current_player.choose_action(env, choose_best_action = args.best, mask_invalid_actions = True)

      obs, step_reward_list, terminated, truncated, _ = env.step(action)

      for r, player in zip(step_reward_list, players):
        total_rewards[player.id] += r
        player.points += r

      if args.cont:
        input('Press any key to continue')
    
    env.render()

    logger.info(f"Played {game + 1} games: {total_rewards}")

    if args.write_results:
      write_results(players, game, args.games, env.turns_taken)

    for p in players:
      p.points = 0

  env.close()
    

# py test.py -e MarquiseMainBase -a best_model rules rules rules -d

def cli() -> None:
  """Handles argument extraction from CLI and passing to main().
  Note that a separate function is used rather than in __name__ == '__main__'
  to allow unit testing of cli().
  """
  # Setup argparse to show defaults on help
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=formatter_class)

  parser.add_argument("--agents","-a", nargs = '+', type=str, default = ['human', 'human']
                , help="Player Agents (human, ppo version)")
  parser.add_argument("--best", "-b", action = 'store_true', default = False
                , help="Make AI agents choose the best move (rather than sampling)")
  parser.add_argument("--games", "-g", type = int, default = 1
                , help="Number of games to play")
  # parser.add_argument("--n_players", "-n", type = int, default = 3
  #               , help="Number of players in the game (if applicable)")
  parser.add_argument("--debug", "-d",  action = 'store_true', default = False
            , help="Show logs to debug level")
  parser.add_argument("--verbose", "-v",  action = 'store_true', default = False
            , help="Show observation on debug logging")
  parser.add_argument("--manual", "-m",  action = 'store_true', default = False
            , help="Manual update of the game state on step")
  # parser.add_argument("--randomise_players", "-r",  action = 'store_true', default = False
  #           , help="Randomise the player order")
  # parser.add_argument("--recommend", "-re",  action = 'store_true', default = False
  #           , help="Make recommendations on humans turns")
  parser.add_argument("--cont", "-c",  action = 'store_true', default = False
            , help="Pause after each turn to wait for user to continue")
  parser.add_argument("--env_name", "-e",  type = str, default = 'TicTacToe'
            , help="Which game to play?")
  parser.add_argument("--write_results", "-w",  action = 'store_true', default = False
            , help="Write results to a file?")
  parser.add_argument("--seed", "-s",  type = int, default = 17
            , help="Random seed")

  # Extract args
  args = parser.parse_args()

  # Enter main
  main(args)
  return


if __name__ == '__main__':
  cli()