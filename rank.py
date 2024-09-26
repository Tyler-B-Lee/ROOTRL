import argparse
import random
import trueskill as ts
import os

import logging
log_level = logging.INFO
  
logging.basicConfig(
  filename='logs/log.txt',
  format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
  filemode='w',
  level=log_level
)
logger = logging.getLogger(__name__)

from utils.agents import Agent
import config
  
# Initialize TrueSkill environment
ts_env = ts.TrueSkill(mu=25.0, sigma=8.333)

# Define a class to track faction-specific models and their ratings.
class TournamentAgent(Agent):
  """
  TournamentAgent class represents an agent participating in a tournament with a specific faction and rating.
  Attributes:
    name (str): The name of the agent.
    action_space_size (int): The size of the action space for the agent.
    faction (str): The faction to which the agent belongs.
    model: The model used by the agent (default is None).
    rating: The rating of the agent, initialized with a default rating.
  Methods:
    __repr__():
      Returns a string representation of the TournamentAgent, including its name, faction, rating, and uncertainty.
  """
  def __init__(self, name:str, faction_id:int, action_space_size:int, model=None, rating:dict|None=None):
    super().__init__(name, faction_id, action_space_size, model)
    if rating is not None:
      self.rating = ts_env.Rating(mu=rating['mu'], sigma=rating['sigma'])
    else:
      self.rating = ts_env.create_rating()  # Initialize with default rating
  
  def __repr__(self):
    mu, sigma = self.rating.mu, self.rating.sigma
    return f'{self.name:<60} | {mu:<10.3f} | {sigma:<10.3f}'
  
# Function to organize a 4-player match (one per faction)
def play_match(models:list[TournamentAgent], env, args):
  """
  Simulates a match between four models (one per faction).
  In actual use, this would be replaced by the game logic.
  
  :param models: List of four Model objects (one per faction).
  :param env: Environment object for the game.
  :param agents: List of all agents.
  :param args: argparse.Namespace, arguments from the command line.
  :return: Rankings of models in order of 1st to 4th place.
  """
  # setup for the match
  model_players = []
  for i,model in enumerate(models):
    logger.debug(f'Player {i+1} = {model.name}')
    model.points = 0  # Reset points for each model
    if 'rules' not in model.name.lower():
      model_players.append(i)

  env.model_players = model_players
  obs, _ = env.reset(seed=args.seed)

  terminated = truncated = False
  while not (terminated or truncated):

    current_player = models[env.current_player_num]
    env.render()
    logger.debug(f'\nCurrent player name: {current_player.name}')

    if 'rules' in current_player.name.lower():
      logger.debug(f'\n{current_player.name} model choices')
      action = current_player.choose_action(env, choose_best_action = False, mask_invalid_actions = True)
    else:
      logger.debug(f'\n{current_player.name} model choices')
      action = current_player.choose_action(env, choose_best_action = args.best, mask_invalid_actions = True)

    obs, step_reward_list, terminated, truncated, _ = env.step(action)

    logger.debug(f"{step_reward_list=}")
    for player in models:
      logger.debug(f"{player.name=} {player.target_id=}")
      player.points += player.multiplier * step_reward_list[player.target_id]
  
  env.render()
  logger.debug(f'\nMatch over! Final scores: {[model.points for model in models]}')

  # it is a declared a draw if the game is truncated
  if truncated:
    return [0,0,0,0]
  
  # Find out the winner(s), everyone else is tied for 2nd
  rankings = [1,1,1,1]
  for i,model in enumerate(models):
    if model.points > 0.6:
      rankings[i] = 0

  return rankings

# Function to update TrueSkill ratings after a match
def update_trueskill_ratings(models:list[TournamentAgent], rankings:list[int]):
  """
  Updates TrueSkill ratings based on match results.
  
  :param models: List of Model objects for all players in the match.
  :param rankings: List of rankings for each player in the match.
  """
  # Extract current ratings
  ratings = [(model.rating,) for model in models]
  
  # Update ratings using their finishing ranks (1st = 0, 2nd = 1, etc.)
  new_ratings = ts_env.rate(ratings, ranks=rankings)
  
  # Assign new ratings to the models
  for i, model in enumerate(models):
    model.rating = new_ratings[i][0] # note tuples are returned, so we need to extract the first element


# Function to select one model per faction
def select_models(models_dict:dict[str, list[TournamentAgent]]):
  """
  Selects one model per faction for the match.
  
  :param models: Dict of all models, with keys as faction names.
  :return: List of one model from each faction.
  """
  selected_models = []
  
  for faction in models_dict.keys():
    # Randomly select one model per faction
    selected_models.append(random.choice(models_dict[faction]))
  
  return selected_models

# Function to run multiple test matches
def run_tournament(models_dict, env, args, num_matches=10):
  """
  Runs a tournament with multiple matches to update ratings.
  
  :param models: List of all models.
  :param env: Environment object for the game.
  :param args: argparse.Namespace, arguments from the command line.
  :param num_matches: Number of matches to play (default is 10).
  """
  logger.info(f'\nPlaying {num_matches} games...')
  for g in range(num_matches):
    # Select one model per faction
    selected_models = select_models(models_dict)
    
    # Play a match and get rankings
    rankings = play_match(selected_models, env, args)
    
    # Update ratings based on match outcome
    update_trueskill_ratings(selected_models, rankings)

    logger.info(f'Game {g+1} Done - New Ratings:')
    for model in selected_models:
      logger.info(model)

def main(args):
  """Main function for the rank.py script.
  This function is used to create a ranking system for the different models
  already created for the game. It will play a number of games between the
  different models and then save the results to a file.

  Args:
    args: argparse.Namespace, arguments from the command line
  """
  # Set logging level from args
  if args.debug:
    logger.setLevel(logging.DEBUG)
    logger.debug("Logging level set at DEBUG or lower!")
  else:
    logger.setLevel(logging.INFO)
    logger.info("Logging level set at INFO or lower!")

  logger.info("Beginning Imports...")
  import json
  from stable_baselines3.common.utils import set_random_seed
  from utils.files import load_model
  from register import get_environment

  set_random_seed(args.seed)

  # load saved model ratings if they exist
  if os.path.exists('ratings.json'):
    with open('ratings.json', 'r') as f:
      ratings = json.load(f)
      logger.info("Loaded ratings from file")
  else:
    ratings = {'Marquise': {}, 'Eyrie': {}, 'Alliance': {}, 'Vagabond': {}}
    logger.info("No ratings file found, starting with default ratings")

  agents = {'Marquise': [], 'Eyrie': [], 'Alliance': [], 'Vagabond': []}
  logger.info("\n> Loading models...")
  # load rules agents first
  for faction_id, faction, aspace_size in zip([0,1,2,3], agents.keys(), config.ACTION_SPACE_SIZES):
    rating_to_load = ratings[faction].get(f'{faction}_Rules', None)
    rules_agent = TournamentAgent(f'{faction}_Rules', faction_id, aspace_size, model=None, rating=rating_to_load)
    agents[faction].append(rules_agent)

  # load the model agents next from each folder in the zoo
  for folder_name in os.listdir(config.MODELDIR):
    for faction_id, faction, aspace_size in zip([0,1,2,3], agents.keys(), config.ACTION_SPACE_SIZES):
      if faction in folder_name:
        for model_name in os.listdir(os.path.join(config.MODELDIR, folder_name)):
          if 'best_model' in model_name:
            continue

          model = load_model(folder_name, model_name, None)
          rating_to_load = ratings[faction].get(f'{folder_name}_{model_name}', None)

          model_agent = TournamentAgent(f'{folder_name}_{model_name}', faction_id, aspace_size, model, rating_to_load)
          agents[faction].append(model_agent)
          logger.debug(f"Loaded model '{model_name}' for {faction} from {folder_name}")
  
  # make environment
  env = get_environment(args.env_rules)(rules_type = args.env_rules, model_players = [0], verbose = args.verbose, manual = args.manual)

  # Run the tournament
  run_tournament(agents, env, args, num_matches=args.games)

  # Write final ratings to a file
  for faction in agents.keys():
    for agent in agents[faction]:
      ratings[faction][agent.name] = {'mu': agent.rating.mu, 'sigma': agent.rating.sigma}
  with open('ratings.json', 'w') as f:
    json.dump(ratings, f, indent=2)

  env.close()

# py rank.py

def cli() -> None:
  """Handles argument extraction from CLI and passing to main().
  Note that a separate function is used rather than in __name__ == '__main__'
  to allow unit testing of cli().
  """
  # Setup argparse to show defaults on help
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=formatter_class)

  parser.add_argument("--best", "-b", action = 'store_true', default = False
                , help="Make AI agents choose the best move (rather than sampling)")
  parser.add_argument("--games", "-g", type = int, default = 10
                , help="Number of games to play")
  parser.add_argument("--debug", "-d",  action = 'store_true', default = False
            , help="Show logs to debug level")
  parser.add_argument("--verbose", "-v",  action = 'store_true', default = False
            , help="Show observation on debug logging")
  parser.add_argument("--manual", "-m",  action = 'store_true', default = False
            , help="Manual update of the game state on step")
  parser.add_argument("--model_types", "-mt", type = str, default = 'MarquiseMain'
            , help="Which agent models to load for each faction? 'None' for no model")
  parser.add_argument("--env_rules", "-e",  type = str, default = 'Algo'
            , help="Which Rules type for this env?")
  parser.add_argument("--seed", "-s",  type = int, default = 17
            , help="Random seed")

  # Extract args
  args = parser.parse_args()

  # Enter main
  main(args)
  return


if __name__ == '__main__':
  cli()