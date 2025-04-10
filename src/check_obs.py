import jericho
import hashlib

# Load the Zork z5 file (adjust the file path as needed)
env = jericho.FrotzEnv('z-machine-games-master/jericho-game-suite/moonlit.z5')

# Define your keyword lists



def get_scene_id(observation):
    # Optionally, process the observation to filter or modify its content.
    # For example, you can lowercase the text.
    text = observation.lower()

    # You might decide to not create a scene id for observations that contain bad words.
    if any(bad in text for bad in badword_list):
        return None  # or handle these cases differently

    # Optionally, you could boost the "importance" of scenes with good words.
    # In this simple example, we just generate a hash regardless.
    scene_hash = hashlib.md5(observation.encode('utf-8')).hexdigest()
    return scene_hash

# Initialize a set to keep track of visited scenes
visited_scenes = set()

# Reset the environment to start the game, which returns the initial observation
obs, info = env.reset()
print("Observation:", obs)

# Check for new scene at the start of the game
scene_id = get_scene_id(obs)
if scene_id and scene_id not in visited_scenes:
    print("New scene encountered at game start!")
    visited_scenes.add(scene_id)
    aux_reward = 0.1  # set your bonus reward here
    print("Auxiliary reward applied:", aux_reward)

done = False
while not done:
    # Get command input from the user
    action = input("Enter your command: ")
    
    # Execute the command in the game environment
    obs, reward, done, info = env.step(action)
    
    # Check if this is a new scene using the scene id function
    scene_id = get_scene_id(obs)
    if scene_id and scene_id not in visited_scenes:
        print("New scene encountered!")
        visited_scenes.add(scene_id)
        aux_reward = 0.1  # define your auxiliary reward bonus
        reward += aux_reward
        print("Auxiliary reward applied:", aux_reward)
    
    # Print the resulting observation and reward
    print("Observation:", obs)
    print("Reward:", reward)
