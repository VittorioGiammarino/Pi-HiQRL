# pointmaze-medium-navigate-v0 (Pi-HiQRL)
python main.py --env_name=pointmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True 

# pointmaze-large-navigate-v0 (Pi-HiQRL)
python main.py --env_name=pointmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True 

# pointmaze-giant-navigate-v0 (Pi-HiQRL)
python main.py --env_name=pointmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.discount=0.995 

# pointmaze-teleport-navigate-v0 (Pi-HiQRL)
python main.py --env_name=pointmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True

# pointmaze-medium-stitch-v0 (Pi-HiQRL)
python main.py --env_name=pointmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5

# pointmaze-large-stitch-v0 (Pi-HiQRL)
python main.py --env_name=pointmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5

# pointmaze-giant-stitch-v0 (Pi-HiQRL)
python main.py --env_name=pointmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 

# pointmaze-teleport-stitch-v0 (Pi-HiQRL)
python main.py --env_name=pointmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5

# antmaze-medium-navigate-v0 (Pi-HiQRL)
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True

# antmaze-large-navigate-v0 (Pi-HiQRL)
python main.py --env_name=antmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True

# antmaze-giant-navigate-v0 (Pi-HiQRL)
python main.py --env_name=antmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.discount=0.995 

# antmaze-teleport-navigate-v0 (Pi-HiQRL)
python main.py --env_name=antmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True

# antmaze-medium-stitch-v0 (Pi-HiQRL)
python main.py --env_name=antmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 

# antmaze-large-stitch-v0 (Pi-HiQRL)
python main.py --env_name=antmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 

# antmaze-giant-stitch-v0 (Pi-HiQRL)
python main.py --env_name=antmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 

# antmaze-teleport-stitch-v0 (Pi-HiQRL)
python main.py --env_name=antmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 

# antmaze-medium-explore-v0 (Pi-HiQRL)
python main.py --env_name=antmaze-medium-explore-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.high_alpha=10.0 --agent.low_alpha=10.0

# antmaze-large-explore-v0 (Pi-HiQRL)
python main.py --env_name=antmaze-large-explore-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.high_alpha=10.0 --agent.low_alpha=10.0

# antmaze-teleport-explore-v0 (Pi-HiQRL)
python main.py --env_name=antmaze-teleport-explore-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.high_alpha=10.0 --agent.low_alpha=10.0

# humanoidmaze-medium-navigate-v0 (Pi-HiQRL)
python main.py --env_name=humanoidmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.discount=0.995 --agent.subgoal_steps=100

# humanoidmaze-large-navigate-v0 (Pi-HiQRL)
python main.py --env_name=humanoidmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.discount=0.995 --agent.subgoal_steps=100

# humanoidmaze-giant-navigate-v0 (Pi-HiQRL)
python main.py --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.discount=0.995 --agent.subgoal_steps=100

# humanoidmaze-medium-stitch-v0 (Pi-HiQRL)
python main.py --env_name=humanoidmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 --agent.subgoal_steps=100

# humanoidmaze-large-stitch-v0 (Pi-HiQRL)
python main.py --env_name=humanoidmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 --agent.subgoal_steps=100

# humanoidmaze-giant-stitch-v0 (Pi-HiQRL)
python main.py --env_name=humanoidmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/pi_hiqrl.py --agent.high_level_coordinates_only=True --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.discount=0.995 --agent.subgoal_steps=100
