import argparse
import logging
import os

from algorithms import get_algorithm
from algorithms import show_agent_play
from algorithms import evaluate
from environment import get_env
from networks import get_actor

__log = logging.getLogger('main')


def main(args):
    __log.setLevel(level=getattr(logging, args.log_level))
    os.makedirs(args.save_dir, exist_ok=True)

    env = get_env(args.env)
    actor = get_actor(args.actor,
                      input_shapes=[[None, env.state_size]],
                      output_shapes=[[None, env.action_size]],
                      load_path=args.ckpt_actor)

    if args.train:
        Algorithm = get_algorithm(args.algorithm)
        algorithm = Algorithm(env=env, actor=actor)
        algorithm.run(**vars(args))

    if args.eval:
        evaluate(env, actor, n_eval_episode=args.n_eval_episode)

    if args.play:
        show_agent_play(env, actor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Reacher',
                        choices=['Reacher', 'MountainCarContinuous-v0'])
    parser.add_argument('--log_level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

    parser.add_argument('--algorithm', default='DDPG', choices=['DDPG'])

    parser.add_argument('--actor', default='DDPGActor', choices=['DDPGActor'])
    parser.add_argument('--critic', default='DDPGCritic', choices=['DDPGCritic'])

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--play', action='store_true', help='The agent play show.')

    parser.add_argument('--save_dir', default='logs')
    parser.add_argument('--ckpt_actor', default='')
    parser.add_argument('--ckpt_critic', default='')

    parser.add_argument('-e', '--n_episode', type=int, default=1, help='Number of episodes to run.')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('-ee', '--n_eval_episode', type=int, default=1, help='Number of episodes to run.')

    args = parser.parse_args()

    __log.info("\n%s\n" % str(args))

    main(args)
