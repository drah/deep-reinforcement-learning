import argparse
import logging

from environment import get_env
from network import get_net

__log = logging.getLogger('main')

def main(args):
    __log.setLevel(level=getattr(logging, args.log_level))

    env = get_env(args.env)
    actor = get_net(args.actor)

    if args.train:
        pass
    
    if args.show_agent_play:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Reacher', choices=['Reacher'])
    parser.add_argument('--log_level', default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

    parser.add_argument('--actor', default='SimpleFC', choices=['SimpleFC'])
    parser.add_argument('--critic', default='SimpleFC', choices=['SimpleFC'])

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--show_agent_play', action='store_true')

    parser.add_argument('--save_dir', default='logs')
    parser.add_argument('--ckpt', default='')
    args = parser.parse_known_args()

    main(args[0])