import argparse
from reinforcepy.environments.interactive_pygame_wrapper import PygameWrapper
from reinforcepy.environments.environment_recorder import EnvironmentRecorder
import json
from reinforcepy.environments import ALEEnvironment
from reinforcepy.environments.ALE.pygame_key_mapping import generate_keymapping, print_key_mapping


def main(rom, rom_args, num_episodes=1, max_fps=60., record_episodes=False, record_dir=None, resize_scale=2.0, **kwargs):
    # create env
    if 'display_screen' in rom_args:
        del rom_args['display_screen']
    if 'rom' in rom_args:
        del rom_args['rom']
    # screen is displayed through pygame don't display it
    environment = ALEEnvironment(rom=rom, **rom_args, display_screen=False)
    key_mapping = generate_keymapping(environment.get_legal_actions())

    # let people know the key maping
    print_key_mapping()

    # if recording
    if record_episodes:
        environment = EnvironmentRecorder(environment, record_dir)

    pygame_wrapper = PygameWrapper(environment, keys_mouse_to_action=key_mapping,
                                   max_fps=max_fps / rom_args['skip_frame'], resize_screen=resize_scale)
    pygame_wrapper.run_keyboard_episode(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play ALE as the agent would see it and optionally record your actions for supervised training')
    parser.add_argument('rom', type=str, help='The rom file to use include file path')
    parser.add_argument('config_file', type=str, help='The config file to use for rom args')
    parser.add_argument('num_episodes', type=int, default=1, help='Number of episodes to play')
    parser.add_argument('-resize_scale', type=float, default=2.0, help='Resize scale to make the play screen larger. Episodes will still be recorded'
                        'in their original resolution.')
    parser.add_argument('-record_episodes', type=bool, default=False, help='Record epsisodes. Default False')
    parser.add_argument('-record_dir', type=str, help='Recording directory for epsiodes, must have record_episodes True')
    parser.add_argument('-max_fps', type=float, default=60.0, help='The max FPS to play at. Default 60.0')
    args = parser.parse_args()

    CONFIG = json.load(open(args.config_file))
    main(args.rom, **CONFIG, max_fps=args.max_fps, num_episodes=args.num_episodes, record_episodes=args.record_episodes, record_dir=args.record_dir,
         resize_scale=args.resize_scale)
