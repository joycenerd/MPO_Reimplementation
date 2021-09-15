import argparse
import gym
from my_mpo import MPO
import dm_control2gym
import os
import json


def main():
    parser = argparse.ArgumentParser( formatter_class=argparse.ArgumentDefaultsHelpFormatter,description='Implementation of MPO on gym environments')
    parser.add_argument('--device', type=str, default='cpu')    # cpu is faster!

    ## set up domain & task argument
    parser.add_argument('--domain', type=str, default='hopper', help='gym environment')
    parser.add_argument('--task', type=str, default='stand', help='gym environment')
    
    # papar parameters
    parser.add_argument('--dual_constraint', type=float, default=0.1, help='hard constraint of the dual formulation in the E-step')
    parser.add_argument('--kl_mean_constraint', type=float, default=0.1, help='hard constraint of the mean in the M-step')
    parser.add_argument('--kl_var_constraint', type=float, default=0.0001, help='hard constraint of the covariance in the M-step')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor used in Policy Evaluation')
    parser.add_argument('--q_loss_type', type=str, default='mse', help='discount factor used in Policy Evaluation')

    parser.add_argument('--alpha_mean_scale', type=float, default=1.0, help='scaling factor of the lagrangian multiplier in the M-step')
    parser.add_argument('--alpha_var_scale', type=float, default=100.0, help='scaling factor of the lagrangian multiplier in the M-step')
    parser.add_argument('--alpha_mean_max', type=float, default=0.1, help='maximum value of the lagrangian multiplier in the M-step')
    parser.add_argument('--alpha_var_max', type=float, default=10.0, help='maximum value of the lagrangian multiplier in the M-step')
    parser.add_argument('--sample_episode_num', type=int, default=50, help='number of episodes to learn')
    parser.add_argument('--sample_episode_maxstep', type=int, default=300, help='maximum sample steps of an episode')
    parser.add_argument('--sample_action_num', type=int, default=64, help='number of sampled actions')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--iteration_num', type=int, default=1000, help='number of iteration to learn')
    parser.add_argument('--episode_rerun_num', type=int, default=3, help='number of reruns of sampled episode')
    parser.add_argument('--mstep_iteration_num', type=int, default=5, help='the number of iterations of the M-Step')
    parser.add_argument('--evaluate_period', type=int, default=10, help='periode of evaluation')
    parser.add_argument('--evaluate_episode_num', type=int, default=1, help='number of episodes to evaluate')
    parser.add_argument('--evaluate_episode_maxstep', type=int, default=300, help='maximum evaluate steps of an episode')
    parser.add_argument('--log_dir', type=str, default="hopper_p1_dual1_mse", help='log directory')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--load', type=str, default=None, help='load path')
    args = parser.parse_args()

    # wrap DM control suite to gym environment
    env = dm_control2gym.make(domain_name=args.domain, task_name=args.task)

    model = MPO( env, args)
    if args.load is not None:
        model.load_model(args.load)

    # only train in the continuous environments
    if(env.action_space.dtype == 'float32'):
        
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)

        # write hyperparameters
        with open(os.path.join(args.log_dir, 'setting.txt'), 'a') as f:
            json.dump(args.__dict__, f, indent=2)

        model.train( args.iteration_num, args.log_dir, render=args.render)

    env.close()

if __name__ == '__main__':
    main()
