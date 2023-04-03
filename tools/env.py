import os
import re
import torch
import torch.distributed as dist


def init_dist(args):
    """Initialize distributed computing environmen
    t."""
    args.ngpus_per_node = torch.cuda.device_count()

    if args.launcher == 'pytorch':
        _init_dist_pytorch(args)
    elif args.launcher == 'mpi':
        _init_dist_mpi(args)
    elif args.launcher == 'slurm':
        _init_dist_slurm(args)
    else:
        raise ValueError('Invalid launcher type: {}'.format(args.launcher))


def _init_dist_pytorch(args, **kwargs):
    """Set up environment."""
    # TODO: use local_rank instead of rank % num_gpus
    args.rank = args.rank * args.ngpus_per_node + args.gpu
    args.world_size = args.world_size
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(args.gpu)
    print(f"{args.dist_url}, ws:{args.world_size}, rank:{args.rank}")

    if args.rank % args.ngpus_per_node == 0:
        args.log = True
    else:
        args.log = False


def _init_dist_slurm(args, port=23333, **kwargs):
    """Set up slurm environment."""
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    node_list = str(os.environ['SLURM_NODELIST'])
    num_gpus = torch.cuda.device_count()

    node_parts = re.findall('[0-9]+', node_list)
    host_ip = '{}.{}.{}.{}'.format(node_parts[1], node_parts[2], node_parts[3], node_parts[4])
    init_method = 'tcp://{}:{}'.format(host_ip, port)

    print(f"{init_method}, rank: {rank}, local rank: {local_rank}")

    dist.init_process_group(backend=args.dist_backend,
                            init_method=init_method,
                            world_size=world_size,
                            rank=rank)

    torch.cuda.set_device(local_rank)
    args.rank = rank
    args.world_size = world_size
    args.ngpus_per_node = num_gpus
    args.gpu = local_rank

    if args.rank == 0:
        args.log = True
    else:
        args.log = False


def _init_dist_mpi(backend, **kwargs):
    raise NotImplementedError
