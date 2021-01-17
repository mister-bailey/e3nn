import torch.distributed as dist
import torch.multiprocessing as mp
from cache_file import *
import traceback
import os

@cached_picklesjar(os.path.join(os.path.dirname(__file__), 'temp_cache'), ext='pickle', temp_ext='temp')
def test(a,b,c,d,e):
    x = a**1 + b**2 + c**3 + d**4 + e**5
    return x

def main():
    # process
    
    num_gpus = 2
    ##################MAIN##################
    print(f"Spawning {num_gpus} processes...", flush=True)
    worker_pool = mp.spawn(aux_train, (num_gpus,), nprocs=num_gpus, join=True)#, start_method='spawn')
    input("Press enter to exit...")
    
def aux_train(rank, world_size):
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'  
        print(f"Rank {rank} starting up...", flush=True)
        
        #device = f"cuda:{rank}"
        #torch.cuda.set_device(rank)
        
        #print("****AAAAAAAA******", flush=True)
        print(f"{rank}: init_process_group", flush=True)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        #print("****BBBBBBBB******", flush=True)
        
        print(f"{rank}: dist.barrier", flush=True)
        dist.barrier()
        
        print(f"Rank {rank}: round 1...", flush=True)
        dist.barrier()
        for x in range(-15*rank,20-5*rank):
            print(f"{rank}A: ", end="")# test({x}) = ({test(*(range(x,x+5)))})", flush=True)
            t = test(*(range(x,x+5)))
            print(f"{rank}B: ", end="")# test({x+3}) = ({test(*(range(x+2,x+7)))})", flush=True)
            tt = test(*(range(x+2,x+7)))

        
    except Exception as e:
        print(e)


                


if __name__ == '__main__':
    mp.freeze_support()
    main()