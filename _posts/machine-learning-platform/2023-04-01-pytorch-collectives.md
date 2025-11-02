---
layout: post
title:  Pytorch Collective Communications
date:   2023-04-01 16:40:16
description: Pytorch collective communication 이해하기
tags: formatting links
categories: sample-posts
taxonomy: pytorch, nccl
---

- https://blog.roboflow.com/collective-communication-distributed-systems-pytorch/
- https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf

## 초기화

```
import os
from typing import Callable

import torch
import torch.distributed as dist

def init_process(rank: int, world_size: int, fn: Callable([int, int], None], backend="gloo")):
    """Initialize the distributed environment."""
    os.environ["MASTER ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend, rank=rank, world_size=world_size, fn(rnk, size))
```

이 코드를 실행하기 위해서는 두가지 전략이 있습니다.

### 1. python multiprocessing을 이용한 process 생성

```
import torch.multiprocessing as mp

def func(rank: int, size: int):
    # each process will call this function
    continue

if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, func))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

### 2. torch distributed를 이용한 multiprocess 동작

```
python -m torch.distributed.launch --nnodes 1 --proc_per_node 4 main.py
```

`main.py`
```
if __name__ == "__main__":
    init_process(
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        fn = func
    )
```

어떤 것을 사용해도 되지만, 만약 2개 이상의 서버를 활용할 수 있는 분산 환경이라면 두번째 방법을 이용하는 것을 권장합니다.
#TODO: 분산에 맞게 옵션을 추가할 것


## Collective Communications

### Reduce

### All Reduce

### Scatter

### Gather

### All Gather

reduce-scatter의 두 동작으로 나눠서 할 수도 있다.
이에 대해서는 다음에 더 자세히 알아보겠습니다

## broadcast