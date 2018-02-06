## TIPs 
I'll briefly introduce how to run horovod for my work.

step1.
prepare clusters H/W with Volta GPU with NVLINK2 :) network, infiniband (non-blocking fat-tree) 
install ubuntu 16.04 on baremetal
install nvidia driver, ofed for each node   ( plz read ubuntu install guide and follow the intruction)
configure network storage ( NFS, RAID, Cache) 
configure ldap, slurm + module environment same as HPC cluster

step2. 
register ngc.nvidia.com for nvcr.io repository. 
 

step3. module-environment configure 
install minium required tools docker-ce, nvidia-docker, singulariry, openmpi-3.0.0 on module-environment for baremetal
for version control, I recommend module-environment

step4. docker pull & run  some repositories to be familiar with docker command 
such as ubuntu  nvidia/cuda  nvcr.io/nvidia/tensorflow 

useful docker commands are 
```
docker images
docker pull
docker run 
nvidia-docker run
docker exec 
docker save
docker ps 
docker rm
docker kill
docker rmi 
docker commit
```
 
step5. docker build nvcr.io to private repository to bypass APIkey
singularity build have some issue with nvcr.io API key. 
private docker repositories is also good solution.
```docker build -t mycuda:01 -f ./Dockerfile . 
```

I recommend to prepare docker build server and apt-cache servers to speed up these thing 
prepare apt-cache server for speed up 

step6. get custom Dockerfiles for horovod 
use this git url.  https://github.com/yhgon/horovod-tf-uber 
check and review it. you can switch python version

step7. build dockerfiles 
```docker build -t hvd:01 -f ./Dockerfile.horovod . 
docker save hvd:01 -output hvd-01.tar
```
step8. docker run test 
```nvidia-docker run -ti hvd:01 bash ```

step9. convert docker image to singularity 
```docker run -v/your_localtion:/output /singularityware/docker2singularity hvd:01```

step10. mpirun on baremetal for each docker in multinode
simple command would  be  

```srun -N8 -p gpu_node --comment=docker bash  
module load singularity openmpi
mpirun -n 32  singularity shell --nv a.img  python train.py 
```
step11. enjoy horovod

step12. use example to be familar with horovod optimizer 
