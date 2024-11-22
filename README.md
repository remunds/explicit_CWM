## Docker 

### Development Setup

1. build container
 ```bash
docker build -t dreamerv3 .
```
2. start container with GPU(s) and current dir as working dir
```bash
docker run -td --gpus '"device=5"' -v ~/logdir/docker:/logdir --volume "$(pwd)":/app/ dreamerv3
```
3. connect to container (e.g. using vscode)
4. observe training within container
```bash
tensorboard --logdir /logdir
```

### Directly Run training
1. build container
 ```bash
docker build -t dreamerv3 .
```
2. run training
```bash
docker run -it --rm --gpus all -v ~/logdir/docker:/logdir img sh -c 'ldconfig; sh dreamerv3/embodied/scripts/xvfb_run.sh python dreamerv3/dreamerv3/main.py --logdir "/logdir/{timestamp}" --configs atari --task atari_pong'
```
3. observe training
```bash
tensorboard --logdir /logdir/docker
```