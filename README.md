## Docker 

### Development Setup

1. build container
 ```bash
cd dreamerv3 && docker build -f dreamerv3/Dockerfile -t dreamerv3 .
```
2. start container with GPU(s) and current dir as working dir
```bash
docker run -td --gpus '"device=5"' -v ~/logdir/docker:/logdir --volume "$(pwd)":/app/ dreamerv3
```
3. connect to container (e.g. using vscode)

### Directly Run training
1. build container
 ```bash
cd dreamerv3 && docker build -f dreamerv3/Dockerfile -t dreamerv3 .
```
2. 
```bash
docker run -it --rm --gpus all -v ~/logdir/docker:/logdir img sh -c 'ldconfig; sh dreamerv3/embodied/scripts/xvfb_run.sh python dreamerv3/dreamerv3/main.py --logdir "/logdir/{timestamp}" --configs atari --task atari_pong'
```