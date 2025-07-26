export DISPLAY=:0
xhost +
docker run -it --rm --runtime nvidia --shm-size=1g -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd)/src:/src --privileged  -e AWS_REGION -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY --device=/dev/video0:/dev/video0 --env="DISPLAY=$DISPLAY" --network host devio2025-sapporo-duck-factory:latest


