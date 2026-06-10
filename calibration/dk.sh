#!/bin/bash
CONTAINER_NAME="ros_data_processor"
IMAGE_NAME="ros_data_processor:latest"
INSTALL_SOURCE="default"

start() {
    if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
            echo "Container exists but is stopped. Restarting the container..."
            docker start $CONTAINER_NAME
            docker exec -it $CONTAINER_NAME /bin/bash
        else
            echo "Building and starting the container with source: $INSTALL_SOURCE..."
            docker build --build-arg INSTALL_SOURCE=$INSTALL_SOURCE -t $IMAGE_NAME .
            # 注释掉的卷挂载选项：
            # -v ./src:/app/src
            # -v ./setup.sh:/app/setup.sh
            # -v ./bagdata:/app/bagdata
            # -v ./video:/app/video
            docker run -it \
                --name $CONTAINER_NAME \
                -v /tmp/.X11-unix:/tmp/.X11-unix \
                -e DISPLAY=unix$DISPLAY \
                -e WORKDIR="/app" \
                -e PATH="/opt/miniconda/envs/datahub/bin:/opt/miniconda/bin:$PATH" \
                -e PYTHONPATH="/app:$PYTHONPATH" \
                -v ./data:/app/data \
                -v ./output:/app/output \
                -v ./src:/app/src \
                -v ./scripts:/app/scripts \
                -v ./docs:/app/docs \
                -v ./setup.sh:/app/setup.sh \
                --net=host \
                --privileged \
                $IMAGE_NAME /app/setup.sh
        fi
    else
        echo "Container already running. Attaching to the container..."
        docker exec -it $CONTAINER_NAME /bin/bash
    fi
}

stop() {
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "Stopping and removing the container..."
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
    elif [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        echo "Container is already stopped. Removing the container..."
        docker rm $CONTAINER_NAME
    else
        echo "Warning: Container not found."
    fi
}

restart() {
    stop
    start
}

help() {
    echo "Usage: $0 {start|stop|restart|--help}"
    echo "start   - Build and start the container if it doesn't exist, restart if it is stopped, or attach to it if it is running."
    echo "stop    - Stop and remove the container if it exists."
    echo "restart - Stop and then start the container."
    echo "--help  - Display this help message."
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        start|stop|restart|--help)
            COMMAND=$1
            ;;
        --install_source=*)
            INSTALL_SOURCE="${1#*=}"
            ;;
        *)
            echo "Invalid option: $1"
            help
            exit 1
            ;;
    esac
    shift
done

case "$COMMAND" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    help|--help)
        help
        ;;
    *)
        echo "Invalid command: $COMMAND"
        help
        exit 1
        ;;
esac
