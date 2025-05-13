
# Build the Docker image
docker build -t lms_video_engagement:1.00 .

# List Docker images
docker image ls

# Remove dangling images
docker image rm -f $(docker images -f dangling=true -q)

# Run the Docker container
docker run -p 80:80 lms_video_engagement:1.00